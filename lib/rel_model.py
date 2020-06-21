"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

# from lib.decoder_rnn import DecoderRNN, lstm_factory, LockedDropout
from lib.lstm.decoder_rnn import DecoderRNN
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, \
    Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import math

from lib.model import EmbeddingImagenet, EdgeUpdateNetwork, GraphNetwork, MetricLearning
from lib.DiscCentroidsLoss import *
from lib.MetaEmbeddingClassifier import *
MODES = ('sgdet', 'sgcls', 'predcls')


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, classes, rel_classes, mode='sgdet',
                 embed_dim=200, hidden_dim=256, obj_dim=2048,
                 nl_obj=2, nl_edge=2, dropout_rate=0.2, order='confidence',
                 pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.nl_obj = nl_obj
        self.nl_edge = nl_edge

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.dropout_rate = dropout_rate
        self.pass_in_obj_feats_to_decoder = pass_in_obj_feats_to_decoder
        self.pass_in_obj_feats_to_edge = pass_in_obj_feats_to_edge

        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order

        # EMBEDDINGS
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])
        self.conver_fusion_feature = nn.Sequential(*[
            nn.BatchNorm1d(self.embed_dim + 128, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(self.embed_dim + 128, 4096),  # self.obj_dim +
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])
        self.decoder_lin_ = nn.Linear(4096, self.num_classes)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None):
        """
        Forward pass through the object and edge context
        :param obj_priors:
        :param obj_fmaps:
        :param im_inds:
        :param obj_labels:
        :param boxes:
        :return:
        """

        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        pos_embed = self.pos_embed(center_size(box_priors))
        # obj_pre_rep = self.conver_fusion_feature(torch.cat((obj_fmaps, obj_embed, pos_embed), 1))
        obj_pre_rep = self.conver_fusion_feature(torch.cat((obj_embed, pos_embed), 1))
        # UNSURE WHAT TO DO HERE
        if self.mode == 'predcls':
            obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))
        else:
            obj_dists2 = self.decoder_lin(obj_pre_rep)

        if self.mode == 'sgdet' and not self.training:
            # NMS here for baseline
            probs = F.softmax(obj_dists2, 1)
            nms_mask = obj_dists2.data.clone()
            nms_mask.zero_()
            for c_i in range(1, obj_dists2.size(1)):
                scores_ci = probs.data[:, c_i]
                boxes_ci = boxes_per_cls.data[:, c_i]

                keep = apply_nms(scores_ci, boxes_ci,
                                 pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                 nms_thresh=0.3)
                nms_mask[:, c_i][keep] = 1

            obj_preds = Variable(nms_mask * probs.data, volatile=True)[:, 1:].max(1)[1] + 1
        else:
            obj_preds = obj_labels if obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1

        return obj_dists2, obj_preds, obj_pre_rep


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=1, nl_edge=2, use_resnet=False, order='confidence', thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True, gnn=True,reachability=False,
                 pass_in_obj_feats_to_edge=True, rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode
        self.reachability=reachability
        self.gnn = gnn
        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.global_embedding = EmbeddingImagenet(4096)
        self.global_logist = nn.Linear(4096, 151, bias=True)  # CosineLinear(4096,150)#
        self.global_logist.weight = torch.nn.init.xavier_normal(self.global_logist.weight, gain=1.0)

        self.disc_center = DiscCentroidsLoss(self.num_rels, self.pooling_dim+256)
        self.meta_classify = MetaEmbedding_Classifier(feat_dim=self.pooling_dim+256, num_classes=self.num_rels)

        # self.global_rel_logist = nn.Linear(4096, 50 , bias=True)
        # self.global_rel_logist.weight = torch.nn.init.xavier_normal(self.global_rel_logist.weight, gain=1.0)

        # self.global_logist = CosineLinear(4096,150)
        self.global_sub_additive = nn.Linear(4096, 1, bias=True)
        self.global_obj_additive = nn.Linear(4096, 1, bias=True)

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )

        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                                         obj_dim=self.obj_dim,
                                         nl_obj=nl_obj, nl_edge=nl_edge, dropout_rate=rec_dropout,
                                         order=order,
                                         pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
                                         pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge)

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096,
                         pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        ###################################
        self.post_lstm = nn.Linear(self.hidden_dim, self.pooling_dim * 2)

        self.edge_coordinate_embedding = nn.Sequential(*[
            nn.BatchNorm1d(5, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])
        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.

        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
        self.post_lstm.bias.data.zero_()

        if nl_edge == 0:
            self.post_emb = nn.Embedding(self.num_classes, self.pooling_dim * 2)
            self.post_emb.weight.data.normal_(0, math.sqrt(1.0))

        self.rel_compress = nn.Linear(4096 + 256, 51, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)

        self.node_transform = nn.Linear(4096, 256, bias=True)
        self.edge_transform = nn.Linear(4096, 256, bias=True)
        # self.rel_compress = CosineLinear(self.pooling_dim+256, self.num_rels)
        # self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias()
        if self.gnn:
            self.graph_network_node = GraphNetwork(4096)
            self.graph_network_edge = GraphNetwork()
            if self.training:
                self.graph_network_node.train()
                self.graph_network_edge.train()
            else:
                self.graph_network_node.eval()
                self.graph_network_edge.eval()
        self.edge_sim_network = nn.Linear(4096, 1, bias=True)
        self.metric_net = MetricLearning()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def coordinate_feats(self, boxes, rel_inds):
        coordinate_rep = {}
        coordinate_rep['center'] = center_size(boxes)
        coordinate_rep['point'] = torch.cat((boxes, coordinate_rep['center'][:, 2:]), 1)
        sub_coordnate = {}
        sub_coordnate['center'] = coordinate_rep['center'][rel_inds[:, 1]]
        sub_coordnate['point'] = coordinate_rep['point'][rel_inds[:, 1]]

        obj_coordnate = {}
        obj_coordnate['center'] = coordinate_rep['center'][rel_inds[:, 2]]
        obj_coordnate['point'] = coordinate_rep['point'][rel_inds[:, 2]]
        edge_of_coordinate_rep = torch.zeros(sub_coordnate['center'].size(0), 5).cuda().float()
        edge_of_coordinate_rep[:, 0] = (sub_coordnate['point'][:, 0] - obj_coordnate['center'][:, 0]) * 1.0 / \
                                       obj_coordnate['center'][:, 2]
        edge_of_coordinate_rep[:, 1] = (sub_coordnate['point'][:, 1] - obj_coordnate['center'][:, 1]) * 1.0 / \
                                       obj_coordnate['center'][:, 3]
        edge_of_coordinate_rep[:, 2] = (sub_coordnate['point'][:, 2] - obj_coordnate['center'][:, 0]) * 1.0 / \
                                       obj_coordnate['center'][:, 2]
        edge_of_coordinate_rep[:, 3] = (sub_coordnate['point'][:, 3] - obj_coordnate['center'][:, 1]) * 1.0 / \
                                       obj_coordnate['center'][:, 3]
        edge_of_coordinate_rep[:, 4] = sub_coordnate['point'][:, 4] * sub_coordnate['point'][:, 5] * 1.0 / \
                                       obj_coordnate['center'][:, 2] \
                                       / obj_coordnate['center'][:, 3]
        return edge_of_coordinate_rep

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):

        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)

        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        global_feature = self.global_embedding(result.fmap.detach())
        result.global_dists = self.global_logist(global_feature)
        # print(result.global_dists)
        # result.global_rel_dists = F.sigmoid(self.global_rel_logist(global_feature))

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        # Prevent gradients from flowing back into score_fc from elsewhere
        result.rm_obj_dists, result.obj_preds, node_rep0 = self.context(
            result.obj_fmap,
            result.rm_obj_dists.detach(),
            im_inds, result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
            boxes.data, result.boxes_all)

        one_hot_multi = torch.zeros((result.global_dists.shape[0], self.num_classes))

        one_hot_multi[im_inds, result.rm_obj_labels] = 1.0
        result.multi_hot = one_hot_multi.float().cuda()
        edge_rep = node_rep0.repeat(1, 2)

        edge_rep = edge_rep.view(edge_rep.size(0), 2, -1)
        # global_feature_re = global_feature[im_inds]
        # subj_global_additive_attention = F.relu(self.global_sub_additive(edge_rep[:, 0] + global_feature_re))
        # obj_global_additive_attention = F.relu(
        #     torch.sigmoid(self.global_obj_additive(edge_rep[:, 1] + global_feature_re)))

        subj_rep = edge_rep[:, 0] #+ subj_global_additive_attention * global_feature_re
        obj_rep = edge_rep[:, 1] #+ obj_global_additive_attention * global_feature_re

        edge_of_coordinate_rep = self.coordinate_feats(boxes.data, rel_inds)

        e_ij_coordinate_rep = self.edge_coordinate_embedding(edge_of_coordinate_rep)

        union_rep = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
        edge_feat_init = union_rep

        prod_rep = subj_rep[rel_inds[:, 1]] * obj_rep[rel_inds[:, 2]] * edge_feat_init
        prod_rep = torch.cat((prod_rep, e_ij_coordinate_rep), 1)

        if self.use_tanh:
            prod_rep = F.tanh(prod_rep)
        if self.reachability:
            if self.training:
                result.center_loss = self.disc_center(prod_rep, result.rel_labels[:, -1]) * 0.05
                # self.centroids = self.disc_center.centroids.data
            result.rel_dists, features = self.meta_classify(prod_rep, self.centroids)
            result.rel_dists2 = features[-1]
        else:
            result.rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
                result.obj_preds[rel_inds[:, 1]],
                result.obj_preds[rel_inds[:, 2]],
            ), 1))

        if self.training:
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)

        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
