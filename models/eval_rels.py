
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg2 import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry

from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os

conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model3 import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
if conf.test:
    val = test
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)
ind_to_predicates = train.ind_to_predicates
detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision
                    )


detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])
detector.centroids = ckpt['state_dict']['disc_center.centroids'].data
# if conf.mode == 'sgdet':
#     det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
#     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

all_pred_entries = []


def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list,
              thrs=(20, 50, 100)):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
        # assert np.all(rels_i[:,2] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }
        all_pred_entries.append(pred_entry)

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)


evaluator = BasicSceneGraphEvaluator.all_modes()
evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
evaluator_list = []  # for calculating recall of each relationship except no relationship
evaluator_multiple_preds_list = []
for index, name in enumerate(ind_to_predicates):
    if index == 0:
        continue
    evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))

if conf.cache is not None and os.path.exists(conf.cache):
    print("Found {}! Loading from it".format(conf.cache))
    with open(conf.cache, 'rb') as f:
        all_pred_entries = pkl.load(f)
    for i, pred_entry in enumerate(tqdm(all_pred_entries)):
        gt_entry = {
            'gt_classes': val.gt_classes[i].copy(),
            'gt_relations': val.relationships[i].copy(),
            'gt_boxes': val.gt_boxes[i].copy(),
        }

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)

    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file='1')
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
                                                      save_file='2')

else:
    detector.eval()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
                  evaluator_multiple_preds_list)
        # if val_b > 5000: break

    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file='1')
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
                                                      save_file='2')

    if conf.cache is not None:
        with open(conf.cache, 'wb') as f:
            pkl.dump(all_pred_entries, f)
