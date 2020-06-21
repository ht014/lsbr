import torch
import torch.nn as nn
from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# encoder for imagenet dataset
class EmbeddingImagenet(nn.Module):
    def __init__(self,
                 emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 512

        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=512,
                                              out_channels=512,
                                              kernel_size=3,
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=512),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=512,
                                              out_channels=int(512),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(512)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))


        # self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
        #                                       out_channels=self.hidden*2,
        #                                       kernel_size=3,
        #                                       padding=1,
        #                                       bias=False),
        #                             nn.BatchNorm2d(num_features=self.hidden * 2),
        #                             nn.MaxPool2d(kernel_size=2),
        #                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                             nn.Dropout2d(0.4))
        # self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
        #                                       out_channels=self.hidden*4,
        #                                       kernel_size=3,
        #                                       padding=1,
        #                                       bias=False),
        #                             nn.BatchNorm2d(num_features=self.hidden * 4),
        #                             nn.MaxPool2d(kernel_size=2),
        #                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                             nn.Dropout2d(0.5))
        self.fc6 = nn.Sequential(nn.Linear(in_features=512*7*7,
                                              out_features=self.emb_size, bias=True),

                                nn.Dropout2d(0.5),
                                 nn.ReLU(),

                                 )

        self.fc7 = nn.Sequential(nn.Linear(in_features=self.emb_size,
                                                  out_features=self.emb_size, bias=True),
       nn.Dropout2d(0.5),nn.ReLU())

        self.classify = nn.Sequential(nn.Linear(in_features=self.emb_size,
                                                  out_features=100, bias=True))

    def forward(self, input_data):
        # print(self.conv_1(input_data).size())
        # print(self.conv_2(self.conv_1(input_data)).size())
        output_data = self.fc7(self.fc6(self.conv_2(self.conv_1(input_data)).contiguous().view(input_data.size(0),-1)))
        # score = F.softmax(F.tanh((output_data)))
        return output_data



class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[1, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features ,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).cuda()

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)
        # print(aggr_feat.split(num_data, 1)[0].size())
        two =  aggr_feat.split(num_data, 1)
        node_feat = ( node_feat + two[0]+two[1]).transpose(1, 2)


        # node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        # print(node_feat.unsqueeze(-1).size(),"###")
        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class MetricLearning(nn.Module):
    def __init__(self,
                 in_features=4096,
                 num_features=1024,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(MetricLearning, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_feat):
        sim_val = F.sigmoid(self.sim_network(node_feat))
        return sim_val


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)



    def forward(self, node_feat, edge_feat,is_only_sim=False):
        # compute abs(x_i, x_j)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = F.sigmoid(self.sim_network(x_ij))
        if is_only_sim:
            return sim_val

        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val


        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).cuda()
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).cuda()
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features=256,
                 node_features=256,
                 edge_features=128,
                 num_layers=3,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.node2edge_first_net = EdgeUpdateNetwork(in_features=self.in_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout= 0.0)


        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat=None):
        # for each layer
        if edge_feat is None:

            edge_feat = self.node2edge_first_net(node_feat,edge_feat,is_only_sim=True)

            return edge_feat

        edge_feat_list = []
        node_feat_list =[]
        for l in range(self.num_layers):
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)
            # save edge feature
            edge_feat_list.append(edge_feat)
            node_feat_list.append(node_feat)

        return edge_feat_list,node_feat_list

if __name__ =='__main__':

        eun = EdgeUpdateNetwork(256, 128).cuda()
        f = torch.randn((1,8, 256)).float().cuda()
        e = torch.randn(( 2, 8, 8)).float().cuda()
        eun(f, e)
        print('edge update network is ok')

        nun = NodeUpdateNetwork(256,256).cuda()
        f = torch.randn((1, 8, 256)).float().cuda()
        e = torch.randn((2, 8, 8)).float().cuda()
        ddd = nun(f, e)
        print("feature:", ddd.size())
        print('node update network is ok')

        gun = GraphNetwork(256,128,128).cuda()
        edge_feat_list, node_feats= gun(f,e)
        print(node_feats[-1].size()) #1 x 8 x 4096
        print('graph network is ok')