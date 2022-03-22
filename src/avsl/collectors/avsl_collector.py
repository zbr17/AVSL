import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init
import numpy as np
from torchdistlog.tqdm import tqdm
from torchdistlog import logging
from torch.cuda.amp import autocast

from gedml.core.collectors import BaseCollector, _DefaultGlobalCollector
from ..misc.utils import generate_slice, topk_mask

class AVSLCollector(_DefaultGlobalCollector, BaseCollector):
    """
    Replace Mahalanobis distance. L2 norm version: 11-7
    """
    def __init__(
        self,
        feature_dim_list=[512, 1024, 2048],
        embed_dim=512,
        num_classes=100,
        is_normalize=True,
        use_proxy=False,
        split_num=None,
        topk_corr=128,
        prob_gamma=10,
        m=0.5,
        index_p=2,
        loss0_weight=0.1,
        loss1_weight=0.2,
        loss2_weight=0.4,
        *args,
        **kwargs
    ):
        _DefaultGlobalCollector.__init__(self, *args, **kwargs)
        BaseCollector.__init__(self, metric=None, *args, **kwargs)
        self.feature_dim_list = feature_dim_list
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.is_normalize = is_normalize
        self.use_proxy = use_proxy
        self.topk_corr = topk_corr
        self.prob_gamma = prob_gamma
        self.m = m
        self.index_p = index_p
        self.loss0_weight = loss0_weight
        self.loss1_weight = loss1_weight
        self.loss2_weight = loss2_weight

        self.split_num = split_num
        self.feature_len = len(feature_dim_list)

        self.is_global_initiate = False
        self.is_link_initiate = False
        self.initiate_params()

        # NOTE: log variables
        self.to_record_list = []
        for i in range(self.feature_len - 1):
            self.to_record_list.extend([
                "delta_link_{}to{}_mean".format(i,i+1),
                "delta_link_{}to{}_max".format(i,i+1),
            ])
        for i in range(self.feature_len):
            self.to_record_list.extend([
                "prob_coef_{}_max".format(i),
                "prob_coef_{}_min".format(i),
                "prob_coef_{}_std".format(i),
                "prob_coef_{}_mean".format(i),
                "prob_bias_{}_max".format(i),
                "prob_bias_{}_min".format(i),
                "prob_bias_{}_std".format(i),
                "prob_bias_{}_mean".format(i)
            ])
    
    # NOTE: log variables
    def stats(self):
        with torch.no_grad():
            for i in range(self.feature_len):
                prob_coef = getattr(self, "prob_coef_{}".format(i))
                prob_bias = getattr(self, "prob_bias_{}".format(i))
                setattr(self, "prob_coef_{}_max".format(i), prob_coef.max())
                setattr(self, "prob_coef_{}_min".format(i), prob_coef.min())
                setattr(self, "prob_coef_{}_std".format(i), prob_coef.std())
                setattr(self, "prob_coef_{}_mean".format(i), prob_coef.mean())
                setattr(self, "prob_bias_{}_max".format(i), prob_bias.max())
                setattr(self, "prob_bias_{}_min".format(i), prob_bias.min())
                setattr(self, "prob_bias_{}_std".format(i), prob_bias.std())
                setattr(self, "prob_bias_{}_mean".format(i), prob_bias.mean())

    # NOTE: log_variables
    def stats_links(self,links):
        with torch.no_grad():
            for i in range(self.feature_len - 1):
                new_link = links[i]
                buffer_link = getattr(self, "link_{}to{}".format(i, i + 1))
                delta_link = torch.abs(new_link - buffer_link)
                setattr(self, "delta_link_{}to{}_mean".format(i,i+1), delta_link.mean())
                setattr(self, "delta_link_{}to{}_max".format(i,i+1), delta_link.max())

    def initiate_params(self):
        # proxy
        if self.use_proxy:
            proxy_labels = torch.arange(self.num_classes)
            self.register_buffer("proxy_labels", proxy_labels)
            for i in range(self.feature_len):
                proxy = torch.randn(self.num_classes, self.embed_dim)
                init.kaiming_normal_(proxy, a=math.sqrt(5))
                setattr(self, "proxy_{}".format(i), nn.Parameter(proxy))
        # prob
        for i in range(self.feature_len):
            prob_coef = torch.ones(self.embed_dim)
            prob_bias = torch.zeros(self.embed_dim)
            setattr(self, "prob_coef_{}".format(i), nn.Parameter(prob_coef))
            setattr(self, "prob_bias_{}".format(i), nn.Parameter(prob_bias))
        # link
        for i in range(self.feature_len - 1):
            link = torch.zeros(self.embed_dim, self.embed_dim)
            self.register_buffer("link_{}to{}".format(i, i + 1), link)
    
    def update_links(self, links):
        # NOTE: log variables
        # if self.is_link_initiate:
        #     self.stats_links(links)
        for i in range(self.feature_len - 1):
            new_link = links[i]
            buffer_link = getattr(self, "link_{}to{}".format(i, i + 1))
            if not self.is_link_initiate:
                buffer_link.data = new_link.data
            else:
                buffer_link.data = self.m * buffer_link.data + (1 - self.m) * new_link.data
        self.is_link_initiate = True
    
    def forward(
        self,
        embed_list,
        certainty_list,
        labels,
        embed_list2=None,
        certainty_list2=None,
        link_list=None,
    ) -> tuple:
        # NOTE: log variables
        # self.stats()
        if self.training:
            # train mode
            assert link_list is not None
            # update links
            self.update_links(link_list)
            # compute metric_mat
            if self.use_proxy:
                if self.num_classes <= 1000:
                    col_labels = self.proxy_labels
                else:
                    label_set = torch.unique(labels) # regard as indices and proxy-labels
                    col_labels = label_set
                embed_list2, certainty_list2 = [], []
                for i in range(self.feature_len):
                    if self.num_classes <= 1000:
                        proxy = getattr(self, "proxy_{}".format(i))
                    else:
                        proxy = getattr(self, "proxy_{}".format(i))[col_labels]
                    embed_list2.append(proxy)
                    certainty_list2.append(
                        torch.ones_like(proxy) * certainty_list[i].mean()
                    )
            else:
                col_labels = labels
                certainty_list2 = certainty_list
                embed_list2 = embed_list
            output_list = self.compute_all_mat(
                embed_list, 
                embed_list2,
                certainty_list,
                certainty_list2
            )
            output_list.extend([
                labels.unsqueeze(1),
                col_labels.unsqueeze(0),
                not self.use_proxy,
                self.loss0_weight,
                self.loss1_weight,
                self.loss2_weight
            ])
            return tuple(output_list)
        else:
            assert self.split_num is not None
            bs = embed_list[0].size(0)
            device = embed_list[0].device
            embed_list2 = embed_list if embed_list2 is None else embed_list2
            certainty_list2 = certainty_list if certainty_list2 is None else certainty_list2
            bs2 = embed_list2[0].size(0)
            metric_mat = torch.zeros((bs, bs2)).to(device)
            slice_dict = generate_slice(bs, self.split_num)
            for slice_index in slice_dict.values():
                metric_mat[slice_index, :] = self.compute_all_mat(
                    embed_list, 
                    embed_list2, 
                    certainty_list,
                    certainty_list2,
                    slice_index=slice_index
                )
            return metric_mat, None, None, None
    
    def compute_all_mat(
        self, 
        embed_list1, 
        embed_list2, 
        certainty_list1,
        certainty_list2,
        slice_index=None,
        slice_index2=None,
    ):
        diff_embed_mat_hat = None
        metric_mat_list = []

        for i in range(self.feature_len):
            embed1, embed2 = embed_list1[i], embed_list2[i]
            embed1 = embed1[slice_index] if slice_index is not None else embed1
            embed2 = embed2[slice_index2] if slice_index2 is not None else embed2
            cert1, cert2 = certainty_list1[i], certainty_list2[i]
            cert1 = cert1[slice_index] if slice_index is not None else cert1
            cert2 = cert2[slice_index2] if slice_index2 is not None else cert2

            # batch normalization
            embed1, embed2 = F.normalize(embed1, dim=-1, p=self.index_p), F.normalize(embed2, dim=-1, p=self.index_p)
            # compute diff_embed_mat
            embed1 = embed1.unsqueeze(1)
            embed2 = embed2.unsqueeze(0)
            diff_embed_mat = torch.abs(embed1 - embed2).pow(self.index_p)

            # compute metric_mat
            cur_metric_mat = torch.sum(diff_embed_mat, dim=-1)
            metric_mat_list.append(cur_metric_mat)

            if diff_embed_mat_hat is None:
                diff_embed_mat_hat = diff_embed_mat.detach()
            else:
                # conversion
                link = torch.relu(getattr(self, "link_{}to{}".format(i-1,i)))
                mask = topk_mask(link, k=self.topk_corr, dim=0, ismax=True).byte()
                link_star = link * mask
                link_star = link_star / (torch.sum(link_star, dim=0, keepdim=True) + 1e-8)
                diff_embed_mat_trans = torch.matmul(diff_embed_mat_hat, link_star)

                # certainty
                prob_coef = getattr(self, "prob_coef_{}".format(i))
                prob_bias = getattr(self, "prob_bias_{}".format(i))
                prob = cert1.unsqueeze(1) * cert2.unsqueeze(0) * prob_coef + prob_bias
                prob = torch.sigmoid(self.prob_gamma * prob) # probability for certainty

                diff_embed_mat_hat = (
                    diff_embed_mat.detach() * prob + 
                    diff_embed_mat_trans * (1 - prob)
                )
        
        # compute final output
        final_metric_mat = torch.sum(diff_embed_mat_hat, dim=-1)
        
        # output
        if self.training:
            return [final_metric_mat] + metric_mat_list
        else:
            return final_metric_mat


