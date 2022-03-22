import torch
from torchdistlog import logging
from torchdistlog.tqdm import tqdm
import numpy as np 
from torch.utils.data import DataLoader

from gedml.launcher.testers import BaseTester
from ..misc.utils import generate_slice


class TesterFromMat(BaseTester):
    def __init__(self, split_num=None, *args, **kwargs):
        super(TesterFromMat, self).__init__(*args, **kwargs)
        self.split_num = split_num
    
    def prepare(
        self,
        models,
        collectors,
        datasets,
        evaluators,
        device,
        device_ids,
    ):
        # pass parameters
        self.models = models
        self.collectors = collectors
        self.datasets = datasets
        self.evaluators = evaluators
        self.device = device
        self.device_ids = device_ids
    
    def get_embeddings(self):
        logging.info(
            "Compute eval embeddings"
        )
        pbar = tqdm(self.dataloader_iter)
        embeddings_list, certainty_list, labels_list = [], [], []
        for info_dict in pbar:
            # get data
            data = info_dict["data"].to(self.device)
            label = info_dict["labels"].to(self.device)
            # forward
            feat_tuple = self.models["trunk"](data)
            embed_tuple, certainty_tuple, *_ = self.models["embedder"](feat_tuple)
            embeddings_list.append(embed_tuple)
            certainty_list.append(certainty_tuple)
            labels_list.append(label)
        self.embeddings = [
            torch.cat([
                item[i] for item in embeddings_list
            ], dim=0)
            for i in range(len(embeddings_list[0]))
        ]
        self.certainty = [
            torch.cat([
                item[i] for item in certainty_list
            ], dim=0)
            for i in range(len(certainty_list[0]))
        ]
        self.labels = torch.cat(labels_list)
        self.collectors["collector"].split_num = 100

    def compute_metrics(self):
        # for CUB200 and Cars196
        if self.split_num is None:
            self.metric_mat, *_ = self.collectors["collector"](
                embed_list=self.embeddings, 
                certainty_list=self.certainty,
                labels=None, 
            )
            self.metric_mat = self.metric_mat.cpu().numpy()
            self.labels = self.labels.cpu().numpy()
            metrics_dict = self.evaluators["default"].get_accuracy(
                self.metric_mat,
                self.labels,
                self.labels,
                True,
                device_ids=self.device_ids
            )
        # for Online-products
        else:
            col_metrics_dict = []
            is_list = isinstance(self.embeddings, list)
            if is_list:
                total_bs = self.embeddings[0].size(0)
            else:
                total_bs = self.embeddings.size(0)
            slice_dict = generate_slice(total_bs, self.split_num)
            for idx, slice_index in tqdm(slice_dict.items()):
                # split slice
                if is_list:
                    embed1 = [item[slice_index] for item in self.embeddings]
                    cert1 = [item[slice_index] for item in self.certainty]
                else:
                    embed1 = self.embeddings[slice_index]
                    cert1 = self.certainty[slice_index]
                sub_labels = self.labels[slice_index]
                embed2 = self.embeddings
                cert2 = self.certainty

                # forward collector
                metric_mat, *_ = self.collectors["collector"](
                    embed_list=embed1,
                    certainty_list=cert1,
                    embed_list2=embed2,
                    certainty_list2=cert2,
                    labels=None
                )
                # to numpy
                metric_mat = metric_mat.cpu().detach().numpy()
                sub_labels = sub_labels.cpu().numpy()

                col_metrics_dict.append(
                    self.evaluators["default"].get_accuracy(
                        metric_mat,
                        sub_labels,
                        self.labels.cpu().numpy(),
                        True,
                        device_ids=self.device_ids
                    )
                )
            # calculate mean
            keys_list = list(col_metrics_dict[0].keys())
            metrics_dict = {}
            for key in keys_list:
                result_list = [
                    item[key] for item in col_metrics_dict
                ]
                metrics_dict[key] = np.mean(result_list)

        return metrics_dict