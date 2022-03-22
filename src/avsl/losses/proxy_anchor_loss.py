import torch

from gedml.core.misc import loss_function as l_f
from gedml.core.losses.base_loss import BaseLoss

class ProxyAnchorLoss(BaseLoss):
    """
    paper: `Proxy Anchor Loss for Deep Metric Learning <http://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html>`_
    """
    def __init__(
        self,
        pos_margin=0.9,
        neg_margin=0.8,
        alpha=16,
        gamma=-0.1,
        **kwargs
    ):
        super(ProxyAnchorLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.gamma = gamma

        # NOTE: log variables
        self.to_record_list = [
            "pos_dist_max", "pos_dist_min", "pos_dist_mean",
            "neg_dist_max", "neg_dist_min", "neg_dist_mean",
            "dist_max", "dist_min", "dist_mean"
        ]
    
    # NOTE: log variables
    def stats(self, metric_mat, pos_mask, neg_mask):
        with torch.no_grad():
            self.dist_max = metric_mat.max()
            self.dist_min = metric_mat.min()
            self.dist_mean = metric_mat.mean()
            # pos pair
            pos_mat = metric_mat[pos_mask]
            self.pos_dist_max = pos_mat.max()
            self.pos_dist_min = pos_mat.min()
            self.pos_dist_mean = pos_mat.mean()
            # neg pair
            neg_mat = metric_mat[neg_mask]
            self.neg_dist_max = neg_mat.max()
            self.neg_dist_min = neg_mat.min()
            self.neg_dist_mean = neg_mat.mean()
    
    def required_metric(self):
        return ["cosine"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple=None,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:

        pos_mask = row_labels == col_labels
        neg_mask = ~ pos_mask
        if is_same_source:
            pos_mask.fill_diagonal_(False)

        with_pos_proxies = torch.where(torch.sum(pos_mask, dim=0) != 0)[0]

        pos_term = l_f.logsumexp(
            self.alpha * (metric_mat - self.pos_margin),
            keep_mask=pos_mask,
            add_one=True,
            dim=0
        ).squeeze()
        neg_term = l_f.logsumexp(
            - self.alpha * (metric_mat - self.neg_margin),
            keep_mask=neg_mask,
            add_one=True,
            dim=0
        ).squeeze()
        # NOTE: log variables
        # self.stats(metric_mat.detach(), pos_mask, neg_mask)
        
        if len(with_pos_proxies) == 0:
            pos_loss = torch.sum(metric_mat * 0)
        else:
            pos_loss = torch.mean(pos_term[with_pos_proxies])
        neg_loss = torch.mean(neg_term)

        return pos_loss + neg_loss