import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.nn.init as init

from gedml.core.modules import WithRecorder

class AVSLEmbedder(WithRecorder):
    """
    Since the 1x1 convolution operation in this module maps 1024-dim vector 
    into a 512-dim space (512 <= 1024 // 2), it can ensure that the mapping
    from R_+^(1024) space to R^(512) can be a surjection.
    """
    def __init__(
        self,
        feature_dim_list=[576, 1056, 1024],
        output_dim=512,
        *args,
        **kwargs
    ):
        super(AVSLEmbedder, self).__init__(*args, **kwargs)
        self.feature_dim_list = feature_dim_list
        self.output_dim = output_dim
        self.feature_len = len(feature_dim_list)
        self.initiate_params()

        # NOTE: log variables
        self.to_record_list = []
        for i in range(self.feature_len - 1):
            self.to_record_list.extend([
                "link_{}to{}_std".format(i,i+1), 
                "link_{}to{}_max".format(i,i+1), 
                "link_{}to{}_min".format(i,i+1), 
                "link_{}to{}_mean".format(i,i+1),
            ])
        for i in range(self.feature_len):
            self.to_record_list.extend([
                "certainty_{}_std".format(i),
                "certainty_{}_max".format(i),
                "certainty_{}_min".format(i),
                "certainty_{}_mean".format(i),
            ])
    
    def stats(self, links_list, certainty_list):
        with torch.no_grad():
            for i, link in enumerate(links_list):
                setattr(self, "link_{}to{}_std".format(i,i+1), link.std())
                setattr(self, "link_{}to{}_max".format(i,i+1), link.max())
                setattr(self, "link_{}to{}_min".format(i,i+1), link.min())
                setattr(self, "link_{}to{}_mean".format(i,i+1), link.mean())
            for i, certainty in enumerate(certainty_list):
                setattr(self, "certainty_{}_std".format(i), certainty.std())
                setattr(self, "certainty_{}_max".format(i), certainty.max())
                setattr(self, "certainty_{}_min".format(i), certainty.min())
                setattr(self, "certainty_{}_mean".format(i), certainty.mean())
    
    def initiate_params(self):
        """
        The maxpooling operation makes the embeddings tend towards a positive bias distribution.
        """
        # pooling
        self.adavgpool = nn.AdaptiveAvgPool2d(1)
        self.admaxpool = nn.AdaptiveMaxPool2d(1)
        
        for idx, dim in enumerate(self.feature_dim_list):
            # 1x1 conv
            conv = nn.Conv2d(dim, self.output_dim, kernel_size=(1, 1), stride=(1, 1))
            init.kaiming_normal_(conv.weight, mode="fan_out")
            init.constant_(conv.bias, 0)
            setattr(
                self,
                "conv1x1_{}".format(idx),
                conv
            )
    
    def compute_embedding_at_i(self, idx, input):
        # pooling
        ap_feat = self.adavgpool(input)
        mp_feat = self.admaxpool(input)
        output = ap_feat + mp_feat
        # get parameters
        conv = getattr(self, "conv1x1_{}".format(idx))
        # compute embeddigs
        output = conv(output)
        output = output.view(output.size(0), -1)
        return output
    
    def compute_embed(self, features):
        embed_list = [
            self.compute_embedding_at_i(idx, item)
            for idx, item in enumerate(features)
        ]
        return embed_list
    
    def _linearize(self, input):
        H, W = input.size(2), input.size(3)
        out = F.max_unpool2d(
            *F.adaptive_max_pool2d(
                input, output_size=1, return_indices=True
            ),
            kernel_size=(H, W)
        ) * H * W
        return out

    def compute_cam_at_i(self, idx, input):
        # linearize
        ap_output = input.detach()
        am_output = self._linearize(input.detach())
        output = ap_output + am_output
        # get parameters
        conv = getattr(self, "conv1x1_{}".format(idx))
        # compute cam
        output = conv(output)
        return output
    
    def compute_cam(self, features):
        cam_list = [
            self.compute_cam_at_i(idx, item)
            for idx, item in enumerate(features)
        ]
        return cam_list
    
    def compute_certainty(self, cams): # FIXME
        certainty_list = []
        for item in cams:
            # normalize
            item = item - item.min(dim=-1, keepdim=True)[0]
            item = item.flatten(2)
            item = F.normalize(item, dim=-1, p=1)
            # std
            certainty = item.std(dim=-1)
            certainty_list.append(certainty)
        return certainty_list
    
    def compute_link_at_i(self, low_input: torch.Tensor, high_input: torch.Tensor):
        low_input = low_input.detach()
        high_input = high_input.detach()
        # pooling if necessary
        if low_input.size()[2:] != high_input.size()[2:]:
            low_input = F.adaptive_avg_pool2d(
                low_input,
                output_size=high_input.size()[2:]
            )
        low_input = low_input.flatten(2)
        high_input = high_input.flatten(2)
        # normalize
        low_input = F.normalize(low_input, p=2, dim=-1)
        high_input = F.normalize(high_input, p=2, dim=-1)
        # compute link
        bs = low_input.size(0)
        link = torch.einsum("imj, inj -> mn", low_input, high_input) / bs
        return link
        
    def forward(self, features):
        """
        From low level to high level
        """
        # compute the reduced output
        embed_list = self.compute_embed(features)

        # for training 
        if self.training:
            with torch.no_grad():
                cam_list = self.compute_cam(features)
                certainty_list = self.compute_certainty(cam_list)
                # compute links
                links_list = []
                for idx in range(self.feature_len - 1):
                    cam_low = cam_list[idx].detach()
                    cam_high = cam_list[idx + 1].detach()
                    link = self.compute_link_at_i(cam_low, cam_high)
                    links_list.append(link)
                # NOTE: log variables
                # self.stats(links_list, certainty_list)
            return (
                embed_list,
                certainty_list,
                links_list
            )
        # for testing
        else:
            with torch.no_grad():
                cam_list = self.compute_cam(features)
                certainty_list = self.compute_certainty(cam_list)
            return (
                embed_list,
                certainty_list,
                None
            )
