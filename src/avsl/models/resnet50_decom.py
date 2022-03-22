import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet50 as raw_resnet50
import torch.utils.model_zoo as model_zoo

class resnet50Decom(nn.Module):
    def __init__(self, pretrained=True, bn_freeze = True):
        super(resnet50Decom, self).__init__()

        self.model = raw_resnet50(pretrained)
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x): # FIXME: use resnet50 to do experiments
        x = self.model.conv1(x) # B x 64 x 112 x 112
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) # B x 64 x 56 x 56
        x = self.model.layer1(x) # B x 256 x 56 x 56
        x_layer2 = self.model.layer2(x) # B x 512 x 28 x 28
        x_layer3 = self.model.layer3(x_layer2) # B x 1024 x 14 x 14
        x_layer4 = self.model.layer4(x_layer3) # B x 2048 x 7 x 7

        # avg_x = self.model.gap(x)
        # max_x = self.model.gmp(x)

        # x = max_x + avg_x
        # x = x.view(x.size(0), -1)
        
        return [ # Mustn't use tuple type
            x_layer2,
            x_layer3,
            x_layer4
        ]

if __name__ == "__main__":
    model = resnet50Decom()
    data = torch.randn(3, 3, 224, 224)
    data.requires_grad = True
    out = model(data)
    loss = torch.sum(out)
    from torchviz import make_dot
    dot = make_dot(loss, params=dict(model.named_parameters()))
    dot.render("model")
    pass