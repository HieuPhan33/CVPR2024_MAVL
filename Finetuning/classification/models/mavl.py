import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

class MAVL_ft(nn.Module):
    def __init__(self, base_model,out_size,linear_probe=False):
        super(MAVL_ft, self).__init__()
        self.model_name = base_model
        self.backbone, num_ftrs = self._get_basemodel(base_model, pretrained=True)
        # num_ftrs = int(resnet.fc.in_features)
        # self.res_features = nn.Sequential(*list(resnet.children())[:-1])
        # self.res_l1 = nn.Sequential(
        #     nn.Conv2d(num_ftrs, out_channels=num_ftrs, kernel_size=1),
        #     nn.PReLU()
        # )
        # d_model = 256
        # self.res_l2 = nn.Conv2d(num_ftrs, out_channels=d_model, kernel_size=1)
        self.out = nn.Linear(num_ftrs, out_size)


    def _get_basemodel(self, model_name, pretrained=False, layers=['blocks.9']):
        # try:
        ''' visual backbone'''
        net_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                        "resnet50": models.resnet50(pretrained=pretrained),
                        "ViT-B/16": models.vit_b_16(torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1),
                        "ViT-L/16": models.vit_l_16(torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)}
        if "resnet" in model_name:
            model = net_dict[model_name]
            num_ftrs = int(model.fc.in_features)
            backbone = nn.Sequential(*list(model.children())[:-1])
        elif 'timm-' in model_name:
            model_name = model_name.replace('timm-', '')
            model = timm.create_model(model_name, pretrained=True)
            backbone = create_feature_extractor(model, return_nodes={layers[0]: 'layer'})
            num_ftrs = backbone.patch_embed.proj.out_channels
        elif "ViT" in model_name:
            model = net_dict[model_name]
            backbone = create_feature_extractor(model, return_nodes={'encoder.ln': 'layer'}) 
            num_ftrs = model.hidden_dim
        return backbone, num_ftrs
    
    def forward(self, img,linear_probe=False):
        batch_size, _, H, W = img.shape
        x = self.backbone(img)
        if self.model_name in ['ViT-B/16', 'ViT-L/16']:
            x = x['encoder.ln'][:, 1:, :]
            x= x.permute(0, 2, 1).contiguous().view(batch_size, -1, H//16, W//16)
        x = x.squeeze()
        if linear_probe:
            return x
        else:
            x = self.out(x)
            return x