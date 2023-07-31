import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.features = torch.nn.Sequential(*list(self.model.children())[:5])
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        x = self.features(x)
        return self.deconv(x)


class VGGNet(torch.nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.features = torch.nn.Sequential(*list(self.model.features.children())[:17])
        self.deconv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        x = self.features(x)
        return self.deconv(x)


class MobileNet(torch.nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V1')
        self.features = torch.nn.Sequential(*list(self.model.features.children())[:2])
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        x = self.features(x)
        return self.deconv(x)


class SwinTransformer(torch.nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.backbone = models.swin_t(weights='Swin_T_Weights.IMAGENET1K_V1')
        self.features = create_feature_extractor(self.backbone, return_nodes=['flatten'])

    def forward(self, x):
        for name, param in self.features.named_parameters():
            param.requires_grad = False
        return self.features(x)


class Head(torch.nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(768, 2),
        )

    def forward(self, x):
        return self.mlp_head(x)


class CECT(torch.nn.Module):
    def __init__(self):
        super(CECT, self).__init__()
        self.vggnet = VGGNet()
        self.resnet = ResNet()
        self.mobilenet = MobileNet()
        self.swin_transformer = SwinTransformer()
        self.head = Head()

    def forward(self, x, features=False):
        f_vgg = self.vggnet(x)
        f_res = self.resnet(x)
        f_mob = self.mobilenet(x)
        inte = (1/3) * f_vgg + (1/3) * f_res + (1/3) * f_mob
        tran_out = self.swin_transformer(inte)
        if features:
            return tran_out['flatten']
        else:
            return self.head(tran_out['flatten'])
