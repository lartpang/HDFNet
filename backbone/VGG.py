import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

__all__ = ["Backbone_VGG_in1", "Backbone_VGG_in3", "Backbone_VGG_in4"]

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)


def Backbone_VGG_in4():
    net = vgg16_bn(pretrained=True, progress=True)
    div_1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=3, padding=1), *list(net.children())[0][1:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16


def Backbone_VGG_in1(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = vgg16_bn(pretrained=pretrained, progress=True)
    div_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), *list(net.children())[0][1:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16


def Backbone_VGG_in3(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = vgg16_bn(pretrained=pretrained, progress=True)
    div_1 = nn.Sequential(*list(net.children())[0][0:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16


def Backbone_VGG19_in1(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = vgg19_bn(pretrained=pretrained, progress=True)
    div_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), *list(net.children())[0][1:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:26])
    div_8 = nn.Sequential(*list(net.children())[0][26:39])
    div_16 = nn.Sequential(*list(net.children())[0][39:52])
    return div_1, div_2, div_4, div_8, div_16


def Backbone_VGG19_in3(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = vgg19_bn(pretrained=pretrained, progress=True)
    div_1 = nn.Sequential(*list(net.children())[0][0:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:26])
    div_8 = nn.Sequential(*list(net.children())[0][26:39])
    div_16 = nn.Sequential(*list(net.children())[0][39:52])
    return div_1, div_2, div_4, div_8, div_16


if __name__ == "__main__":
    div_1, div_2, div_4, div_8, div_16 = Backbone_VGG19_in3()
    in_data = torch.randn((1, 3, 320, 320))

    x1 = div_1(in_data)
    print(x1.size())
    x2 = div_2(x1)
    print(x2.size())
    x4 = div_4(x2)
    print(x4.size())
    x8 = div_8(x4)
    print(x8.size())
    x16 = div_16(x8)
    print(x16.size())
