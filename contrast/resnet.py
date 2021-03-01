import math

import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet18_d', 'resnet34_d', 'resnet50_d', 'resnet101_d', 'resnet152_d',
           'resnet50_16s', 'resnet50_w2x', 'resnext101_32x8d', 'resnext152_32x8d']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=-1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=3, width=1,
                 groups=1, width_per_group=64,
                 mid_dim=1024, low_dim=128,
                 avg_down=False, deep_stem=False,
                 head_type='mlp_head', layer4_dilation=1):
        super(ResNet, self).__init__()
        self.avg_down = avg_down
        self.inplanes = 64 * width
        self.base = int(64 * width)
        self.groups = groups
        self.base_width = width_per_group

        mid_dim = self.base * 8 * block.expansion

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3_bn_relu(in_channel, 32, stride=2),
                conv3x3_bn_relu(32, 32, stride=1),
                conv3x3(32, 64, stride=1)
            )
        else:
            self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        if layer4_dilation == 1:
            self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        elif layer4_dilation == 2:
            self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=1, dilation=2)
        else:
            raise NotImplementedError
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.head_type = head_type
        if head_type == 'mlp_head':
            self.fc1 = nn.Linear(mid_dim, mid_dim)
            self.relu2 = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(mid_dim, low_dim)
        elif head_type == 'reduce':
            self.fc = nn.Linear(mid_dim, low_dim)
        elif head_type == 'conv_head':
            self.fc1 = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(2048)
            self.relu2 = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(mid_dim, low_dim)
        elif head_type in ['pass', 'early_return', 'multi_layer']:
            pass
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # zero gamma for batch norm: reference bag of tricks
        if block is Bottleneck:
            gamma_name = "bn3.weight"
        elif block is BasicBlock:
            gamma_name = "bn2.weight"
        else:
            raise RuntimeError(f"block {block} not supported")
        for name, value in self.named_parameters():
            if name.endswith(gamma_name):
                value.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        if self.head_type == 'multi_layer':
            return c2, c3, c4, c5

        if self.head_type == 'early_return':
            return c5

        if self.head_type != 'conv_head':
            c5 = self.avgpool(c5)
            c5 = c5.view(c5.size(0), -1)

        if self.head_type == 'mlp_head':
            out = self.fc1(c5)
            out = self.relu2(out)
            out = self.fc2(out)
        elif self.head_type == 'reduce':
            out = self.fc(c5)
        elif self.head_type == 'conv_head':
            out = self.fc1(c5)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc2(out)
        elif self.head_type == 'pass':
            return c5
        else:
            raise NotImplementedError

        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_d(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], deep_stem=True, avg_down=True, **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet34_d(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], deep_stem=True, avg_down=True, **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50_w2x(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], width=2, **kwargs)


def resnet50_16s(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], layer4_dilation=2, **kwargs)


def resnet50_d(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], deep_stem=True, avg_down=True, **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet101_d(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], deep_stem=True, avg_down=True, **kwargs)


def resnext101_32x8d(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet152_d(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], deep_stem=True, avg_down=True, **kwargs)


def resnext152_32x8d(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], groups=32, width_per_group=8, **kwargs)
