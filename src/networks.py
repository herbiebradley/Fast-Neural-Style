from torch import nn
from torchvision import models
from collections import namedtuple

def conv_block(in_depth, out_depth, padding_size, *args, inst_norm=True, activation=True, upsample=False, **kwargs):
    conv_block = []
    if upsample:
        conv_block += [Interpolate(scale_factor=2, mode='nearest')]

    conv_block += [nn.ReflectionPad2d(padding_size),
                   nn.Conv2d(in_depth, out_depth, *args, **kwargs)]
    if inst_norm:
        conv_block += [nn.InstanceNorm2d(out_depth, affine=True)]
    if activation:
        conv_block += [nn.ReLU()]

    return nn.Sequential(*conv_block)

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        base_filters = opt.base_filters
        kernel_size = opt.kernel_size
        paddings = [(kernel_size * 3) // 2, kernel_size // 2, kernel_size // 2]
        sampling_layers = 2

        # Encoder (initial layer)
        model = [conv_block(3, base_filters, paddings[0], kernel_size * 3, stride=1)]

        # Encoder (downsampling)
        for i in range(sampling_layers):
            mult = 2 ** i
            model += [conv_block(base_filters * mult, base_filters * mult * 2, paddings[i+1], kernel_size, stride=2)]

        # Add resnet blocks
        mult = 2 ** sampling_layers
        for i in range(opt.residual_blocks):
            model += [Residual_Block(base_filters * mult, kernel_size)]

        # Decoder (upsampling)
        for i in range(sampling_layers):
            mult = 2 ** (sampling_layers - i)
            model += [conv_block(base_filters * mult, int(base_filters * mult / 2), paddings[-(i+1)], kernel_size, upsample=True, stride=1)]

        # Decoder (final layer)
        model += [conv_block(base_filters, 3, paddings[0], kernel_size * 3, inst_norm=False, activation=False, stride=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Residual_Block(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2

        res_block = [conv_block(channels, channels, padding, kernel_size, stride=1)]
        res_block += [conv_block(channels, channels, padding, kernel_size, activation=False, stride=1)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, input):
        output = input + self.res_block(input)
        return output

# Workaround class to use nn.functional.interpolate as part of a Sequential model
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        input = self.interp(input, scale_factor=self.scale_factor, mode=self.mode)
        return input
########################################################
"---------------------VGG Network----------------------"
########################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        return (h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
