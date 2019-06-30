import torch

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        kernel_size = 3
        upsample = 2
        self.encoder = Encoder(kernel_size)
        self.res1 = Residual(128, kernel_size)
        self.res2 = Residual(128, kernel_size)
        self.res3 = Residual(128, kernel_size)
        self.res4 = Residual(128, kernel_size)
        self.res5 = Residual(128, kernel_size)
        self.decoder = Decoder(kernel_size, upsample)

    def forward(self, input):
        x = self.encoder(input)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.decoder(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, kernel_size):
        super(Encoder, self).__init__()

        self.ref_pad1 = torch.nn.ReflectionPad2D((kernel_size * 3) // 2)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size * 3, stride=1)
        self.inst1 = torch.nn.InstanceNorm2d(32, affine=True)

        self.ref_pad2 = torch.nn.ReflectionPad2D(kernel_size // 2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size, stride=2)
        self.inst2 = torch.nn.InstanceNorm2d(64, affine=True)

        self.ref_pad3 = torch.nn.ReflectionPad2D(kernel_size // 2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size, stride=2)
        self.inst3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = self.ref_pad1(input)
        x = self.conv1(x)
        x = self.inst1(x)
        x = self.relu(x)
        x = self.ref_pad2(x)
        x = self.conv2(x)
        x = self.inst2(x)
        x = self.relu(x)
        x = self.ref_pad3(x)
        x = self.conv3(x)
        x = self.inst3(x)
        x = self.relu(x)
        return x

class Residual(torch.nn.Module):
    def __init__(self, channels, kernel_size):
        super(Residual, self).__init__()

        self.ref_pad1 = torch.nn.ReflectionPad2D(kernel_size // 2)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size, stride=1)
        self.inst1 = torch.nn.InstanceNorm2d(channels, affine=True)

        self.ref_pad2 = torch.nn.ReflectionPad2D(kernel_size // 2)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size, stride=1)
        self.inst2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        residual = input
        x = self.ref_pad1(input)
        x = self.conv1(x)
        x = self.inst1(x)
        x = self.relu(x)
        x = self.ref_pad2(x)
        x = self.conv2(x)
        x = self.inst2(x)
        x = x + residual
        return x

class Decoder(torch.nn.Module):
    def __init__(self, kernel_size, upsample=None):
        super(Decoder, self).__init__()
        self.upsample = upsample
        self.ref_pad1 = torch.nn.ReflectionPad2D(kernel_size // 2)
        self.conv1 = torch.nn.Conv2d(128, 64, kernel_size, stride=1) # Fractional
        self.inst1 = torch.nn.InstanceNorm2d(64, affine=True)

        self.ref_pad2 = torch.nn.ReflectionPad2D(kernel_size // 2)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size, stride=1) # Fractional
        self.inst2 = torch.nn.InstanceNorm2d(32, affine=True)

        self.ref_pad3 = torch.nn.ReflectionPad2D((kernel_size * 3) // 2)
        self.conv3 = torch.nn.Conv2d(32, 3, kernel_size * 3, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = input
        x = torch.nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
        x = self.ref_pad1(x)
        x = self.conv1(x)
        x = self.inst1(x)
        x = self.relu(x)
        x = torch.nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
        x = self.ref_pad2(x)
        x = self.conv2(x)
        x = self.inst2(x)
        x = self.relu(x)
        x = self.ref_pad3(x)
        x = self.conv3(x)
