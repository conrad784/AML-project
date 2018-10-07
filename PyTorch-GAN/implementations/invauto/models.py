import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#  CUSTOM ACTIVATION FUNCTION
##############################

class ModifiedLeakyReLUModule(nn.Module):
    def __init__(self, alpha = 0.5):
        super(ModifiedLeakyReLUModule, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return self.alpha * torch.max(torch.zeros(x.shape).cuda(), x.cuda()) + 1/self.alpha * torch.min(torch.zeros(x.shape).cuda(), x.cuda())

class ArcTanH(nn.Module):
    def __init__(self):
        super(ArcTanH, self).__init__()
    def forward(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = [nn.ConvTranspose2d(in_features, in_features, 3, bias = False).cuda(),
                        nn.Parameter(torch.ones((1, in_features, 1, 1)), requires_grad=True).cuda(),
                        nn.Conv2d(in_features, in_features, 3, bias = False).cuda(),
                        nn.Parameter(torch.ones((1, in_features, 1, 1)), requires_grad=True).cuda()]


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=9, alpha = 2):
        super(Generator, self).__init__()

        # Initial convolution block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 1, padding = 3, bias = False) # stride is 1
        self.bias1 = nn.Parameter(torch.ones((1, 64, 1, 1)), requires_grad = True)
        self.act = ModifiedLeakyReLUModule(alpha=0.5)
        self.invact = ModifiedLeakyReLUModule(alpha=2)
        # Downsampling
        in_features = 64
        out_features = 128

        self.conv2 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1, bias = False)
        self.bias2 = nn.Parameter(torch.ones((1, out_features, 1, 1)), requires_grad=True)
        in_features = 128
        out_features = 256
        self.conv3 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1, bias = False)
        self.bias3 = nn.Parameter(torch.ones((1, out_features, 1, 1)), requires_grad=True)
        in_features = 256
        out_features = 512

        # Residual blocks
        self.resblocks = []
        for _ in range(res_blocks):
            self.resblocks.append(ResidualBlock(in_features))

        # Upsampling
        out_features = 128
        self.deconv4 =  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1, bias = False)
        self.bias4 = nn.Parameter(torch.ones((1, out_features, 1, 1)), requires_grad=True)

        in_features = 128
        out_features = 64
        self.deconv5 =  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1, bias = False)
        self.bias5 = nn.Parameter(torch.ones((1, out_features, 1, 1)), requires_grad=True)

        in_features = 64
        out_features = 32
        # Output layer
        self.conv6 = nn.Conv2d(64, out_channels, 7, stride =1, padding = 3, bias = False)
        self.bias6 = nn.Parameter(torch.ones((1, out_channels, 1, 1)), requires_grad=True)

    def forward(self, x, type_ = 'encoder'):
        assert type_ in ['encoder', 'decoder']
        if type_ == 'encoder':
            # print('ENCODING')
            x = self.act(x)
            x = F.conv2d(x, self.conv1.weight, bias=None, padding=3, stride=1)
            x = x + self.bias1

            x = self.act(x)
            x = F.conv2d(x, self.conv2.weight, bias=None, padding=1, stride=2)
            x = x + self.bias2

            x = self.act(x)
            x = F.conv2d(x, self.conv3.weight, bias=None, padding=1, stride=2)
            x = x + self.bias3
            for i in range(len(self.resblocks)):
                x = self.act( x + self.resblocks[i].conv_block[2](self.resblocks[i].conv_block[0](x)+self.resblocks[i].conv_block[1])+self.resblocks[i].conv_block[3])

            x = self.act(x)
            x = F.conv_transpose2d(x, self.deconv4.weight, bias=None, padding=1, stride=2, output_padding=1)
            x = x + self.bias4

            x = self.act(x)
            x = F.conv_transpose2d(x, self.deconv5.weight, bias=None, padding=1, stride=2,  output_padding=1)
            x = x + self.bias5

            x = self.act(x)
            x = F.conv2d(x, self.conv6.weight, bias=None, padding=3, stride=1)
            x = self.bias6
            return x
        else:
            # print('DECODING')
            x = self.invact(x)
            x = x - self.bias6
            x = F.conv_transpose2d(x, self.conv6.weight, bias = None, padding = 3, stride = 1)

            x = self.invact(x)
            x = x - self.bias5
            x = F.conv2d(x, self.deconv5.weight, bias = None, padding = 1, stride = 2)

            x = self.invact(x)
            x = x - self.bias4
            x = F.conv2d(x, self.deconv4.weight, bias = None, padding = 1, stride = 2)

            for i in reversed(range(len(self.resblocks))):
                x = self.invact(x)
                x_ = x - self.resblocks[i].conv_block[3]
                x_ = F.conv_transpose2d(x_, self.resblocks[i].conv_block[2].weight, bias=None, padding=1, stride=1)
                x_ = x_ - self.resblocks[i].conv_block[1]
                x_ = F.conv2d(x_, self.resblocks[i].conv_block[0].weight, bias=None, padding=1, stride=1)
                x = x + x_

            x = self.invact(x)
            x = x - self.bias3
            x = F.conv_transpose2d(x, self.conv3.weight, bias=None, padding=1, stride=2, output_padding=1)

            x = self.invact(x)
            x = x - self.bias2
            x = F.conv_transpose2d(x, self.conv2.weight, bias=None, padding=1, stride=2,
                                   output_padding=1)

            x = self.invact(x)
            x = x - self.bias1
            x = F.conv_transpose2d(x, self.conv1.weight, bias=None, padding=3, stride=1)
            return x


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
