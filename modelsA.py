import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
import math
from spectral import SpectralNorm

def start_grad(model):
    for param in model.parameters():
        if 'running_mean' in param.name() or 'running_var' in param.name(): continue
        param.start_grad()

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)


class Self_Attn(nn.Module):

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = jt.zeros(1)


    def execute(self, x):

        m_batchsize, C, width, height = x.shape
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = jt.bmm(proj_query, proj_key)  # transpose check
        attention =nn.softmax(energy, dim=-1)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = jt.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out


class UNetDown(nn.Module):

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=None))
        layers.append(nn.LeakyReLU(scale=0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        return self.model(x)


class Equalblock(nn.Module):

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(Equalblock, self).__init__()
        layers = [nn.Conv(in_size, out_size, 3, stride=1, padding=1, bias=False)]
        if normalize:
            layers.append(nn.GroupNorm(out_size // 32, out_size, affine=None))
        layers.append(nn.LeakyReLU(scale=0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self.up = nn.Upsample(scale_factor=2)

    def execute(self, x, skip):
        skip = self.up(skip)
        fuse = jt.contrib.concat((x, skip), dim=1)
        return self.model(fuse)


class UNetUp(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0, attention=False):
        super(UNetUp, self).__init__()

        layers = [nn.UpsamplingNearest2d(scale_factor=2),
                  nn.Conv2d(in_size, out_size, 3, padding=1, bias=False),
                  nn.GroupNorm(out_size // 8, out_size, affine=None), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if attention:
            layers.append(Self_Attn(out_size))
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        x = self.model(x)
        return x


class concat(nn.Module):
    def __init__(self):
        super(concat, self).__init__()

    def execute(self, x, skip_input):
        x = jt.contrib.concat((x, skip_input), dim=1)
        return x


class GeneratorUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.con = concat()
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.f1 = Equalblock(1024, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5, attention=True)
        self.f2 = Equalblock(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.f3 = Equalblock(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256, dropout=0.5, attention=True)
        self.f4 = Equalblock(768, 256, dropout=0.5)
        self.up5 = UNetUp(512, 128)
        self.f5 = Equalblock(384, 128, dropout=0.5)
        self.up6 = UNetUp(256, 64)
        self.f6 = Equalblock(192, 64, dropout=0.5)

        #self.final = nn.Sequential(nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv(128, out_channels, 4, padding=1), nn.Tanh())
        self.final = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv(128, out_channels, 3, padding=1), nn.Tanh())

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7)
        f1 = self.f1(u1, d7)
        u1 = self.con(f1, d6)
        u2 = self.up2(u1)
        f2 = self.f2(u2, d6)
        u2 = self.con(f2, d5)
        u3 = self.up3(u2)
        f3 = self.f3(u3, d5)
        u3 = self.con(f3, d4)
        u4 = self.up4(u3)
        f4 = self.f4(u4, d4)
        u4 = self.con(f4, d3)
        u5 = self.up5(u4)
        f5 = self.f5(u5, d3)
        u5 = self.con(f5, d2)
        u6 = self.up6(u5)
        f6 = self.f6(u6, d2)
        u6 = self.con(f6, d1)
        fi = self.final(u6)

        return fi


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_filters, out_filters, stride=2, normalization=True):
        super(DiscriminatorBlock, self).__init__()
        layers = [nn.Conv(in_filters, out_filters, 4, stride=stride, padding=1)]
        if normalization:
            layers.append(nn.BatchNorm2d(out_filters, eps=1e-05, momentum=0.1, affine=True))
            #layers.append(SpectralNorm(nn.BatchNorm2d(out_filters, eps=1e-05, momentum=0.8, affine=True)))
        layers.append(nn.LeakyReLU(scale=0.2))
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        x = self.model(x)
        return x



class DiscriminatorUnet(nn.Module):

    def __init__(self, in_channels=3):
        super(DiscriminatorUnet, self).__init__()
        self.d1 = DiscriminatorBlock(in_channels * 2, 64, normalization=False)
        self.d2 = DiscriminatorBlock(64, 128)
        self.d3 = DiscriminatorBlock(128, 256)
        self.d4 = DiscriminatorBlock(256, 512, stride=1)

        self.fe = nn.Conv(512, 1, 4, padding=1, bias=False)
        self.final = nn.Conv(2, 1, 4, padding=1, bias=False)


        for m in self.modules():
            weights_init_normal(m)

    def execute(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        fe = self.fe(d4)
        out_std = np.std(fe.data)
        mean_std = jt.array(out_std.mean())
        mean_std = mean_std.expand((fe.size(0), 1, 46, 62))
        out = jt.concat([fe, mean_std], 1)

        out = self.final(out)

        return out
