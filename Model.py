import torch.nn as nn
import math
import torch
import cbam
class Splitting(nn.Module):
    def __init__(self, horizontal):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        self.horizontal = horizontal
        if(horizontal):
            self.conv_even = lambda x: x[:, :, :, ::2]
            self.conv_odd = lambda x: x[:, :, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, :, ::2, :]
            self.conv_odd = lambda x: x[:, :, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))
class WaveletHaar(nn.Module):
    def __init__(self, horizontal):
        super(WaveletHaar, self).__init__()
        self.split = Splitting(horizontal)
        self.norm = math.sqrt(2.0)

    def forward(self, x):
        '''Returns the approximation and detail part'''
        (x_even, x_odd) = self.split(x)

        # Haar wavelet definition
        d = (x_odd - x_even) / self.norm
        c = (x_odd + x_even) / self.norm
        return (c, d)

class WaveletHaar2D(nn.Module):
    def __init__(self):
        super(WaveletHaar2D, self).__init__()
        self.horizontal_haar = WaveletHaar(horizontal=True)
        self.vertical_haar = WaveletHaar(horizontal=False)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.horizontal_haar(x)
        (LL, LH) = self.vertical_haar(c)
        (HL, HH) = self.vertical_haar(d)
        return (LL, LH, HL, HH)
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
from wavelet import DWT_Haar, IWT_Haar
class SKConv1(nn.Module):
    def __init__(self, features, M=3, G=32, r=16, stride=1, L=128):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv1, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        channel =features
        reduction =16
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

        kernel_size = 7
        self.compress = cbam.ChannelPool()
        self.spatial = cbam.BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
    def forward(self, x):

        batch_size = x.shape[0]

        feats = x
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        b, c, _, _ = feats_U.size()
        y = self.gap(feats_U).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #feats_S = self.gap(feats_U)
        #feats_Z = self.fc(feats_S)
        #feats_Z = y.expand_as(feats_U)
        x_compress = self.compress(feats_U)
        x_out = self.spatial(x_compress)
        #scale = self.softmax(x_out)  # broadcasting

        feats_Z = y * x_out
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, feats.shape[3], feats.shape[4])
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V
class SKConv(nn.Module):
    def __init__(self, features, M=3, G=32, r=16, stride=1, L=64):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        channel = features
        reduction = 16
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

        kernel_size = 7
        self.compress = cbam.ChannelPool()
        self.spatial = cbam.BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        batch_size = x.shape[0]

        feats = x
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        b, c, _, _ = feats_U.size()
        y = self.gap(feats_U).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # feats_S = self.gap(feats_U)
        # feats_Z = self.fc(feats_S)
        # feats_Z = y.expand_as(feats_U)
        x_compress = self.compress(feats_U)
        x_out = self.spatial(x_compress)
        # scale = self.softmax(x_out)  # broadcasting

        feats_Z = y * x_out
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, feats.shape[3], feats.shape[4])
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv3_31 = nn.Conv2d(in_channels=3, out_channels=31, kernel_size=3, padding=(1,1))
        self.conv32_32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=(1, 1))
        self.conv16_16 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=(1, 1))
        self.conv6_31 = nn.Conv2d(in_channels=6, out_channels=31, kernel_size=3, padding=(1, 1))
        self.conv9_9 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, padding=(1, 1))
        self.conv31_64 = nn.Conv2d(in_channels=31, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv80_31 = nn.Conv2d(in_channels=80, out_channels=31, kernel_size=3, padding=(1, 1))
        self.conv24_12 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=1, padding=0)
        self.conv64_64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv64_128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(1, 1))
        self.conv45_45 = nn.Conv2d(in_channels=45, out_channels=45, kernel_size=3, padding=(1, 1))
        self.conv34_64 = nn.Conv2d(in_channels=34, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv73_128 = nn.Conv2d(in_channels=73, out_channels=128, kernel_size=3, padding=(1, 1))
        self.conv128_64 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv173_256 = nn.Conv2d(in_channels=173, out_channels=256, kernel_size=3, padding=(1, 1))
        self.conv128_128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(1, 1))
        self.conv256_128 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=(1, 1))
        self.conv208_64 = nn.Conv2d(in_channels=208, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv576_64 = nn.Conv2d(in_channels=576, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv416_128 = nn.Conv2d(in_channels=416, out_channels=128, kernel_size=3, padding=(1, 1))
        self.conv448_64_3 = nn.Conv2d(in_channels=448, out_channels=448, kernel_size=3, padding=(1, 1))
        self.conv448_64_1 = nn.Conv2d(in_channels=448, out_channels=64, kernel_size=1, padding=0)
        self.conv448_128_1 = nn.Conv2d(in_channels=448, out_channels=128, kernel_size=1, padding=0)
        self.Upsa = nn.Upsample(scale_factor=8, mode='bilinear',align_corners=False)
        self.re = nn.ReLU(inplace=True).cuda()
        self.Maxpool = nn.MaxPool2d((2,2),stride=None)
        self.Transpose_64 = nn.ConvTranspose2d(64,64,4,stride=2,padding=1)
        self.Transpose_128 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.Transpose_128_64 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.Transpose_45_32 = nn.ConvTranspose2d(45, 32, 4, stride=2, padding=1)
        self.Transpose_32_16 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.Transpose_256 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.Transpose_256_64 = nn.ConvTranspose2d(256, 64, 6, stride=4, padding=1)
        self.Transpose_256_128 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.wavelet1 = WaveletHaar2D()
        self.RB_64 = DBlock(channel=64)
        self.RB_128 = DBlock(channel=128)
        self.RB_256 = DBlock(channel=256)
        self.DWT = DWT_Haar().cuda()
        self.IWT = IWT_Haar().cuda()
        self.Sk = SKConv(features=64)
        self.Sk1 = SKConv1(features=128)
    def forward(self, x, y):
        x1 = x
        Wavelet1_1 = self.DWT(x1)

        Wavelet1_2 = self.DWT(Wavelet1_1[:,0:3,:,:])
        Wavelet1_3 = Wavelet1_2[:,3:12,:,:]
        Wavelet1_4 = self.conv9_9(Wavelet1_3)

        Wavelet2_1 = Wavelet1_1[:,3:12,:,:]
        Wavelet2_1 = self.conv9_9(Wavelet2_1)
        Wavelet2_2 = self.DWT(Wavelet2_1)
        Wavelet2_2 = torch.cat([Wavelet1_4,Wavelet2_2], dim=1)

        Wavelet2_3 = self.conv45_45(Wavelet2_2)
        #Wavelet2_2 = self.re(Wavelet2_2).cuda()

        Wavelet2_3 = self.conv45_45(Wavelet2_3)
        Wavelet2_3 = self.Transpose_45_32(Wavelet2_3)

        Wavelet2_4 = self.conv32_32(Wavelet2_3)
        Wavelet2_4 = self.Transpose_32_16(Wavelet2_4)

        Wavelet3_1 = self.conv16_16(Wavelet2_4)
       # Wavelet1_3 = torch.cat([Wavelet3_1, Wavelet1_5], dim=1)
        Wavelet3_2 = Wavelet3_1

        Conv1_1 = self.Upsa(y)
        Conv1_2 = torch.cat([x,Conv1_1], dim=1)
        Conv1_3 = self.conv34_64(Conv1_2)
        Conv1_4 = self.RB_64(Conv1_3)

        Conv2_1 = self.Maxpool(Conv1_4)
        Conv2_1 = torch.cat([Wavelet2_1,Conv2_1], dim=1)
        Conv2_2 = self.conv73_128(Conv2_1)
        Conv2_3 = self.RB_128(Conv2_2)

        Conv3_1 = self.Maxpool(Conv2_3)
        Conv3_1 = torch.cat([Wavelet2_2, Conv3_1], dim=1)
        Conv3_2 = self.conv173_256(Conv3_1)

        Conv3_2 = self.RB_256(Conv3_2)

        AFF1_1 = self.Transpose_128_64(Conv2_3)
        AFF1_2 = self.Transpose_256_64(Conv3_2)
        AFF1_3 = torch.cat([AFF1_1, AFF1_2, Conv1_4], dim=1)
        AFF1_4 = self.Sk(AFF1_3)
        AFF1_5 = self.conv64_64(AFF1_4)

        AFF2_1 = self.Maxpool(Conv1_4)
        AFF2_1 = self.conv64_128(AFF2_1)
        AFF2_2 = self.Transpose_256_128(Conv3_2)
        AFF2_3 = torch.cat([AFF2_1, AFF2_2, Conv2_3], dim=1)
        AFF2_4 = self.Sk1(AFF2_3)
        AFF2_5 = self.conv128_128(AFF2_4)

        Conv3_3 = self.RB_256(Conv3_2)

        Conv4_1 = self.Transpose_256(Conv3_3)
        Conv4_1 = torch.cat([Conv4_1, AFF2_5, Wavelet2_3], dim=1)
        Conv4_2 = self.conv416_128(Conv4_1)
        Conv4_3 = self.RB_128(Conv4_2)

        Conv5_1 = self.Transpose_128(Conv4_3)
        Conv5_2 = torch.cat([Conv5_1, AFF1_5, Wavelet2_4], dim=1)
        Conv5_3 = self.conv208_64(Conv5_2)
        Conv5_4 = self.RB_64(Conv5_3)
        Conv5_4 = torch.cat([Conv5_4, Wavelet3_2], dim=1)
        conv5_5 = self.conv80_31(Conv5_4)
        conv5_6 = conv5_5 + Conv1_1

        HR = self.re(conv5_6)
        return HR