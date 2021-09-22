import torch
from torch import nn
import torch.nn.functional as F
# from Res2Net.res2net import Bottle2neck, Res2Net
# Bottle2neck.expansion = 1
from torchvision.models.resnet import BasicBlock, Bottleneck


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def upsample(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)   # upsample
    return src


class main_stream(nn.Module):
    def __init__(self, inplanes=6, midplanes1=32, midplanes2=64, midplanes3=128, outplanes=256):
        super(main_stream, self).__init__()
        self.conv1_1 = nn.Conv2d(inplanes, midplanes1, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes1)
        self.conv1_2 = BasicBlock(midplanes1, midplanes1)
        print("load the basicBlock of ResNet model")
        self.downsample1 = nn.Sequential(
            nn.Conv2d(midplanes1, midplanes2, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(midplanes2),)
        self.conv2_1 = BasicBlock(midplanes1, midplanes2, stride=2, downsample=self.downsample1)   # w/2,h/2
        self.conv2_2 = BasicBlock(midplanes2, midplanes2)

        self.downsample2 = nn.Sequential(
            nn.Conv2d(midplanes2, midplanes3, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(midplanes3),)
        self.conv3_1 = BasicBlock(midplanes2, midplanes3, stride=2, downsample=self.downsample2)  # w/4,h/4
        self.conv3_2 = BasicBlock(midplanes3, midplanes3)

        self.downsample3 = nn.Sequential(
            nn.Conv2d(midplanes3, outplanes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(outplanes),)
        self.conv4_1 = BasicBlock(midplanes3, outplanes, stride=2, downsample=self.downsample3)      # w/8,h/8
        self.conv4_2 = BasicBlock(outplanes, outplanes)

        self.relu = nn.ReLU(True)

    def forward(self, input):
        # first layer  320*320
        conv1_1 = self.relu(self.bn1(self.conv1_1(input)))
        conv1_2 = self.conv1_2(conv1_1)
        # 2nd layer  160*160
        conv2_1 = self.conv2_1(conv1_2)
        conv2_2 = self.conv2_2(conv2_1)
        # 3th layer  80*80
        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        # 4th layer  40*40
        conv4_1 = self.conv4_1(conv3_2)
        conv4_2 = self.conv4_2(conv4_1)

        return conv1_2, conv2_2, conv3_2, conv4_2


class MSFF_left(nn.Module):
    def __init__(self, in_lc=32, in_mc=64, reduction=16, threshold=4, height=2, bias=False):
        super(MSFF_left, self).__init__()
        mid_c = in_lc      # mid_c = min(in_lc, in_mc)
        self.relu = nn.ReLU(True)
        self.height = height
        d = max(int(mid_c / reduction), threshold)
        # stage 0
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 1, bias=bias)
        self.bnl_0 = nn.BatchNorm2d(mid_c)
        self.m2m_0 = nn.Conv2d(in_mc, mid_c, 1, bias=bias)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        # stage 1
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=bias)
        # self.m2l_1 = nn.ConvTranspose2d(mid_c, mid_c, 3, 2, 1, 1)    # upsample
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        # stage 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(mid_c, d, 1, padding=0, bias=False), nn.PReLU())
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, mid_c, kernel_size=1, stride=1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_l, in_m):
        batch_size = in_l.shape[0]
        # stage 0
        l2l_0 = self.relu(self.bnl_0(self.l2l_0(in_l)))
        m2m_0 = self.relu(self.bnm_0(self.m2m_0(in_m)))
        # stage 1
        l2l_1 = self.relu(self.bnl_1(self.l2l_1(l2l_0)))
        m2l_1 = upsample(m2m_0, l2l_1)      # upsample
        l_sum_1 = l2l_1 + m2l_1
        n_features = l2l_1.shape[1]
        # stage 2
        feats_S = self.avg_pool(l_sum_1)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors_l = attention_vectors[:, 0, :, :]
        attention_vectors_m = attention_vectors[:, 1, :, :]
        l2l_2 = attention_vectors_l * l2l_1
        m2l_2 = attention_vectors_m * m2l_1
        l_sum_2 = l2l_2 + m2l_2

        return l_sum_2


class MSFF_right(nn.Module):
    def __init__(self, in_mc=128, in_hc=256, reduction=16, threshold=4, height=2, bias=False):
        super(MSFF_right, self).__init__()
        mid_c = in_hc         # mid_c = min(in_mc, in_hc)
        self.relu = nn.ReLU(True)
        self.height = height
        d = max(int(mid_c / reduction), threshold)
        # stage 0
        self.m2m_0 = nn.Conv2d(in_mc, mid_c, 1, bias=bias)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 1, bias=bias)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=bias)
        self.bnh_1 = nn.BatchNorm2d(mid_c)
        self.m2h_1 = nn.Conv2d(mid_c, mid_c, 3, 2, 1, bias=bias)       # downsample
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        # stage 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(mid_c, d, 1, padding=0, bias=False), nn.ReLU())
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, mid_c, kernel_size=1, stride=1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_m, in_h):
        batch_size = in_h.shape[0]
        # stage 0
        h2h_0 = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m2m_0 = self.relu(self.bnm_0(self.m2m_0(in_m)))
        # stage 1
        h2h_1 = self.relu(self.bnh_1(self.h2h_1(h2h_0)))
        m2h_1 = self.relu(self.bnm_1(self.m2h_1(m2m_0)))
        h_sum_1 = h2h_1 + m2h_1
        n_features = h2h_1.shape[1]
        # stage 2
        feats_S = self.avg_pool(h_sum_1)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors_h = attention_vectors[:, 0, :, :]
        attention_vectors_m = attention_vectors[:, 1, :, :]
        h2h_2 = attention_vectors_h * h2h_1
        m2h_2 = attention_vectors_m * m2h_1
        h_sum_2 = h2h_2 + m2h_2

        return h_sum_2


class MSFF_mid(nn.Module):
    def __init__(self, in_lc=32, in_mc=64, in_hc=128, reduction=16, threshold=4, height=3, bias=False):
        super(MSFF_mid, self).__init__()
        mid_c = in_mc       # mid_c = min(in_lc, in_mc, in_hc)
        self.relu = nn.ReLU(True)
        self.height = height
        d = max(int(mid_c / reduction), threshold)
        # stage 0
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 1, bias=bias)
        self.bnl_0 = nn.BatchNorm2d(mid_c)
        self.m2m_0 = nn.Conv2d(in_mc, mid_c, 1, bias=bias)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 1, bias=bias)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        # stage 1
        self.m2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=bias)
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        self.l2m_1 = nn.Conv2d(mid_c, mid_c, 3, 2, 1, bias=bias)              # downsample
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        # stage 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(mid_c, d, 1, padding=0, bias=False), nn.ReLU())
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, mid_c, kernel_size=1, stride=1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_l, in_m, in_h):
        batch_size = in_m.shape[0]
        # stage 0
        l2l_0 = self.relu(self.bnl_0(self.l2l_0(in_l)))
        m2m_0 = self.relu(self.bnm_0(self.m2m_0(in_m)))
        h2h_0 = self.relu(self.bnh_0(self.h2h_0(in_h)))
        # stage 1
        l2m_1 = self.relu(self.bnl_1(self.l2m_1(l2l_0)))
        m2m_1 = self.relu(self.bnm_1(self.m2m_1(m2m_0)))
        h2m_1 = upsample(h2h_0, m2m_1)       # upsample
        m_sum_1 = m2m_1 + l2m_1 + h2m_1
        n_features = m2m_1.shape[1]
        # stage 2
        feats_S = self.avg_pool(m_sum_1)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors_l = attention_vectors[:, 0, :, :]
        attention_vectors_m = attention_vectors[:, 1, :, :]
        attention_vectors_h = attention_vectors[:, 2, :, :]
        l2m_2 = attention_vectors_l * l2m_1
        m2m_2 = attention_vectors_m * m2m_1
        h2m_2 = attention_vectors_h * h2m_1
        m_sum_2 = l2m_2 + m2m_2 + h2m_2

        return m_sum_2


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32, threshold=8):     # dafault=32, dafault=8
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(threshold, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x, identity):
        # identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CA(nn.Module):    # coordinate Attention block
    def __init__(self, small_size=256, large_size=128, reduction=16, threshold=4):
        super(CA, self).__init__()
        self.down_dim = nn.Sequential(nn.Conv2d(small_size, large_size, 1, bias=False),
                                      nn.BatchNorm2d(large_size),
                                      nn.ReLU(inplace=True), )
        self.coordatt = CoordAtt(large_size*2, large_size, reduction=reduction, threshold=threshold)

    def forward(self, in_large, in_small):
        in_small_downdim = self.down_dim(in_small)
        in_small_up = upsample(in_small_downdim, in_large)
        input = torch.cat([in_large, in_small_up], dim=1)
        out_ca = self.coordatt(input, in_large)
        output = out_ca + in_small_up

        return output


class MODEL(nn.Module):
    def __init__(self, inplanes=6, planes1=32, planes2=64, planes3=128, planes4=256, outplanes=1):
        super(MODEL, self).__init__()
        self.main_stream = main_stream(inplanes, planes1, planes2, planes3, planes4)

        self.msim32 = MSFF_left(planes1, planes2)
        self.msim16 = MSFF_mid(planes1, planes2, planes3)
        self.msim8 = MSFF_mid(planes2, planes3, planes4)
        self.msim4 = MSFF_right(planes3, planes4)

        self.ca4t8 = CA(planes4, planes3)
        self.ca8t16 = CA(planes3, planes2)
        self.ca16t32 = CA(planes2, planes1)

        self.pred32 = nn.Conv2d(planes1, outplanes, 3, 1, 1)
        self.pred16 = nn.Conv2d(planes2, outplanes, 3, 1, 1)
        self.pred8 = nn.Conv2d(planes3, outplanes, 3, 1, 1)
        self.pred4 = nn.Conv2d(planes4, outplanes, 3, 1, 1)

    def forward(self, input):
        input32, input16, input8, input4 = self.main_stream(input)

        msim32_out = self.msim32(input32, input16)
        msim16_out = self.msim16(input32, input16, input8)
        msim8_out = self.msim8(input16, input8, input4)
        msim4_out = self.msim4(input8, input4)

        ca4t8_out = self.ca4t8(msim8_out, msim4_out)
        ca8t16_out = self.ca8t16(msim16_out, ca4t8_out)
        ca16t32_out = self.ca16t32(msim32_out, ca8t16_out)

        out32 = self.pred32(ca16t32_out)
        out16 = self.pred16(ca8t16_out)
        out8 = self.pred8(ca4t8_out)
        out4 = self.pred4(msim4_out)

        return torch.sigmoid(out32), torch.sigmoid(out16), torch.sigmoid(out8), torch.sigmoid(out4)


if __name__ == "__main__":
    # module = MODEL()
    # print(module)
    m = nn.AdaptiveAvgPool2d((None, 1))
    input = torch.randn(8, 64, 240, 240)
    output = m(input)
    # output = output.squeeze(3).unsqueeze(2)
    x_split, y_split = torch.split(output, [100, 140], dim=2)
    print('end')


