import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn import init
import math


class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


def adjust_learning_rate(optimizer, base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def scaled_l2(X, C, S):
    """
    scaled_l2 distance
    Args:
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
        S (k):      scale cofficient
    Return:
        D (b*n*k):  relative distance to each code
    Note:
        apparently the X^2 + C^2 - 2XC computation is 2x faster than
        elementwise sum, perhaps due to friendly cache in gpu
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert S.numel() == C.shape[0], "scale, codeword num mismatch"

    b, n, d = X.shape
    X = X.contiguous().view(-1, d)  # [bn, d]
    Ct = C.t()  # [d, k]
    X2 = X.pow(2.0).sum(-1, keepdim=True)  # [bn, 1]
    C2 = Ct.pow(2.0).sum(0, keepdim=True)  # [1, k]
    norm = X2 + C2 - 2.0 * X.mm(Ct)  # [bn, k]
    scaled_norm = S * norm
    D = scaled_norm.view(b, n, -1)  # [b, n, k]
    return D


def aggregate(A, X, C):
    """
    aggregate residuals from N samples
    Args:
        A (b*n*k):  weight of each feature contribute to code residual
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
    Return:
        E (b*k*d):  residuals to each code
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert A.shape[:2] == X.shape[:2], "weight, input dim mismatch"
    X = X.unsqueeze(2)  # [b, n, d] -> [b, n, 1, d]
    C = C[None, None, ...]  # [k, d] -> [1, 1, k, d]
    A = A.unsqueeze(-1)  # [b, n, k] -> [b, n, k, 1]
    R = (X - C) * A  # [b, n, k, d]
    E = R.sum(dim=1)  # [b, k, d]
    return E


class DilatedFCN(nn.Module):
    def __init__(self, num_features=103, num_classes=9, conv_features=64):
        super(DilatedFCN, self).__init__()
        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_cls = nn.Conv2d(conv_features, num_classes, kernel_size=1, stride=1, padding=0,
                                  bias=True)

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))
        x = self.conv_cls(x)
        x = interpolation(x)
        return x


class CAnet(nn.Module):
    def __init__(self, num_features=103, num_classes=9, conv_features=64, reduction=16, trans_features=32, K=48, D=32,
                 kernel_size=7):
        super(CAnet, self).__init__()
        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,
                               bias=True)

        self.convp = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

        # 添加channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(conv_features, conv_features // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(conv_features // 16, conv_features, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        ##

        self.encoding = nn.Conv2d(conv_features, D, kernel_size=1, stride=1, padding=0,
                                  bias=False)

        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.attention = nn.Linear(D, conv_features)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_cls = nn.Conv2d(conv_features * 3, num_classes, kernel_size=1, stride=1, padding=0,
                                  bias=True)

        self.drop = nn.Dropout(0.5)
        self.conv_features = conv_features
        self.trans_features = trans_features
        self.K = K
        self.D = D

        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        self.BN = nn.BatchNorm1d(K)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv0(x))  # C1
        conv1 = x

        x = self.relu(self.conv1(x))  # C2
        conv2 = x
        x = self.avgpool(x)  # P1

        x = self.relu(self.conv2(x))  # C3
        conv3 = x
        b, c, h, w = x.size()

        # channel attention
        avg_out_c = self.fc(self.avg_pool(x))
        max_out_c = self.fc(self.max_pool(x))
        out = avg_out_c + max_out_c

        x = self.sigmoid(out) * conv3
        ##

        ### position-attention ⬇
        avg_out_p = torch.mean(x, dim=1, keepdim=True)
        max_out_p, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out_p, max_out_p], dim=1)
        x = self.convp(x)
        x_p = self.sigmoid(x) * conv3
        ### ⬆

        x = conv3 + x_p

        Z = self.relu(self.encoding(x)).view(1, self.D, -1).permute(0, 2, 1)  # n,h*w,D

        A = F.softmax(scaled_l2(Z, self.codewords, self.scale), dim=2)  # b,n,k
        E = aggregate(A, Z, self.codewords)  # b,k,d
        E_sum = torch.sum(self.relu(self.BN(E)), 1)  # b,d
        gamma = self.sigmoid(self.attention(E_sum))  # b,num_conv
        gamma = gamma.view(-1, self.conv_features, 1, 1)
        x = x + x * gamma
        context3 = interpolation(x)
        conv2 = interpolation(conv2)
        conv1 = interpolation(conv1)

        x = torch.cat((conv1, conv2, context3), 1)
        x = self.conv_cls(x)

        return x


class SACNet(nn.Module):
    def __init__(self, in_chs, patch_size, class_nums, conv_features=64, trans_features=32, K=48, D=32):
        super(SACNet, self).__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size

        self.conv0 = nn.Conv2d(in_chs, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,  # 3
                               bias=True)

        self.alpha3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.beta3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.gamma3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.deta3 = nn.Conv2d(trans_features, conv_features, kernel_size=1, stride=1, padding=0,
                               bias=False)

        self.encoding = nn.Conv2d(conv_features, D, kernel_size=1, stride=1, padding=0,
                                  bias=False)

        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.attention = nn.Linear(D, conv_features)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_cls = nn.Conv2d(conv_features * 3, class_nums, kernel_size=1, stride=1, padding=0,
                                  bias=True)

        self.drop = nn.Dropout(0.5)
        self.conv_features = conv_features
        self.trans_features = trans_features
        self.K = K
        self.D = D

        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        self.BN = nn.BatchNorm1d(K)
        self.dense1 = nn.Sequential(
            nn.Linear(69312, 256), # 32448,43200,69312
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense3 = nn.Sequential(
            nn.Linear(128, class_nums)
        )

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv0(x))  # C1
        conv1 = x

        x = self.relu(self.conv1(x))  # C2
        conv2 = x
        # x = self.avgpool(x)  # P1

        x = self.relu(self.conv2(x))  # C3
        n, c, h, w = x.size()
        interpolation_context3 = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        # x_half = self.avgpool(x)  # P2
        x_half = x
        n, c, h, w = x_half.size()
        alpha_x = self.alpha3(x_half)
        beta_x = self.beta3(x_half)
        gamma_x = self.relu(self.gamma3(x_half))

        # alpha_x = alpha_x.squeeze().permute(1, 2, 0)  # 从n,c,h,w变成h,w,c
        # h*w x c
        alpha_x = alpha_x.view(n, -1, self.trans_features)  # 从h,w,c变成h*w,c    对应论文中(hw/16)*n
        # c x h*w
        beta_x = beta_x.view(n, self.trans_features, -1)  # 对应论文中n*(hw/16)
        gamma_x = gamma_x.view(n, self.trans_features, -1)

        context_x = torch.matmul(alpha_x, beta_x)
        context_x = F.softmax(context_x)

        context_x = torch.matmul(gamma_x, context_x)
        context_x = context_x.view(n, self.trans_features, h, w)
        context_x = interpolation_context3(context_x)

        deta_x = self.relu(self.deta3(context_x))  # F(U(B))
        x = deta_x + x

        Z = self.relu(self.encoding(x)).view(n, self.D, -1).permute(0, 2, 1)  # n,h*w,D

        A = F.softmax(scaled_l2(Z, self.codewords, self.scale), dim=2)  # b,n,k
        E = aggregate(A, Z, self.codewords)  # b,k,d
        E_sum = torch.sum(self.relu(self.BN(E)), 1)  # b,d
        gamma = self.sigmoid(self.attention(E_sum))  # b,num_conv
        gamma = gamma.view(-1, self.conv_features, 1, 1)
        x = x + x * gamma
        context3 = interpolation(x)
        conv2 = interpolation(conv2)
        conv1 = interpolation(conv1)

        x = torch.cat((conv1, conv2, context3), 1)

        # x = self.conv_cls(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out


class SpeFCN(nn.Module):
    def __init__(self, num_features=103, num_classes=9):
        super(SpeFCN, self).__init__()

        self.conv1 = nn.Conv2d(num_features, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0,
                                  bias=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        conv1 = x
        x = self.relu(self.conv2(x))
        conv2 = x
        x = self.relu(self.conv3(x))
        conv3 = x

        x = self.conv_cls(conv1 + conv2 + conv3)
        return x


class SpaFCN(nn.Module):
    def __init__(self, num_features=103, num_classes=9):
        super(SpaFCN, self).__init__()

        self.conv1 = nn.Conv2d(num_features, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0,
                                  bias=True)

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv1(x))
        conv1 = x
        x = self.avgpool(self.relu(self.conv2(x)))
        conv2 = x
        x = self.avgpool(self.relu(self.conv3(x)))
        conv3 = x

        x = self.conv_cls(conv1 + interpolation(conv2) + interpolation(conv3))

        return x


class SSFCN(nn.Module):
    def __init__(self, in_chs, patch_size, class_nums):
        super(SSFCN, self).__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size
        self.spe_conv1 = nn.Conv2d(in_chs, 64, kernel_size=1)
        self.spe_conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.spe_conv3 = nn.Conv2d(64, 64, kernel_size=1)

        self.spa_conv1 = nn.Conv2d(in_chs, 64, kernel_size=1)
        self.spa_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.spa_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.w_spe = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.w_spa = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.w_spe.data.uniform_(1, 2)
        self.w_spa.data.uniform_(1, 2)

        self.relu = nn.ReLU(inplace=True)

        self.x_shape = self.get_shape_after_2dconv()
        # self.conv_cls = nn.Conv2d(64, class_nums, kernel_size=1, stride=1, padding=0,
        #                           bias=True)
        self.conv_cls = nn.Linear(self.x_shape, class_nums)

    def get_shape_after_2dconv(self):
        x = torch.zeros((1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.spe_conv1(x)
            print
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        hsi = x

        x = self.relu(self.spe_conv1(hsi))
        spe_conv1 = x
        x = self.relu(self.spe_conv2(x))
        spe_conv2 = x
        x = self.relu(self.spe_conv3(x))
        spe_conv3 = x
        spe = spe_conv1 + spe_conv2 + spe_conv3

        x = self.relu(self.spa_conv1(hsi))
        spa_conv1 = x
        x = self.avgpool(self.relu(self.spa_conv2(x)))
        spa_conv2 = x
        x = self.avgpool(self.relu(self.spa_conv3(x)))
        spa_conv3 = x
        spa = spa_conv1 + interpolation(spa_conv2) + interpolation(spa_conv3)
        
        x = self.w_spe * spe + self.w_spa * spa
        x = x.contiguous().view(x.shape[0], -1)

        x = self.conv_cls(x)
        return x


# class HybridSN(nn.Module):
#     def __init__(self, num_features=103, num_classes=9):
#         super(HybridSN, self).__init__()
#         self.conv_layer1 = nn.Conv3d(num_features, 64, kernel_size=(3, 3, 7), stride=1, padding=1, dilation=1)
#         self.conv_layer2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 5), stride=1, padding=1, dilation=1)
#         self.conv_layer3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1, dilation=1)
#         self.conv_layer4 = nn.Conv2d(64, 64, kernel_size=(3, 3))
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv_cls = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0,
#                                   bias=True)
#         self.drop = nn.Dropout(0.4)
#
#     def forward(self, x):
#         interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
#         x = torch.unsqueeze(x, dim=2)
#         conv1 = self.relu(self.conv_layer1(x))
#         conv2 = self.relu(self.conv_layer2(conv1))
#         conv3 = self.relu(self.conv_layer3(conv2))
#         conv3 = torch.squeeze(conv3, dim=2)
#         p = self.avgpool(self.relu(self.conv_layer4(conv3)))
#         conv4 = interpolation(p)
#
#         x = self.conv_cls(conv4)
#         return x

class HybridSN(nn.Module):
    def __init__(self, in_chs, patch_size, class_nums):
        super().__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True))

        self.x1_shape = self.get_shape_after_3dconv()
        # print(self.x1_shape)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.x1_shape[1] * self.x1_shape[2], out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True))
        self.x2_shape = self.get_shape_after_2dconv()
        # print(self.x2_shape)
        self.dense1 = nn.Sequential(
            nn.Linear(self.x2_shape, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense3 = nn.Sequential(
            nn.Linear(128, class_nums)
        )

    def get_shape_after_2dconv(self):
        x = torch.zeros((1, self.x1_shape[1] * self.x1_shape[2], self.x1_shape[3], self.x1_shape[4]))
        with torch.no_grad():
            x = self.conv4(x)
            print
        return x.shape[1] * x.shape[2] * x.shape[3]

    def get_shape_after_3dconv(self):
        x = torch.zeros((1, 1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape

    def forward(self, X):
        X = X.unsqueeze(1)
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        # print(x.shape)
        x = self.conv4(x)
        x = x.contiguous().view(x.shape[0], -1)
        # print(x.shape)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out

class CNN_2D(nn.Module):
    def __init__(self, in_chs, patch_size, class_nums):
        super(CNN_2D, self).__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chs, 32, kernel_size=1,stride=1,padding=1),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1,stride=1,padding=1),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.x_shape = self.get_shape_after_2dconv()
        # self.conv_cls = nn.Conv2d(self.x_shape, class_nums, kernel_size=1, stride=1, padding=0,
        #                           bias=True)
        self.drop = nn.Dropout(0.4)
        self.conv_cls = nn.Linear(self.x_shape, class_nums)
        self.dense1 = nn.Sequential(
            nn.Linear(self.x_shape, 256),
            nn.ReLU(inplace=True))

        self.dense2 = nn.Sequential(
            nn.Linear(256, class_nums)
        )

    def get_shape_after_2dconv(self):
        x = torch.zeros((1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            conv1 = self.conv1(x)
            p1 = self.maxpool(conv1)
            conv2 = self.conv2(p1)
            x = self.maxpool(conv2)
        return x.shape[1] * x.shape[2] * x.shape[3]


    def forward(self, x):
        conv1 = self.conv1(x)
        p1 = self.maxpool(conv1)
        conv2 = self.conv2(p1)
        x = self.maxpool(conv2)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        out = self.dense2(x)
        # out = self.conv_cls(x)
        return out


class CNN_3D(nn.Module):
    def __init__(self, in_chs, patch_size, class_nums):
        super().__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True))

        self.x1_shape = self.get_shape_after_3dconv()
        # print(self.x1_shape)


        self.dense1 = nn.Sequential(
            nn.Linear(self.x1_shape, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense3 = nn.Sequential(
            nn.Linear(128, class_nums)
        )


    def get_shape_after_3dconv(self):
        x = torch.zeros((1, 1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, X):
        X = X.unsqueeze(1)
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.contiguous().view(x.shape[0], -1)
        # print(x.shape)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out


#  SSRN
class SPCModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SPCModuleIN, self).__init__()

        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=False)
        # self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        input = input.unsqueeze(1)

        out = self.s1(input)

        return out.squeeze(1)


class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(SPAModuleIN, self).__init__()

        # print('k=',k)
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k, 3, 3), bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        # print(input.size())
        out = self.s1(input)
        out = out.squeeze(2)
        # print(out.size)

        return out


class ResSPC(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPC, self).__init__()

        self.spc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(in_channels), )

        self.spc2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True), )

        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))

        return F.leaky_relu(out + input)


class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPA, self).__init__()

        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(in_channels), )

        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(inplace=True), )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))

        return F.leaky_relu(out + input)


class SSRN(nn.Module):
    def __init__(self, in_chs, patch_size, num_classes=16, k=97): # k=49
        super(SSRN, self).__init__()

        self.layer1 = SPCModuleIN(1, 28)
        # self.bn1 = nn.BatchNorm3d(28)

        self.layer2 = ResSPC(28, 28)

        self.layer3 = ResSPC(28, 28)

        # self.layer31 = AKM(28, 28, [97,1,1])
        self.layer4 = SPAModuleIN(28, 28, k=k)
        self.bn4 = nn.BatchNorm2d(28)

        self.layer5 = ResSPA(28, 28)
        self.layer6 = ResSPA(28, 28)

        self.fc = nn.Linear(28, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))  # self.bn1(F.leaky_relu(self.layer1(x)))
        # print(x.size())
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer31(x)

        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())

        return x


# SSTN
class SpatAttn(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # BxCxHxW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxHWxC
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW
        energy = torch.bmm(proj_query, proj_key)  # BxHWxHW, attention maps
        attention = self.softmax(energy)  # BxHWxHW, normalized attn maps
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxHW
        out = out.view(m_batchsize, C, height, width)  # BxCxHxW

        out = self.gamma * out + x
        return out


class SpatAttn_(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn_, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.Sequential(nn.ReLU(),
                                nn.BatchNorm2d(in_dim))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # BxCxHxW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxHWxC
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW
        energy = torch.bmm(proj_query, proj_key)  # BxHWxHW, attention maps
        attention = self.softmax(energy)  # BxHWxHW, normalized attn maps
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxHW
        out = out.view(m_batchsize, C, height, width)  # BxCxHxW

        out = self.gamma * out  # + x
        return self.bn(out)

class SARes(nn.Module):
    def __init__(self, in_dim, ratio=8, resin=False):
        super(SARes, self).__init__()

        if resin:
            self.sa1 = SpatAttn(in_dim, ratio)
            self.sa2 = SpatAttn(in_dim, ratio)
        else:
            self.sa1 = SpatAttn_(in_dim, ratio)
            self.sa2 = SpatAttn_(in_dim, ratio)

    def forward(self, x):
        identity = x
        x = self.sa1(x)
        x = self.sa2(x)

        return F.relu(x + identity)


class SPC32(nn.Module):
    def __init__(self, msize=24, outplane=49, kernel_size=[7, 1, 1], stride=[1, 1, 1], padding=[3, 0, 0], spa_size=9,
                 bias=True):
        super(SPC32, self).__init__()

        self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask0
        self.bn1 = nn.BatchNorm2d(outplane)

        self.convm2 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask2
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x  # NCHW
        n, c, h, w = identity.size()

        mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
        mask0 = torch.softmax(mask0.view(n, -1, h * w), -1)
        mask0 = mask0.view(n, -1, h, w)
        _, d, _, _ = mask0.size()

        fk = torch.einsum('ndhw,nchw->ncd', mask0, x)  # NCD

        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask0)  # NCDHW

        out = F.leaky_relu(out)
        out = out.sum(2)

        out = out  # + identity

        out0 = self.bn1(out.view(n, -1, h, w))

        mask2 = self.convm2(out0.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
        mask2 = torch.softmax(mask2.view(n, -1, h * w), -1)
        mask2 = mask2.view(n, -1, h, w)

        fk = torch.einsum('ndhw,nchw->ncd', mask2, x)  # NCD

        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask2)  # NCDHW

        out = F.leaky_relu(out)
        out = out.sum(2)

        out = out + identity

        out = self.bn2(out.view(n, -1, h, w))

        return out

class SSTN(nn.Module):
    def __init__(self, in_chs, patch_size, num_classes=9, msize=18, inter_size=49):
        super(SSTN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(103, inter_size, 1, bias=False),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(inter_size), )
        # nn.LeakyReLU())

        self.layer2 = SARes(inter_size, ratio=8)  # ResSPA(inter_size, inter_size)
        self.layer3 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])

        self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(msize)

        self.layer5 = SARes(msize, ratio=8)  # ResSPA(msize, msize)
        self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize, 1, 1], padding=[0, 0, 0])

        self.fc = nn.Linear(msize, num_classes)

    def forward(self, x):
        n, c, h, w = x.size()

        x = self.layer1(x)
        #         x = self.bn1(F.leaky_relu(self.layer1(x)))

        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer31(x)

        #         x = x.contiguous()
        #         x = x.reshape(n,-1,h,w)

        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)
        #         x = self.layer7(x)

        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())

        return x
