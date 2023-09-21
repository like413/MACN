# -*- codeing =utf-8 -*-
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
import torch.nn.init as init


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class MACT(nn.Module):
    def __init__(self, in_planes=64, out_planes=64, kernel_att=7, head=8, kernel_conv=3, stride=1, dilation=1):
        super(MACT, self).__init__()
        self.in_planes = in_planes  # 输入通道
        self.out_planes = out_planes  # 输出通道
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation  # 扩张
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2

        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)

        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:  # 在 h，w方向按步长取值
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


class MCGF(nn.Module):
    def __init__(self, dim, heads=8, dim_head=8, dropout=0.1):
        super(MCGF, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2, kv_include_self=False):
        b, n, _, h = *x1.shape, self.heads
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # print(x1.shape)
        # print(out.shape)
        out = x1 + out
        f_q = self.to_q(out)
        f_k = self.to_k(out)
        f_v = self.to_v(x1)

        f_q = rearrange(f_q, 'b n (h d) -> b h n d', h=h)
        f_k = rearrange(f_k, 'b n (h d) -> b h n d', h=h)
        f_v = rearrange(f_v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', f_q, f_k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, f_v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


BATCH_SIZE_TRAIN = 1
NUM_CLASS = 6


class MixConvNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            num_classes=NUM_CLASS,
            num_tokens=4,
            dim=64,
            emb_dropout=0.1,

    ):
        super(MixConvNet, self).__init__()
        self.L = num_tokens
        self.cT = dim

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        # 28 23 18 13 8 3
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * 28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2d_features2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.cross = MCGF(dim)
        self.mact = MACT()

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x1, x2, mask=None):
        x1 = self.conv3d_features(x1)
        x1 = rearrange(x1, 'b c h w y ->b (c h) w y')
        x1 = self.conv2d_features(x1)

        x1 = self.mact(x1)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')

        x2 = self.conv2d_features2(x2)
        x2 = self.mact(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')

        wa1 = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A1 = torch.einsum('bij,bjk->bik', x1, wa1)
        A1 = rearrange(A1, 'b h w -> b w h')  # Transpose
        A1 = A1.softmax(dim=-1)

        VV1 = torch.einsum('bij,bjk->bik', x1, self.token_wV)
        T1 = torch.einsum('bij,bjk->bik', A1, VV1)

        wa2 = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A2 = torch.einsum('bij,bjk->bik', x2, wa2)
        A2 = rearrange(A2, 'b h w -> b w h')  # Transpose
        A2 = A2.softmax(dim=-1)

        VV2 = torch.einsum('bij,bjk->bik', x2, self.token_wV)
        T2 = torch.einsum('bij,bjk->bik', A2, VV2)

        cls_tokens1 = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_tokens1, T1), dim=1)
        x1 += self.pos_embedding
        x1 = self.dropout(x1)

        cls_tokens2 = self.cls_token.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_tokens2, T2), dim=1)
        x2 += self.pos_embedding
        x2 = self.dropout(x2)

        x_1 = self.cross(x1, x2)
        x_2 = self.cross(x2, x1)

        x1, x2 = map(lambda t: t[:, 0], (x_1, x_2))
        x = self.mlp_head(x1) + self.mlp_head(x2)

        return x


"""
if __name__ == '__main__':
    model = MixConvNet()
    model.eval()
    # print(model)
    input1 = torch.randn(64, 1, 30, 11, 11)  # 30 25 20 15 10 5 1
    input2 = torch.randn(64, 1, 11, 11)
    x = model(input1, input2)
    print(x.size())  # torch.Size([64, 6])
    # summary(model, ((64, 1, 30, 11, 11), (64, 1, 11, 11)))
"""
