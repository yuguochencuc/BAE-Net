import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils import weight_norm, spectral_norm, remove_weight_norm
from utils import NormSwitch
from ptflops import get_model_complexity_info
from thop import profile
import numpy as np
torch_eps = torch.finfo(torch.float).eps

import pdb
CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_layer_norm', 'layer_norm', 'time_group_norm'])

def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module

class EquivalentRectangularBandwidth():
    def __init__(self,
                 nfreqs: int = 768,
                 sample_rate: int = 48000,
                 total_erb_bands: int = 128,
                 low_freq: float = 20,
                 max_freq: float = 48000//2,
                 ):
        if not low_freq:
            low_freq = 20
        if not max_freq:
            max_freq = sample_rate // 2
        freqs = np.linspace(0, max_freq, nfreqs)  # 每个STFT频点对应多少Hz
        self.EarQ = 9.265  # _ERB_Q
        self.minBW = 24.7  # minBW
        # 在ERB刻度上建立均匀间隔
        erb_low = self.freq2erb(low_freq)  # 最低 截止频率
        erb_high = self.freq2erb(max_freq)  # 最高 截止频率
        # 在ERB频率上均分为(total_erb_bands +2)个 频带
        erb_lims = np.linspace(erb_low, erb_high, total_erb_bands + 2)
        cutoffs = self.erb2freq(erb_lims)  # 将 ERB频率再转到 hz频率, 在线性频率Hz上找到ERB截止频率对应的频率
        # self.nfreqs  F
        # self.freqs # 每个STFT频点对应多少Hz
        self.filters = self.get_bands(total_erb_bands, nfreqs, freqs, cutoffs)

    def freq2erb(self, frequency):
        """ [Hohmann2002] Equation 16"""
        return self.EarQ * np.log(1 + frequency / (self.minBW * self.EarQ))

    def erb2freq(self, erb):
        """ [Hohmann2002] Equation 17"""
        return (np.exp(erb / self.EarQ) - 1) * self.minBW * self.EarQ

    def get_bands(self,
                  total_erb_bands,
                  nfreqs,
                  freqs,
                  cutoffs):
        """
        获取erb bands、索引、带宽和滤波器形状
        :param erb_bands_num: ERB 频带数
        :param nfreqs: 频点数 F
        :param freqs: 每个STFT频点对应多少Hz
        :param cutoffs: 中心频率 Hz
        :param erb_points: ERB频带界限 列表
        :return:
        """
        cos_filts = np.zeros([nfreqs, total_erb_bands])  # (F, ERB)
        for i in range(total_erb_bands):
            lower_cutoff = cutoffs[i]  # 上限截止频率 Hz
            higher_cutoff = cutoffs[i + 2]  # 下限截止频率 Hz, 相邻filters重叠50%

            lower_index = np.min(np.where(freqs > lower_cutoff))  # 下限截止频率对应的Hz索引 Hz。np.where 返回满足条件的索引
            higher_index = np.max(np.where(freqs < higher_cutoff))  # 上限截止频率对应的Hz索引
            avg = (self.freq2erb(lower_cutoff) + self.freq2erb(higher_cutoff)) / 2
            rnge = self.freq2erb(higher_cutoff) - self.freq2erb(lower_cutoff)
            cos_filts[lower_index:higher_index + 1, i] = np.cos(
                (self.freq2erb(freqs[lower_index:higher_index + 1]) - avg) / rnge * np.pi)  # 减均值，除方差

        # 加入低通和高通，得到完美的重构
        filters = np.zeros([nfreqs, total_erb_bands + 2])  # (F, ERB)
        filters[:, 1:total_erb_bands + 1] = cos_filts
        # 低通滤波器上升到第一个余cos filter的峰值
        higher_index = np.max(np.where(freqs < cutoffs[1]))  # 上限截止频率对应的Hz索引
        filters[:higher_index + 1, 0] = np.sqrt(1 - np.power(filters[:higher_index + 1, 1], 2))
        # 高通滤波器下降到最后一个cos filter的峰值
        lower_index = np.min(np.where(freqs > cutoffs[total_erb_bands]))
        filters[lower_index:nfreqs, total_erb_bands + 1] = np.sqrt(
            1 - np.power(filters[lower_index:nfreqs, total_erb_bands], 2))
        return cos_filts

class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm_type = norm
    def forward(self, x):
        x = self.conv(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)

class Avg_downsampling(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.avg = nn.Sequential(
            nn.ConstantPad1d((kernel_size - 1, 0), 0),
            nn.AvgPool1d(kernel_size, stride=stride))

    def forward(self, x):
        x = self.avg(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm_type = norm
    def forward(self, x):
        x = self.conv(x)
        return x
    def remove_weight_norm(self):
        remove_weight_norm(self.conv)

class NormConv2d(nn.Module):
    def __init__(self, *args, norm, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return x

class Gate_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, de_flag):
        super(Gate_Conv, self).__init__()
        if de_flag == 0:
            self.conv = NormConv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride)
            self.gate_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())
        else:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride)
            self.gate_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())

    def forward(self, x):
        return self.conv(x) * self.gate_conv(x)

class Encoder(nn.Module):
    def __init__(self, input_size, kernel_size, norm: str = 'weight_norm'):
        super().__init__()
        #encode
        nfreqs = 769
        sampling_rate = 48000
        erb_dim = 128
        erb = EquivalentRectangularBandwidth(
            nfreqs=nfreqs,
            sample_rate=sampling_rate,
            total_erb_bands=erb_dim,
            low_freq=20,
            max_freq=sampling_rate//2
        )
        filter_bank_matrix = torch.from_numpy(erb.filters).float()
        self.register_buffer("linear2erb", filter_bank_matrix)

        kernel_size = kernel_size

        enc_c_in = [128, 128, 64, 64]

        enc_1 = nn.Sequential(
            nn.ConstantPad1d((kernel_size[0]-1, 0), 0),
            NormConv1d(erb_dim, enc_c_in[0], kernel_size=kernel_size[0], groups=1, stride=1, norm=norm),
            nn.LeakyReLU(0.2),
        )
        enc_2 = nn.Sequential(
            nn.ConstantPad1d((kernel_size[1]-1, 0), 0),
            NormConv1d(enc_c_in[0], enc_c_in[1], kernel_size=kernel_size[1], groups=1, stride=1, norm=norm),
            nn.LeakyReLU(0.2),
        )
        enc_3 = nn.Sequential(
            nn.ConstantPad1d((kernel_size[2]-1, 0), 0),
            NormConv1d(enc_c_in[1], enc_c_in[2], kernel_size=kernel_size[2], groups=1, stride=1, norm=norm),
            nn.LeakyReLU(0.2),
        )

        enc_4 = nn.Sequential(
            nn.ConstantPad1d((kernel_size[3]-1, 0), 0),
            NormConv1d(enc_c_in[2], enc_c_in[3], kernel_size=kernel_size[3], groups=1, stride=1, norm=norm),
            nn.LeakyReLU(0.2),
        )

        self.en = nn.ModuleList([enc_1, enc_2, enc_3, enc_4])

    def forward(self, x):
        x_list = []
        x_list.append(x)
        inputs_mag = x.permute(0, 2, 1).contiguous()
        erb_inputs = (torch.einsum("btf,fr->btr", [inputs_mag, self.linear2erb]) + torch_eps)
        x = erb_inputs.permute(0, 2, 1).contiguous()
        for i in range(len(self.en)):
            x = self.en[i](x)
            x_list.append(x)
        return x, x_list

    def remove_weight_norm(self):
        for i in range(len(self.en)):
            for idx, layer in enumerate(self.en[i]):
                if len(layer.state_dict()) != 0:
                    try:
                        nn.utils.remove_weight_norm(layer)
    #                    print("remove_layer_transpose==" + str(layer))
                    except:
                        layer.remove_weight_norm()
    #                    print("remove_layer_transpose==" + str(layer))


class Decoder(nn.Module):
    def __init__(self, input_size, kernel_size, norm: str = 'weight_norm'):
        super().__init__()
        # decode
        kernel_size = kernel_size
        de_c_in = [64, 128, 128, 769]

        dec_1 = nn.Sequential(
            nn.ConstantPad1d((kernel_size[0] - 1, 0), 0),
            NormConv1d(input_size, de_c_in[0], kernel_size=kernel_size[0], groups=1, stride=1, norm=norm),
            nn.LeakyReLU(0.2)
        )
        dec_2 = nn.Sequential(
            nn.ConstantPad1d((kernel_size[1] - 1, 0), 0),
            NormConv1d(de_c_in[0], de_c_in[1], kernel_size=kernel_size[1], groups=1, stride=1, norm=norm),
            nn.LeakyReLU(0.2)
        )
        dec_3 = nn.Sequential(
            nn.ConstantPad1d((kernel_size[2] - 1, 0), 0),
            NormConv1d(de_c_in[1], de_c_in[2], kernel_size=kernel_size[2], groups=1, stride=1, norm=norm),
            nn.LeakyReLU(0.2))
        # dec_4 = nn.Sequential(
        #     nn.ConstantPad1d((kernel_size[3] - 1, 0), 0),
        #     NormConv1d(de_c_in[2], de_c_in[3], kernel_size=kernel_size[3], groups=1, stride=1, norm=norm),
        #     nn.LeakyReLU(0.2)
        # )
        self.de = nn.ModuleList([dec_1, dec_2, dec_3])



        self.de_out = nn.Sequential(
            nn.ConstantPad1d((kernel_size[3] - 1, 0), 0),
            NormConv1d(de_c_in[2], de_c_in[3], kernel_size=kernel_size[3], groups=1, stride=1, norm=norm)
            )

        self.mask1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Tanh()
        )
        self.maskconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.mask_gate = nn.Sigmoid()

    def forward(self, x, x_list):
        for i in range(len(x_list)-2):
            x = torch.add(x, x_list[-(i + 1)])
            x = self.de[i](x)
        x = self.de_out(torch.add(x, x_list[1]))
        x_out = x.unsqueeze(1) # (B, 1, F, T)
        x_in = x_list[0].unsqueeze(1)  # (B, 1, F, T)
        x_mask_in = torch.add(x_out, x_in)
        x_mask_in = x_mask_in.squeeze(1)
        x_dual_mask = self.mask1(x_mask_in) * self.mask2(x_mask_in)
        out_mask = self.mask_gate(self.maskconv(x_dual_mask))  # mask
        out_full = x_out * out_mask
        out_full = out_full.squeeze(dim=1)
        out_full = torch.add(out_full, x_list[0])
        return out_full

    def remove_weight_norm(self):
        for i in range(len(self.de)):
            for idx, layer in enumerate(self.de[i]):
                if len(layer.state_dict()) != 0:
                    try:
                        nn.utils.remove_weight_norm(layer)
                    except:
                        layer.remove_weight_norm()
        for idx, layer in enumerate(self.de_out):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()


class Generator(nn.Module):
    def __init__(self, input_size, dec_dim, norm='weight_norm'):
        super().__init__()
        self.enc_kernel_size = [3, 3, 3, 3]
        self.dec_kernel_size = [3, 3, 3, 3]
        self.enc = Encoder(input_size, kernel_size=self.enc_kernel_size, norm=norm)
        self.dec = Decoder(dec_dim, kernel_size=self.dec_kernel_size, norm=norm)
        self.gru = GroupRNN(dec_dim, dec_dim, split_group=1, rnn_type="GRU", is_causal=True)

    def forward(self, x):
        enc_out, enc_list = self.enc(x)
        enc_out = enc_out.permute(0, 2, 1).contiguous()
        enc_out = self.gru(enc_out)
        enc_out = enc_out.permute(0, 2, 1).contiguous()
        dec_out = self.dec(enc_out, enc_list)
        return dec_out

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
        self.dec.remove_weight_norm()

class GroupRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 split_group: int,
                 rnn_type: str,
                 is_causal: bool,
                 ):
        super(GroupRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.split_group = split_group
        self.rnn_type = rnn_type
        self.is_causal = is_causal
        input_size_t = input_size // split_group
        hidden_size_t = hidden_size // split_group
        causal_flag = 1 if is_causal else 2
        self.rnn_list1 = nn.ModuleList(
            [getattr(nn, rnn_type)(input_size=input_size_t,
                                   hidden_size=hidden_size_t // causal_flag,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=not is_causal) for _ in range(split_group)])
        self.rnn_list2 = nn.ModuleList(
            [getattr(nn, rnn_type)(input_size=hidden_size_t,
                                   hidden_size=hidden_size_t // causal_flag,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=not is_causal) for _ in range(split_group)])
        self.norm1 = NormSwitch('BN', "1D", hidden_size)
        self.norm2 = NormSwitch('BN', "1D", hidden_size)

    def forward(self, inpt):
        """
            inpt: (B, T, F)
            return: (B, T, F)
        """
        x_list = torch.chunk(inpt, self.split_group, dim=-1)
        x = torch.stack([self.rnn_list1[i](x_list[i])[0] for i in range(self.split_group)], dim=-1)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm1(x)
        x = x.permute(0, 2, 1).contiguous()

        x = torch.chunk(x, self.split_group, dim=-1)
        x = torch.cat([self.rnn_list2[i](x[i])[0] for i in range(self.split_group)], dim=-1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm2(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


if __name__ == "__main__":
    # enc_kernel_size = [3, 3, 3, 3]
    # model = Encoder(input_size=768, kernel_size=enc_kernel_size, norm='weight_norm')
    # in1 = torch.randn(1, 768, 100)
    # flops, params = profile(model, inputs=[in1])
    # outputs = model(in1)
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    # macs, params = get_model_complexity_info(model, (768, 100), as_strings=True,
    #                                           print_per_layer_stat=True, verbose=True)
    # dec_kernel_size = [3, 3, 3, 3]
    # model_dec = Decoder(input_size=64, kernel_size=dec_kernel_size, norm='weight_norm')
    # out_full = model_dec(outputs[0], outputs[1])
    # print(out_full.shape)

    model = Generator(input_size=769, dec_dim=64, norm='weight_norm') #Generator(1, 16, 3, 4, 6, 161)
    model.remove_weight_norm()
#     # dec = Decoder(input_size=64, kernel_size=3, norm='weight_norm')
#     # dec.remove_weight_norm()

#     model_test= Generator_for_ptflops(input_size=257, dec_dim=32, norm='weight_norm') #Generator(1, 16, 3, 4, 6, 161)


    in1 = torch.randn(1, 769, 100)
    flops, params = profile(model, inputs=[in1])

    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    # outputs = model(in1)
    # print(outputs.shape)

    # en_test, en_list  = model_test(in1)
    # dec_out = dec(en_test, en_list)
    # print(dec_out.shape)
    # flops, params = profile(dec, inputs=[en_test, en_list])
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    macs, params = get_model_complexity_info(model, (769, 100), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
