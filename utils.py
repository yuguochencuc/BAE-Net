import os
import json
import torch
import logging
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
EPSILON = np.finfo(np.float32).eps

def logger_print(log):
    logging.info(log)
    print(log)

def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num

class ToTensor(object):
    def __call__(self,
                 x,
                 type="float"):
        if type == "float":
            return torch.FloatTensor(x)
        elif type == "int":
            return torch.IntTensor(x)


def pad_to_longest(batch_data):
    """
    pad the waves with the longest length among one batch chunk
    :param batch_data:
    :return:
    """
    mix_wav_batch_list, target_wav_batch_list, wav_len_list = batch_data[0]
    mix_tensor, target_tensor = nn.utils.rnn.pad_sequence(mix_wav_batch_list, batch_first=True), \
                                nn.utils.rnn.pad_sequence(target_wav_batch_list, batch_first=True)  # (B,L,M)
    return mix_tensor, target_tensor, wav_len_list


class BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list


def json_extraction(file_path, json_path, data_type):
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    file_list = os.listdir(file_path)
    file_num = len(file_list)
    json_list = []

    for i in range(file_num):
        file_name = file_list[i]
        file_name = os.path.splitext(file_name)[0]
        json_list.append(file_name)

    with open(os.path.join(json_path, "{}_files.json".format(data_type)), "w") as f:
        json.dump(json_list, f, indent=4)
    return os.path.join(json_path, "{}_files.json".format(data_type))


def complex_mul(inpt1, inpt2):
    """
    inpt1: (B,2,...) or (...,2)
    inpt2: (B,2,...) or (...,2)
    """
    if inpt1.shape[1] == 2:
        out_r = inpt1[:,0,...]*inpt2[:,0,...] - inpt1[:,-1,...]*inpt2[:,-1,...]
        out_i = inpt1[:,0,...]*inpt2[:,-1,...] + inpt1[:,-1,...]*inpt2[:,0,...]
        return torch.stack((out_r, out_i), dim=1)
    elif inpt1.shape[-1] == 2:
        out_r = inpt1[...,0]*inpt2[...,0] - inpt1[...,-1]*inpt2[...,-1]
        out_i = inpt1[...,0]*inpt2[...,-1] + inpt1[...,-1]*inpt2[...,0]
        return torch.stack((out_r, out_i), dim=-1)
    else:
        raise RuntimeError("Only supports two tensor formats")

def complex_conj(inpt):
    """
    inpt: (B,2,...) or (...,2)
    """
    if inpt.shape[1] == 2:
        inpt_r, inpt_i = inpt[:,0,...], inpt[:,-1,...]
        return torch.stack((inpt_r, -inpt_i), dim=1)
    elif inpt.shape[-1] == 2:
        inpt_r, inpt_i = inpt[...,0], inpt[...,-1]
        return torch.stack((inpt_r, -inpt_i), dim=-1)

def complex_div(inpt1, inpt2):
    """
    inpt1: (B,2,...) or (...,2)
    inpt2: (B,2,...) or (...,2)
    """
    if inpt1.shape[1] == 2:
        inpt1_r, inpt1_i = inpt1[:,0,...], inpt1[:,-1,...]
        inpt2_r, inpt2_i = inpt2[:,0,...], inpt2[:,-1,...]
        denom = torch.norm(inpt2, dim=1)**2.0 + EPSILON
        out_r = inpt1_r * inpt2_r + inpt1_i * inpt2_i
        out_i = inpt1_i * inpt2_r - inpt1_r * inpt2_i
        return torch.stack((out_r/denom, out_i/denom), dim=1)
    elif inpt1.shape[-1] == 2:
        inpt1_r, inpt1_i = inpt1[...,0], inpt1[...,-1]
        inpt2_r, inpt2_i = inpt2[...,0], inpt2[...,-1]
        denom = torch.norm(inpt2, dim=-1)**2.0 + EPSILON
        out_r = inpt1_r * inpt2_r + inpt1_i * inpt2_i
        out_i = inpt1_i * inpt2_r - inpt1_r * inpt2_i
        return torch.stack((out_r/denom, out_i/denom), dim=-1)


class NormSwitch(nn.Module):
    def __init__(self,
                 norm_type: str,
                 format: str,
                 num_features: int,
                 affine: bool = True,
                 ):
        super(NormSwitch, self).__init__()
        self.norm_type = norm_type
        self.format = format
        self.num_features = num_features
        self.affine = affine

        if norm_type == "BN":
            if format == "1D":
                self.norm = nn.BatchNorm1d(num_features, affine=True)
            else:
                self.norm = nn.BatchNorm2d(num_features, affine=True)
        elif norm_type == "cLN":
            if format == "1D":
                self.norm = CumulativeLayerNorm1d(num_features, affine)
            else:
                self.norm = CumulativeLayerNorm2d(num_features, affine)
        elif norm_type == "cIN":
            if format == "2D":
                self.norm = CumulativeLayerNorm2d(num_features, affine)
        elif norm_type == "iLN":
            if format == "1D":
                self.norm = InstantLayerNorm1d(num_features, affine)
            else:
                self.norm = InstantLayerNorm2d(num_features, affine)

    def forward(self, inpt):
        return self.norm(inpt)


class CumulativeLayerNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1,num_features,1,1))
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        else:
            self.gain = Variable(torch.ones(1,num_features,1,1), requires_grad=False)
            self.bias = Variable(torch.zeros(1,num_features,1,1), requires_grad=False)

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1,3], keepdim=True)  # (B,1,T,1)
        step_pow_sum = inpt.pow(2).sum([1,3], keepdim=True)  # (B,1,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,1,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,1,T,1)

        entry_cnt = np.arange(channel*freq_num, channel*freq_num*(seq_len+1), channel*freq_num)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1,1,seq_len,1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

class CumulativeLayerNorm1d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1,num_features,1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1,num_features,1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_gra=False)

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)  # (B,T)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)  # (B,T)

        entry_cnt = np.arange(channel, channel*(seq_len+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)  # (B,T)

        cum_mean = cum_sum / entry_cnt  # (B,T)
        cum_var = (cum_power_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean.unsqueeze(dim=1).expand_as(inpt)) / cum_std.unsqueeze(dim=1).expand_as(inpt)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeInstanceNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1,num_features,1,1))
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        else:
            self.gain = Variable(torch.ones(1,num_features,1,1), requires_grad=False)
            self.bias = Variable(torch.zeros(1,num_features,1,1), requires_grad=False)


    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([3], keepdim=True)  # (B,C,T,1)
        step_pow_sum = inpt.pow(2).sum([3], keepdim=True)  # (B,C,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,C,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,C,T,1)

        entry_cnt = np.arange(freq_num, freq_num*(seq_len+1), freq_num)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1,1,seq_len,1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class InstantLayerNorm1d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(InstantLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1,num_features,1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1,num_features,1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_gra=False)

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        ins_mean = torch.mean(inpt, dim=1, keepdim=True)  # (B,1,T)
        ins_std = (torch.var(inpt, dim=1, keepdim=True) + self.eps).pow(0.5)  # (B,1,T)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class InstantLayerNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(InstantLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.gain = Variable(torch.ones(1, num_features, 1, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1, 1), requires_grad=False)

    def forward(self, inpt):
        # inpt: (B,C,T,F)
        ins_mean = torch.mean(inpt, dim=[1,3], keepdim=True)  # (B,C,T,1)
        ins_std = (torch.std(inpt, dim=[1,3], keepdim=True) + self.eps).pow(0.5)  # (B,C,T,1)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())



def sisnr(est, label):
    label_power = np.sum(label**2.0) + 1e-8
    scale = np.sum(est*label) / label_power
    est_true = scale * label
    est_res = est - est_true
    true_power = np.sum(est_true**2.0, axis=0) + 1e-8
    res_power = np.sum(est_res**2.0, axis=0) + 1e-8
    sdr = 10*np.log10(true_power) - 10*np.log10(res_power)
    return sdr

def cal_pesq(id, esti_utts, clean_utts, fs):
    clean_utt, esti_utt = clean_utts[id,:], esti_utts[id,:]
    from pypesq import pesq
    pesq_score = pesq(clean_utt, esti_utt, fs=fs)
    return pesq_score

def cal_stoi(id, esti_utts, clean_utts, fs):
    clean_utt, esti_utt = clean_utts[id,:], esti_utts[id,:]
    from pystoi import stoi
    stoi_score = stoi(clean_utt, esti_utt, fs, extended=True)
    return 100*stoi_score

def cal_sisnr(id, esti_utts, clean_utts, fs):
    clean_utt, esti_utt = clean_utts[id,:], esti_utts[id,:]
    sisnr_score = sisnr(esti_utt, clean_utt)
    return sisnr_score