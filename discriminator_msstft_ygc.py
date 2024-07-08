import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

from .complexnn import ComplexConv2d, ConvSTFT
from ptflops import get_model_complexity_info
from thop import profile
from .norm_utils import NormConv1d, NormConv2d

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, N, m, st, sf, flag, norm='spectral_norm'):
        super(ResnetBlock, self).__init__()
        
        self.conv1 = NormConv2d(in_channels, N, kernel_size=3, padding=1, norm=norm)
        self.act1 = nn.LeakyReLU(0.2, True)
        self.shortcut = NormConv2d(in_channels, m*N, kernel_size =1, stride =(st, sf),
                                 padding=(0, 0), norm=norm)
        
#         if st == 1 and sf == 2 and flag != 4 and flag!=5:
#             self.conv2 = nn.Conv2d(N, m*N, kernel_size = ((st+2), (sf+2)),
#                                stride = (st, sf), padding = ((2, 2)))
                                  
#         if st == 2 and sf ==2 and flag != 4 and flag!=5:
#             self.conv2 = nn.Conv2d(N, m*N, kernel_size = ((st+2), (sf+2)),
#                                stride = (st, sf), padding = ((2, 2)))
                                  
#         if flag == 4:
#             self.conv2 = nn.Conv2d(N, m*N, kernel_size = ((st+2), (sf+2)),
#                                stride = (st, sf), padding = ((2, 2)))
#         if flag == 5:
        self.conv2 = NormConv2d(N, m*N, kernel_size = ((st+2), (sf+2)),
                               stride = (st, sf), padding = ((2, 2)), norm=norm)

    def forward(self, x):
             
        y = self.act1(self.conv1(x))  
        y = self.conv2(y)   
        shortcut = self.shortcut(x)
        result = shortcut+y[:,:,:shortcut.shape[2],:shortcut.shape[3]]
                               
        return result
    
class STFT_Discriminator(nn.Module):
    def __init__(self, C_out, C_in, norm):
        super().__init__()
        
        s = [1, 2]
        self.model = nn.ModuleDict()
        
        self.model["layer_0"] = nn.Sequential(
            NormConv2d(C_in, C_out, kernel_size=7, norm=norm),
            nn.LeakyReLU(0.2, True),
        )
        

        self.model["STFT_Discriminator_0"] = ResnetBlock(
            C_out, C_out, 2, 1, 2, 0, norm)
                               
        self.model["STFT_Discriminator_1"] = ResnetBlock(
            2*C_out, 2*C_out, 2, 2, 1, 1, norm)
                               
        self.model["STFT_Discriminator_2"] = ResnetBlock(
            4*C_out, 4*C_out, 1, 1, 2, 2, norm)
        
        self.model["STFT_Discriminator_3"] = ResnetBlock(
            4*C_out, 4*C_out, 2, 2, 1, 3, norm)
        
        self.model["STFT_Discriminator_4"] = ResnetBlock(
            8*C_out, 8*C_out, 1, 1, 2, 4, norm)
                               
        self.model["STFT_Discriminator_5"] = ResnetBlock(
            8*C_out, 8*C_out, 2, 2, 1, 5, norm)
        
        
        self.model["layer_1"] = NormConv2d(16*C_out, 1, kernel_size=(7,7), padding = ((3, 3)), norm=norm)
        
    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x.view(-1, 1))
    
        return results

class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.5)
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2),
                #nn.Dropout(0.5)
            )
            

        nf_prev = nf
        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.5)
        )

        model["layer_%d" % (n_layers + 2)] = nn.Conv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )
        
        #model["layer_%d" % (n_layers + 3)] = nn.Dropout(0.5)

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x.view(-1,1).detach())
       
        return results

class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                WNConv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=16, out_channels=64, kernel_size=41,
                         stride=4, padding=20, groups=4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=64, out_channels=256, kernel_size=41,
                         stride=4, padding=20, groups=16),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=256, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=64),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=256),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=5,
                         stride=1, padding=2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            WNConv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1,
                     padding=1)
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths,
            torch.div(lengths+3, 4, rounding_mode="floor"),
            torch.div(lengths+15, 16, rounding_mode="floor"),
            torch.div(lengths+63, 64, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


class WaveDiscriminator(nn.Module):
    def __init__(self, num_D, downsampling_factor):
        super().__init__()
        
        self.num_D = num_D
        self.downsampling_factor = downsampling_factor
        
        self.model = nn.ModuleDict({
            f"disc_{downsampling_factor**i}": WaveDiscriminatorBlock()
            for i in range(num_D)
        })
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)
    
    def features_lengths(self, lengths):
        return {
            f"disc_{self.downsampling_factor**i}": self.model[f"disc_{self.downsampling_factor**i}"].features_lengths(torch.div(lengths, 2**i, rounding_mode="floor")) for i in range(self.num_D)
        }
    
    def forward(self, x):
        results = {}
        for i in range(self.num_D):
            disc = self.model[f"disc_{self.downsampling_factor**i}"]
            results[f"disc_{self.downsampling_factor**i}"] = disc(x)
            x = self.downsampler(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.model = nn.ModuleDict()
        
#         self.model["stft_disc_0"] = STFT_Discriminator(8, 1, norm=norm)
#         self.model["stft_disc_1"] = STFT_Discriminator(8, 1, norm=norm)
        self.model["stft_disc_2"] = STFT_Discriminator(8, 1, norm=norm)
        
#         self.model["wav_disc_0"] = WaveDiscriminator(num_D, 2)
        
#         self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
#         self.apply(weights_init)

    def forward(self, x):
#         f = torch.stft(x.squeeze(1), n_fft=512, hop_length=160, win_length=512, window=torch.hann_window(window_length=512).cuda(), return_complex=True)
#         s_t = torch.abs(f).unsqueeze(1).permute(0,1,3,2)
#         f1 = torch.stft(x.squeeze(1), n_fft=1024, hop_length=320, win_length=1024, window=torch.hann_window(window_length=1024).cuda(), return_complex=True)
#         s_t1 = torch.abs(f1).unsqueeze(1).permute(0,1,3,2)
        f2 = torch.stft(x.squeeze(1), n_fft=1536, hop_length=480, win_length=1536, window=torch.hann_window(window_length=1536).cuda(), return_complex=True)
#        f2 = torch.stft(x.squeeze(1), n_fft=1024, hop_length=160, win_length=1024, window=torch.hann_window(window_length=1024), return_complex=True)
        s_t2 = torch.abs(f2).unsqueeze(1).permute(0,1,3,2)
        results = []
        for key, disc in self.model.items():
            if key == 'wav_disc_0':
                wave_disc = disc(x)
                for i_w, disc_w in wave_disc.items():
                    results.append(disc_w)
#             elif key == 'stft_disc_0':
#                 results.append(disc(s_t))
#             elif key == 'stft_disc_1':
#                 results.append(disc(s_t1))
            elif key == 'stft_disc_2':
                results.append(disc(s_t2))
                
        return results
    
# class Discriminator(nn.Module):
#     def __init__(self, num_D, ndf, n_layers, downsampling_factor):
#         super().__init__()
#         self.model = nn.ModuleDict()
        
#         self.model["stft_disc_0"] = STFT_Discriminator(32, 1)
        
#         for i in range(num_D):
#             self.model[f"wav_disc_{i}"] = NLayerDiscriminator(
#                 ndf, n_layers, downsampling_factor // (i + 1)
#             )
        
#         self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
# #         self.apply(weights_init)

#     def forward(self, x):
#         f = torch.stft(x.squeeze(1), n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(window_length=1024).cuda(), return_complex=True)
#         s_t = torch.abs(f).unsqueeze(1).permute(0,1,3,2)
#         results = []
#         for key, disc in self.model.items():
#             if key != 'stft_disc_0':
#                 results.append(disc(x))
#                 x = self.downsample(x)
#             else:
#                 results.append(disc(s_t))
                
#         return results


    
class Discriminator_stft(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.model = nn.ModuleDict()
        
        self.model["stft_disc_0"] = STFT_Discriminator(8, 1, norm=norm)
#         self.model["stft_disc_1"] = STFT_Discriminator(8, 1, norm=norm)
        self.model["stft_disc_2"] = STFT_Discriminator(8, 1, norm=norm)
        
#         self.model["wav_disc_0"] = WaveDiscriminator(num_D, 2)
        
#         self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
#         self.apply(weights_init)

    def forward(self, x):
        f = torch.stft(x.squeeze(1), n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(window_length=1024).cuda(), return_complex=True)
        s_t = torch.abs(f).unsqueeze(1).permute(0,1,3,2)
#         f1 = torch.stft(x.squeeze(1), n_fft=512, hop_length=160, win_length=512, window=torch.hann_window(window_length=512).cuda(), return_complex=True)
#         s_t1 = torch.abs(f1).unsqueeze(1).permute(0,1,3,2)
        f2 = torch.stft(x.squeeze(1), n_fft=1536, hop_length=480, win_length=1536, window=torch.hann_window(window_length=1536).cuda(), return_complex=True)
#        f2 = torch.stft(x.squeeze(1), n_fft=1024, hop_length=160, win_length=1024, window=torch.hann_window(window_length=1024), return_complex=True)
        s_t2 = torch.abs(f2).unsqueeze(1).permute(0,1,3,2)
        results = []
        for key, disc in self.model.items():
            if key == 'wav_disc_0':
                wave_disc = disc(x)
                for i_w, disc_w in wave_disc.items():
                    results.append(disc_w)
            elif key == 'stft_disc_0':
                results.append(disc(s_t))
#             elif key == 'stft_disc_1':
#                 results.append(disc(s_t1))
            elif key == 'stft_disc_2':
                results.append(disc(s_t2))
                
        return results

if __name__ == '__main__':
    model = Discriminator(norm='spectral_norm')
    '''
    input: torch.Size([3, 1, 2560])
    output: torch.Size([])
    '''

    x = torch.randn(3, 1, 7680)
#    x_1 = x.squeeze(1)
#    print(str(x_1.shape))
    x1 = torch.randn(3, 3, 44, 256)
    out = model(x)
#    print(out)
    flops, params = profile(model, inputs=[x,])
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

    macs, params = get_model_complexity_info(model, (1, 7680), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)

