import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
import librosa

import pdb

device = torch.device('cuda:0')

def SNR(pred, target):
    return (20 * torch.log10(
        torch.norm(target, dim=-1).clamp(min=1e-8) / torch.norm(pred - target, dim=-1).clamp(min=1e-8))).mean()

def aymmetric_mse(pred, target):
    log_mag_eror = torch.abs(pred - target)
    threhold = 10
    positive_index = 20 * pred < 20 * target + threhold
    negtiave_index = 20 * pred >= 20 * target + threhold
#     print('neg:', str(negtiave_index))
#     print('pos:', str(positive_index))
    loss = torch.sum(log_mag_eror[positive_index])
    x = log_mag_eror[negtiave_index]

    loss += torch.sum(x + x ** 2)
    return loss/(pred.shape[0]*pred.shape[1]*pred.shape[2])

def LSD(pred, target, nfft=2048, hop=512):
    window = torch.hann_window(nfft).to(pred.device)
    stft_p = torch.stft(pred, nfft, hop, window=window, return_complex=False)
    stft_t = torch.stft(target, nfft, hop, window=window, return_complex=False)

    mag_p = torch.norm(stft_p, p=2, dim=-1)
    mag_t = torch.norm(stft_t, p=2, dim=-1)

    sp = torch.log10(mag_p.square().clamp(1e-8))
    st = torch.log10(mag_t.square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window.to(x.device))
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def stft_cpx(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window.to(x.device))
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return real.transpose(2, 1), imag.transpose(2, 1), x_stft.transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag).clamp(1e-7), torch.log(x_mag).clamp(1e-7))



class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=960, shift_size=480, win_length=960, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window).to(x.device)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window).to(x.device)
        #         x_mag = self.melspec(x)
        #         y_mag = self.melspec(y)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss

class STFTLoss1(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss1, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss1()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss1()
        self.n_mels = fft_size // 4
        if self.n_mels > 64:
            self.n_mels = 64
        self.n_mels = 64
        self.melspec = MelSpectrogram(sample_rate=16000, n_fft=self.fft_size, hop_length=self.shift_size, n_mels=self.n_mels)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
#         x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
#         y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
#         x_mag = librosa.feature.melspectrogram(S=x_stft, sr = 16000, n_mels=self.n_mels)
#         y_mag = librosa.feature.melspectrogram(S=y_stft, sr = 16000, n_mels=self.n_mels)
        melspec = self.melspec.to(x.device)
        x_mag = torch.clamp(melspec(x), min=1e-10)
        y_mag = torch.clamp(melspec(y), min=1e-10)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss    
    
class MultiResolutionSTFTLoss_asymmetric(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 256, 2048, 512, 1024, 512],
                 hop_sizes=[120, 96, 240, 128, 256, 50],
                 win_lengths=[600, 256, 1200, 512, 1024, 240],
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f, s in zip(self.stft_losses, self.fft_sizes):
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l #* (s/2)**0.5
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss + mag_loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 256, 2048, 512, 1024, 512],
                 hop_sizes=[120, 96, 240, 128, 256, 50],
                 win_lengths=[600, 256, 1200, 512, 1024, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f, s in zip(self.stft_losses, self.fft_sizes):
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l  # * (s / 2) ** 0.5
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss + mag_loss

class STFTMag(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTMag, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)

        return x_mag, y_mag


class MultiSTFTMag(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[64, 128, 256, 512, 1024, 2048],
                 hop_sizes=[16, 32, 64, 128, 256, 512],
                 win_lengths=[64, 128, 256, 512, 1024, 2048],
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiSTFTMag, self).__init__()
        self.stft_mags = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_mags += [STFTMag(fs, ss, wl, window)]

    def forward(self, x, y):
        x_mags = []
        y_mags = []
        for f in self.stft_mags:
            x_mag, y_mag = f(x, y)
            x_mag = torch.reshape(x_mag[:, :, :-1], (x_mag.shape[0], 44, 256))
            y_mag = torch.reshape(y_mag[:, :, :-1], (y_mag.shape[0], 44, 256))
            x_mags.append(x_mag)
            y_mags.append(y_mag)
        x_mags = torch.stack(x_mags, dim=1)
        y_mags = torch.stack(y_mags, dim=1)
        return x_mags, y_mags


def osisnr(source_x, estimate_source_x, fft_size, hop_size, win_length):
    """implement optimized freq-domain si-snr loss function interface"""
    # source: B T
    # estimate_source: B T
    EPS = 1e-7
    window_fn = getattr(torch, "hann_window")(int(win_length))
    #     source = trans(source)
    source = torch.stft(source_x, fft_size, hop_size, win_length, window_fn.to(source_x.device))
    source = (((source[:, :, :, 0]).pow(2) + (source[:, :, :, 1]).pow(2)) + EPS) ** 0.5
    #     estimate_source = trans(estimate_source)
    estimate_source = torch.stft(estimate_source_x, fft_size, hop_size, win_length,
                                 window_fn.to(estimate_source_x.device))
    estimate_source = (((estimate_source[:, :, :, 0]).pow(2) + (estimate_source[:, :, :, 1]).pow(2)) + EPS) ** 0.5
    source = source.permute(0, 2, 1)
    estimate_source = estimate_source.permute(0, 2, 1)
    if len(source.shape) < 3:
        source = source.unsqueeze(-1)
        estimate_source = estimate_source.unsqueeze(-1)
    len_min = min(source.size(1), estimate_source.size(1))
    source = source[:, :len_min, 1:]
    estimate_source = estimate_source[:, :len_min, 1:]
    B, T, F = source.size()
    # bin_mask = torch.gt(torch.mean(source, dim=1, keepdim=True), 1.0e-4)
    # bin_mask = bin_mask.permute(0,2,1).contiguous().view(B*F)

    # Step 2. SI-SNR
    s_target = source - torch.mean(source, dim=(1, 2), keepdim=True)  # [B, T, F]
    s_estimate = estimate_source - torch.mean(estimate_source, dim=(1, 2), keepdim=True)  # [B, T, F]

    # s_target = source  # [B, T, F]
    # s_estimate = estimate_source   # [B, T, F]

    # s_target = <s', s>s / ||s||^2
    # pair_wise_dot = torch.einsum("ijk,ikj->ijj", [s_estimate.permute(0,2,1), s_target]) # trace and matrix innter product [B 1 1]
    pair_wise_dot = torch.matmul(s_estimate.permute(0, 2, 1), s_target.to(s_estimate.dtype))
    pair_wise_dot = torch.einsum('bii->b', pair_wise_dot)
    pair_wise_dot = pair_wise_dot.unsqueeze(-1).unsqueeze(-1) + EPS
    # print(f"pair_wise_dot shape {pair_wise_dot.shape}")
    s_estimate_energy = torch.sum(s_estimate ** 2, dim=(1, 2), keepdim=True)  # [B, 1, C]
    # ||s'||^2 * s // <s, s'>
    pair_wise_proj = s_target * s_estimate_energy / pair_wise_dot  # [B, T, F]
    sin_angle = torch.clamp(1.0 - (torch.sum(s_estimate ** 2, dim=(1, 2)) /
                                   (torch.sum(pair_wise_proj ** 2, dim=(1, 2)) + EPS)),
                            min=0.001,
                            max=0.999)
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = 10 * torch.log10(1.0 / (sin_angle + EPS))  # [B]
    # pair_wise_si_snr = torch.mean(pair_wise_si_snr, dim=1) # [B]
    loss_sisnr = -torch.mean(pair_wise_si_snr)
    # print(loss_sisnr, sin_angle, torch.sum(s_estimate ** 2, dim=(1,2)), ((torch.sum(pair_wise_proj ** 2, dim=(1,2)) + EPS)))
    # print(f"loss_sisnr {loss_sisnr}")
    return loss_sisnr


class MultiOSISNR(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiOSISNR, self).__init__()

        self.fft_sizes = [64, 128, 256, 512, 1024, 2048]
        self.hop_sizes = [16, 32, 64, 128, 256, 512],
        self.win_lengths = [64, 128, 256, 512, 1024, 2048],

    def forward(self, x, y):
        loss = 0
        for idx in range(6, 12):
            #             print(idx)
            #             pdb.set_trace()
            loss += osisnr(x, y, 2 ** idx, 2 ** idx // 4, 2 ** idx)

        return loss


class SpectralConvergengeLossCpx(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLossCpx, self).__init__()

    def forward(self, x_real, x_imag, y_real, y_imag, x_stft, y_stft):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        #         spec_loss = torch.mean(torch.log(((x_real - y_real) + (x_imag - y_imag))**2 + 1e-10), -1)
        #         spec_loss = torch.mean(spec_loss, [0, 1])

        spec_loss = F.mse_loss(x_stft, y_stft)
        #         pdb.set_trace()
        #         print(spec_loss)
        return spec_loss


class STFTLossCpx(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLossCpx, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLossCpx()
        #         self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.melspec = MelSpectrogram(sample_rate=16000, n_fft=self.fft_size, hop_length=self.shift_size, n_mels=64)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_real, x_imag, x_stft = stft_cpx(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_real, y_imag, y_stft = stft_cpx(y, self.fft_size, self.shift_size, self.win_length, self.window)
        #         x_mag = self.melspec(x)
        #         y_mag = self.melspec(y)
        sc_loss = self.spectral_convergenge_loss(x_real, x_imag, y_real, y_imag, x_stft, y_stft)
        #         mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss


class MultiResolutionSTFTLossCpx(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[64, 128, 256, 512, 1024, 2048],
                 hop_sizes=[16, 32, 64, 128, 256, 512],
                 win_lengths=[64, 128, 256, 512, 1024, 2048],
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLossCpx, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLossCpx(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f, s in zip(self.stft_losses, self.fft_sizes):
            sc_l = f(x, y)
            sc_loss += sc_l * (s / 2) ** 0.5
        #             mag_loss += mag_l * (s/2)**0.5
        #         sc_loss /= len(self.stft_losses)
        #         mag_loss /= len(self.stft_losses)

        return sc_loss