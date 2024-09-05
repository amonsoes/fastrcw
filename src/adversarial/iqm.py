import torch
import csv

from math import log10
from IQA_pytorch import MAD, SSIM
from torch.nn import functional as F
from DISTS_pytorch import DISTS
from torchmetrics.image import VisualInformationFidelity as VIF
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM


class ImageQualityMetric:
    
    def __init__(self, metrics, filepath, device='cpu', n_channels=3):
        if not isinstance(metrics, list):
            raise ValueError('Input for ImageQualityMetric should be list.')
        self.mapping = {
            'mad' : 0,
            'ssim' : 1,
            'psnr' : 2,
            'dists' : 3,
            'vif' : 4,
            'msssim' : 5,
        }
        self.metrics_dict = {}
        self.device = device
        self.filepath = filepath
        if self.filepath:
            self.filepath = "/".join(filepath.split('/')[:-2]) + '/' + 'iqa_results.csv' # put results in this csv
            with open(self.filepath, 'w') as results_file:
                results_obj = csv.writer(results_file)
                results_obj.writerow([key for key, _ in sorted(self.mapping.items(), key=lambda x:x[1])])
        for metric in metrics:
            if metric == 'mad':
                self.mad = MAD(channels=n_channels)
                self.metrics_dict['mad'] = self.mad
            elif metric == 'ssim':
                self.ssim = SSIM(channels=n_channels)
                self.metrics_dict['ssim'] = self.ssim
            elif metric == 'psnr':
                self.psnr = PSNR(channels=n_channels)
                self.metrics_dict['psnr'] = self.psnr
            elif metric == 'dists':
                self.dists = DISTS().to(self.device)
                self.metrics_dict['dists'] = self.dists
            elif metric == 'vif':
                self.vif = VIF()
                self.metrics_dict['vif'] = self.vif
            elif metric == 'msssim':
                self.msssim = MSSSIM(data_range=1.0).to(self.device)
                self.metrics_dict['msssim'] = self.msssim
        self.n_channels = n_channels
        self.mad_total = 0
        self.ssim_total = 0
        self.dists_total = 0
        self.n = 0
        
    def __call__(self, ref_image, adv_image):
        row = ['n/a'] * len(self.mapping)
        if len(ref_image.shape) < 4:
            ref_image = ref_image.unsqueeze(0)
        if len(adv_image.shape) < 4:
            adv_image = adv_image.unsqueeze(0)
        for name, iqa_fn in self.metrics_dict.items():
            result = iqa_fn(ref_image, adv_image)
            row[self.mapping[name]] = float(result)
            if name == 'mad':
                #mad_r = torch.tanh((1/10)*result)
                mad_r = result
                self.mad_total += result
            elif name == 'ssim':
                ssim_r = result
                self.ssim_total += result
            elif name == 'dists':
                dists_r = result
                self.dists_total += result
        if self.filepath:
            with open(self.filepath, 'a') as results_file:
                results_obj = csv.writer(results_file)
                results_obj.writerow(row)
        self.n += 1
        return mad_r.item(), ssim_r.item(), dists_r.item()
    
    def get_avg_mad(self):
        return self.mad_total / self.n

    def get_avg_ssim(self):
        return self.ssim_total / self.n

    def get_avg_dists(self):
        return self.dists_total / self.n

def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
    return out
    
    

class PSNR(torch.nn.Module):
    def __init__(self, channels=3):
    
        super(PSNR, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, X, Y, as_loss=True):
        assert X.shape == Y.shape
        if as_loss:
            score = self.psnr(X, Y)
            norm_score = score / 255 # to better balance the other loss
            return 1 - norm_score # return for min-optimization
        else:
            with torch.no_grad():
                score = self.psnr(X, Y)
            return score
    
    def psnr(self, X, Y):
        """
        This returns the PSNR in decibels. The higher the better
        so it needs to be inverted for optimization (minimization)
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        the maximum for pixel vals is 255 but normalized to 1.0.
        
        For color images with three RGB values per pixel, the definition 
        of PSNR is the same except that the MSE is the sum over all 
        squared value differences, divided by image 
        size and by three. For 8bit the maximum PSNR is 255
        """
        mse_score = self.mse(X, Y)
        psnr_score = 20*log10(1) - 10*torch.log10((mse_score/3)/224)
        return psnr_score
    