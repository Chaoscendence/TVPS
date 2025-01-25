import torch
import math
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import KLDivLoss
import torch.nn.functional as F
import numpy as np
import random


class ProbNet(nn.Module):
    def __init__(self, gaussian_shape, num_classes, temp, hard):
        super(ProbNet, self).__init__()
        self.temp = temp
        self.hard = hard
        channel_shape = 512
        data_shape = 16
        y_dim = 1*16*16
        self.num_classes = num_classes
        self.gaussian_shape = gaussian_shape
        self.encode_layer = nn.Linear(channel_shape*data_shape*data_shape, y_dim)
        self.st_calculate = nn.Linear(y_dim, num_classes)
        self.mu_calculate = nn.Linear(y_dim, gaussian_shape * num_classes)
        self.var_calculate = nn.Linear(y_dim, gaussian_shape * num_classes)
        self.decode_layer = nn.Linear(gaussian_shape, channel_shape*data_shape*data_shape)

  
    def reparameterize(self, mu, logvar, pi_chosen):
        pi_chosen = pi_chosen.unsqueeze(-1)
        mu_chosen = torch.bmm(pi_chosen.permute(0,2,1), mu).squeeze(-2)
        logvar_chosen = torch.bmm(pi_chosen.permute(0,2,1), logvar).squeeze(-2)
        std = torch.exp(0.5 * logvar_chosen)
        eps = torch.randn_like(mu_chosen)
        z = mu_chosen + eps * std
        
        return z
    
    def forward(self, x, local_rank):
        x = x.view(x.shape[0], -1)
        qyx = self.encode_layer(x)
        mu = self.mu_calculate(qyx).view(qyx.shape[0], self.num_classes, self.gaussian_shape)
        logvar = self.var_calculate(qyx).view(qyx.shape[0], self.num_classes, self.gaussian_shape)
        logvar = torch.clamp(logvar, max=40.)
        st = F.softmax(self.st_calculate(qyx), dim=1)
        pi_chosen = F.gumbel_softmax(st, tau=1., hard=True)

        z = self.reparameterize(mu, logvar, pi_chosen)
        z = self.decode_layer(z)
        
        output = {'mean': mu, 'logvar': logvar, 'z': z, 'st': st}
        return output

class double_conv(nn.Module):
    '''Conv => BN =>ReLU => Conv1d => BN => ReLU
    '''
    def __init__(self, in_ch, out_ch, kernel, width):
        super(double_conv, self).__init__()
        conv_layers = []
        for _ in range(width):
            conv_layers += [nn.Conv2d(in_ch, out_ch,kernel_size = kernel, padding=math.floor(kernel*0.5))]#, padding=math.floor(kernel*0.5)
            conv_layers += [nn.BatchNorm2d(out_ch)]
            conv_layers += [nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

    def forward(self,x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    '''downsample encode input
    '''
    def __init__(self, out_ch, kernel, width):
        super(inconv, self).__init__()
        self.conv = double_conv(2, out_ch, kernel, width)

    def forward(self, x):
        x = self.conv(x)
        return x

class encode(nn.Module):
    '''downsample encode
    '''
    def __init__(self, in_ch, out_ch, kernel, width):
        super(encode, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_conv(in_ch, out_ch, kernel, width)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class decode(nn.Module):
    '''upsample decode
    '''
    def __init__(self, in_ch, out_ch, kernel, width):
        super(decode, self).__init__()
        # transpose
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch, kernel, width)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class gaussianloss(nn.Module):
    def _init__(self):
        super(gaussianloss).__init__()
    
    def forward(self, output, num_components, gaussian_dimension):
        mu = output['mean']
        logvar = output['logvar']
        st = output['st'].unsqueeze(-1)

        logvar = torch.clamp(logvar, max=40)
        gauss_num = st.shape[1]
        chosen = torch.ones_like(st)

        st = st * chosen
        kl_div = -0.5 * (-2.*np.log(num_components) + gaussian_dimension + torch.sum(st * (logvar - torch.pow(mu, 2) - torch.exp(logvar))) - torch.sum(2. * st * torch.log(st + 1e-20)))

        return kl_div

    def log_normal(self, x, mu, var):
        var = var + 1e-10
        return -0.5 * torch.sum(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)


class Unet(nn.Module):
    def __init__(self, settings, gauss):
        super(Unet, self).__init__()
        self.gaussian_shape = gauss
        self.temp = 1.0
        self.hard = 0
        self.fix = settings['mask_fix']
        self.kernel = settings['kernel']
        self.width = settings['width']
        self.var_shape = settings['var_shape']
        self.layer_En = self.Encoder_layer()
        self.layer_p = ProbNet(gaussian_shape=self.var_shape, num_classes=self.gaussian_shape, temp=self.temp, hard=self.hard)
        self.layer_De = self.Decoder_layer()
        self.kl = gaussianloss()


    def forward(self, x, training_G, local_rank):
        Encoder_output = []
        for layer in self.layer_En:
            input_shape = x.shape
            x = layer(x)
            output_shape = x.shape
            if input_shape != output_shape:
                Encoder_output.append(x)
        tensor_shape = x.shape

        output = self.layer_p(x, local_rank)
        x_rec = output['z']

        kl_div = self.kl(output, self.gaussian_shape, self.var_shape)

        x = x_rec.view(tensor_shape)
        layer_count = len(self.layer_De)
        for idx in range(layer_count, 1, -1):
            x = self.layer_De[layer_count-idx](x, Encoder_output[idx-2])
        x = self.layer_De[layer_count-1](x)

        if training_G:
            return {'x':x, 'kl':kl_div}
        else:
            return x

    def Encoder_layer(self):
        layer_E = []
        layer_E.append(inconv(16, self.kernel, self.width))
        layer_E.append(encode(16, 32, self.kernel, self.width))
        layer_E.append(encode(32, 64, self.kernel, self.width))
        layer_E.append(encode(64, 128, self.kernel, self.width))
        layer_E.append(encode(128, 256, self.kernel, self.width))
        layer_E.append(encode(256, 512, self.kernel, self.width))
        
        return nn.Sequential(*layer_E)


    def Decoder_layer(self):
        layer_D = []
        layer_D.append(decode(512, 256, self.kernel, self.width))
        layer_D.append(decode(256, 128, self.kernel, self.width))
        layer_D.append(decode(128, 64, self.kernel, self.width))
        layer_D.append(decode(64, 32, self.kernel, self.width))
        layer_D.append(decode(32, 16, self.kernel, self.width))
        layer_D.append(outconv(16))

        return nn.Sequential(*layer_D)