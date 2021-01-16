import torch
import torch.nn as nn
import torch.nn.functional as F

# CartoonGAN model architecture

class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding, padding_mode = 'reflect')
        self.conv1_norm = InstanceNormalization(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding, padding_mode = 'reflect')
        self.conv2_norm = InstanceNormalization(channel)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))

        return input + x # elementwise sum

# Original model uses this class and pretrained weights include these parameters.
# Cannot replace with nn.InstanceNorm2d if using pretrained weights since results
# produced were not desirable.
class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def __call__(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out

class Generator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32, nb=6):
        super(Generator, self).__init__()
        self.down_convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3, padding_mode = 'reflect'), # k7n64s1
            InstanceNormalization(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1), # k3n128s2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), # k3n128s1
            InstanceNormalization(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1), # k3n256s1
            nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), # k3n256s1
            InstanceNormalization(nf * 4),
            nn.ReLU(True),
        )

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(nf * 4, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1), # k3n128s1/2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), # k3n128s1
            InstanceNormalization(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1), # k3n64s1/2
            nn.Conv2d(nf, nf, 3, 1, 1), # k3n64s1
            InstanceNormalization(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3, padding_mode = 'reflect'), # k7n3s1
            nn.Tanh(),
        )

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)

        return output