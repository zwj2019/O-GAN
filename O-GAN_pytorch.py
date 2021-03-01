import os

import numpy as np
import imageio
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F


if not os.path.exists('samples'):
    os.mkdir('samples')


imgs = '/home/gl/ZWJ_FaceDataset/face_dataset/FFHQ'

img_dim = 128
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = 64


def imread(file_path):
    with open(file_path, 'rb') as f:
        img_data = Image.open(f)

        if img_data.mode != 'RGB':
            r, g, b, _ = img_data.split()
            img_data = Image.merge(mode='RGB', bands=[r, g, b])

        if img_data.size != (img_dim, img_dim):
            img_data = img_data.resize((img_dim, img_dim))

    tensor = F.to_tensor(img_data) * 2 - 1

    return tensor


class FaceDataset(Dataset):
    def __init__(self, data_root):
        self.image_list = [os.path.join(data_root, file_name) for file_name in os.listdir(data_root)]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return imread(self.image_list[index])


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)

# 编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.ModuleList()

        for i in range(num_layers + 1):
            in_channels = max_num_channels // 2 ** (num_layers - i + 1) if i > 0 else 3
            out_channels = max_num_channels // 2 ** (num_layers - i)
            print(in_channels, out_channels)
            self.net.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(2, 2), padding=1))

            if i > 0:
                self.net.append(nn.BatchNorm2d(out_channels))

            self.net.append(nn.LeakyReLU(0.2))

        self.linear = nn.Linear(1024 * 4 * 4, z_dim)

    def forward(self, x):
        print(x.size())
        for layer in self.net:
            x = layer(x)
            print(x.size())
        # print(x.size())
        x = x.view(-1, 1024 * 4 * 4)
        return self.linear(x)


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(128, f_size ** 2 * max_num_channels)
        self.bn = nn.BatchNorm2d(max_num_channels)
        self.relu = nn.ReLU()

        self.net = nn.ModuleList()
        for i in range(num_layers):
            in_channels = max_num_channels // 2 ** i
            out_channels = max_num_channels // 2 ** (i + 1)
            self.net.append(nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size=4, stride=2, padding=1))
            self.net.append(nn.BatchNorm2d(out_channels))
            self.net.append(nn.ReLU())

        self.conv_t = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, max_num_channels, f_size, f_size)
        x = self.bn(x)
        x = self.relu(x)
        for layer in self.net:
            x = layer(x)
        x = self.conv_t(x)
        x = self.tanh(x)
        return x


def correlation(x, y):
    x = x - torch.mean(x, 1, True)
    y = y - torch.mean(y, 1, True)
    x = nn.functional.normalize(x, p=2, dim=1)
    y = nn.functional.normalize(y, p=2, dim=1)
    return torch.sum(x * y, 1, keepdim=True)


g_model = Generator()
g_model.apply(weight_init)
e_model = Encoder()
e_model.apply(weight_init)
dataset = FaceDataset(imgs)
dataloader = DataLoader(dataset, shuffle=True, batch_size=64, num_workers=4, drop_last=True)
optimizer = optim.RMSprop([e_model.parameters(), g_model.parameters()], lr=1e-4)

# 重构采样函数
def sample_ae(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = imread(os.path.join(imgs, np.random.choice(os.listdir(imgs))))
                x_sample = torch.unsqueeze(x_sample, dim=0)
            else:
                z_sample = e_model(x_sample)
                z_sample -= torch.mean(z_sample, dim=1, keepdim=True)
                z_sample /= torch.std(z_sample, dim=1, keepdim=True)
                x_sample = g_model(z_sample * 0.9)

            digit = x_sample[0].cpu().numpy().transpose((1, 2, 0))
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)

nn.MSELoss
for epoch in range(1000):
    for i, x_real in enumerate(dataloader):
        optimizer.zero_grad()

        z_real = e_model(x_real)
        z_fake = torch.randn(x_real.size(0), 128)
        z_fake_ng = z_fake.detach()

        x_fake = g_model(z_fake)
        x_fake_ng = x_fake.detach()

        t1_loss = torch.mean(z_real) - torch.mean(z_fake_ng)
        t2_loss = torch.mean(z_fake) - torch.mean(z_fake_ng)

        z_corr = correlation(z_fake, x_fake)
        qp_loss = 0.25 * t1_loss[:, 0] ** 2 / torch.mean((x_real - x_fake_ng) ** 2, [1, 2, 3])

        loss = torch.mean(t1_loss + t2_loss - 0.5 * z_corr) + torch.mean(qp_loss)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or i % 500 == 0:
            sample_ae('samples/test_ae_%d_%d.png' % (epoch, i))

