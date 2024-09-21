import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

from tools import ConvBlock, InvConvBlock, MyMNIST, to2d


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_dim = 16

        # encoder
        block = ConvBlock(1, 28, 28)
        self.conv1 = block.nextLayer(out_channels=4, kernel_size=3, stride=2, padding=1)
        self.conv2 = block.nextLayer(out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv3 = block.nextLayer(out_channels=16, kernel_size=3, stride=2, padding=1)
        self.fc_enc1 = nn.Linear(in_features=16 * 4 * 4, out_features=128)
        self.fc_mu = nn.Linear(in_features=128, out_features=self.z_dim)
        self.fc_sigma = nn.Linear(in_features=128, out_features=self.z_dim)

        # decoder
        self.fc_dec1 = nn.Linear(in_features=self.z_dim, out_features=128)
        self.fc_dec2 = nn.Linear(in_features=128, out_features=16 * 4 * 4)
        self.inv_start = InvConvBlock(16, 4, 4)
        block = InvConvBlock(16, 4, 4)
        self.iconv1 = block.nextLayer(out_channels=8, kernel_size=3, stride=2, padding=1)
        self.iconv2 = block.nextLayer(out_channels=4, kernel_size=3, stride=2, padding=1)
        self.iconv3 = block.nextLayer(out_channels=2, kernel_size=3, stride=2, padding=1)
        self.iconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.optimizer = opt.Adam(self.parameters(), lr=1e-3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def totalParametersNum(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_enc1(x))
        mu = self.fc_mu(x)
        log_var = self.fc_sigma(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # sample from Gaussian distributions of (mu, sigma)
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z

    def decode(self, z):
        z = F.relu(self.fc_dec1(z))
        z = F.relu(self.fc_dec2(z))
        z = z.view(z.size(0), *self.inv_start.full_size)
        z = self.iconv1(z)
        z = self.iconv2(z)
        z = self.iconv3(z)
        z = self.iconv4(z)
        return z

    def loss_function(self, x, x_reconstruct, mu, log_var):
        """
        Note:
        L = BCE + alpha * KLD
        low alpha (~1e-4) => accurate reconstruction, but unnatural generation
        high alpha (~1e-1) => less accurate reconstruction, better generation, but it tends to learn a "mean value"
        """
        # BCE = F.binary_cross_entropy(x_reconstruct, x, reduction='sum')
        BCE = F.mse_loss(x_reconstruct, x)
        KLD = -0.5 * torch.mean(1 + log_var - torch.pow(mu, 2) - torch.exp(log_var))  # original: torch.sum
        return BCE + KLD * 2e-3

    def singleEpoch(self, x: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        x = x.to(self.device)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_pred = self.decode(z)
        loss = self.loss_function(x, x_pred, mu, log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device)
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_pred = self.decode(z)
        return x_pred

    def testReconstruct(self, x: torch.Tensor) -> float:
        loss = F.mse_loss(self.reconstruct(x), x.to(self.device))
        return loss.item()


class TrainCVAE:
    # best batch_size: around 10000
    def __init__(self, debug=True):
        self.model = CVAE()
        self.batch_size = 64
        self.train_set = MyMNIST(train=True).minmaxNormalize(0.1, 0.9)
        self.test_set = MyMNIST(train=False).minmaxNormalize(0.1, 0.9)
        num_workers = 0 if debug else 4
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                       num_workers=num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=1024, shuffle=True,
                                      num_workers=num_workers)

    def train(self, epoches: int) -> (list[float], list[float]):
        print('Num of parameters:', self.model.totalParametersNum())
        losses, v_losses = [], []
        for epoch in range(epoches):
            loss = 0
            for i, data in enumerate(self.train_loader):
                loss += self.model.singleEpoch(data)
            print(f'epoch: {epoch}, loss: {loss}')
            losses.append(loss)
            v_losses.append(self.validate())
        return losses, v_losses

    def validate(self) -> float:
        for i, data in enumerate(self.test_loader):
            loss = self.model.testReconstruct(data)
            print('validation: pure mse loss:', loss)
            break
        return loss

    def reconstructGrid(self):
        sample_data = next(iter(self.test_loader))[:8]
        reconstructed_data = self.model.reconstruct(sample_data)
        img = np.vstack((to2d(sample_data), to2d(reconstructed_data)))
        return img

    def generateGrid(self):
        sample_data = next(iter(self.test_loader))[:3]
        with torch.no_grad():
            sample_z = self.model.reparameterize(*self.model.encode(sample_data.to(self.model.device)))
            za, zb, zc = sample_z.to('cpu').numpy()

        def f(x, y):
            # x, y in [0, 1]
            return (1 - x - y) * za + x * zb + y * zc

        xs = np.linspace(0, 1, 7)
        ys = np.linspace(0, 1, 7)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros((*X.shape, za.shape[0]))
        m, n = len(xs), len(ys)
        for i in range(m):
            for j in range(n):
                Z[i, j, :] = f(X[i, j], Y[i, j])
        with torch.no_grad():
            z = torch.Tensor(Z.reshape(m * n, -1)).to(self.model.device)
            imgs = self.model.decode(z).to('cpu').numpy()
        imgs = imgs.reshape(m, n, -1)
        lst = [[imgs[i, j, :].reshape(28, 28) for j in range(n)] for i in range(m)]
        return np.vstack(list(map(np.hstack, lst)))


model = TrainCVAE()
losses, v_losses = model.train(10)
plt.plot(losses)
plt.plot(v_losses)
plt.show()
plt.imshow(model.reconstructGrid())
plt.show()
plt.imshow(model.generateGrid())
plt.show()
