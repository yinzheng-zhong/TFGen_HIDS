import math
import time

import torch
from torch.utils.data import DataLoader
import numpy as np


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, x, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        # reshape the data back to 2D
        size = int(math.sqrt(x.shape[1]))
        x = np.reshape(x, (x.shape[0], 1, size, size))

        self.x = x
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.x[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)

        return torch.from_numpy(sample), idx


class ConvAutoencoder(torch.nn.Module):
    """
    Convolutional Autoencoder
    """

    def __init__(
            self,
            learning_rate=1e-3,
            epochs=10,
            batch_size=32,
            weight_decay=1e-5,
    ):
        """
        Initialize the autoencoder
        :param input_size: (int) The size of the input
        :param hidden_size: (int) The size of the hidden layer
        :param output_size: (int) The size of the output
        """
        super(ConvAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=1),  # (53 - 3) / 1 + 1 = 51
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(16, 32, 3, stride=2),  # (51 - 3) / 2 + 1 = 25
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(32, 16, 3, stride=2),  # (25 - 3) / 2 + 1 = 12
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),

            torch.nn.MaxPool2d(2, stride=1, padding=1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(32, 16, 3, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),

            torch.nn.ConvTranspose2d(16, 1, 3, stride=1),
            torch.nn.Sigmoid()
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = None

    def forward(self, x):
        """
        Forward pass of the autoencoder
        :param x: (tensor) The input to the autoencoder
        :return: (tensor) The output of the autoencoder
        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def fit(self, x):
        """
        Fit the autoencoder to the data
        :param x: (tensor) The input to the autoencoder
        :return: (tensor) The output of the autoencoder
        """

        dataset = PyODDataset(x=x)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = self.to(self.device)

        self._train_autoencoder(train_loader)

    def _train_autoencoder(self, train_loader):
        """
        Train the autoencoder
        :param train_loader: (DataLoader) The dataloader for the training data
        :return: (tensor) The output of the autoencoder
        """
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            for data, data_idx in train_loader:
                data = data.to(self.device)
                outputs = self.model(data)

                loss = criterion(outputs, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch [{}/{}], Loss: {}'.format(epoch + 1, self.epochs, loss.item()))

    def decision_function(self, x):
        # note the shuffle may be true but should be False
        dataset = PyODDataset(x=x)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=32,
                                                 shuffle=False)

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([x.shape[0], ])
        with torch.no_grad():
            for data, data_idx in dataloader:
                data = data.to(self.device)
                outputs = self.model(data)

                # outlier_scores
                outlier_scores[data_idx] = (outputs - data).abs().sum(dim=(1, 2, 3)).cpu().numpy()

        return outlier_scores
