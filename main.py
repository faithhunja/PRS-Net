from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
from os.path import join, getsize
from os.path import dirname, join as pjoin
import scipy.io as sio

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution layer followed by max pooling layer, them Leaky ReLU
        self.conv1 = nn.Conv3d(1, 4, 3, 1, 1)
        self.maxpool1 = nn.MaxPool3d(2)
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv3d(4, 8, 3, 1, 1)
        self.maxpool2 = nn.MaxPool3d(2)
        self.leakyrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv3d(8, 16, 3, 1, 1)
        self.maxpool3 = nn.MaxPool3d(2)
        self.leakyrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv3d(16, 32, 3, 1, 1)
        self.maxpool4 = nn.MaxPool3d(2)
        self.leakyrelu4 = nn.LeakyReLU(0.1)
        self.conv5 = nn.Conv3d(32, 64, 3, 1, 1)
        self.maxpool5 = nn.MaxPool3d(2)
        self.leakyrelu5 = nn.LeakyReLU(0.1)
        # 3 branches of fully connected layers
        self.fc11 = nn.Linear(64, 32)
        self.fc12 = nn.Linear(32, 16)
        self.fc13 = nn.Linear(16, 4)
        self.fc21 = nn.Linear(64, 32)
        self.fc22 = nn.Linear(32, 16)
        self.fc23 = nn.Linear(16, 4)
        self.fc31 = nn.Linear(64, 32)
        self.fc32 = nn.Linear(32, 16)
        self.fc33 = nn.Linear(16, 4)
    # Forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.leakyrelu1(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.leakyrelu2(x)
        print(x.shape)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.leakyrelu3(x)
        print(x.shape)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.leakyrelu4(x)
        print(x.shape)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.leakyrelu5(x)
        print(x.shape)
        x = torch.reshape(x, (1,64))
        print(x.shape)
        x1 = self.fc11(x)
        x1 = self.fc12(x1)
        x1 = self.fc13(x1)
        x2 = self.fc21(x)
        x2 = self.fc22(x2)
        x2 = self.fc23(x2)
        x3 = self.fc31(x)
        x3 = self.fc32(x3)
        x3 = self.fc33(x3)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def loadMatDataset(x):
    for filename in os.listdir('/home/faith/Downloads/Documents/train'):
        data_dir = pjoin(dirname(sio.__file__), '/home/faith/Downloads/Documents/train')
        mat_fname = pjoin(data_dir, filename)
        mat_contents = sio.loadmat(mat_fname)
        sorted(mat_contents.keys())
        x = mat_contents['Volume'];

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    dataset1 = datasets.MNIST('/home/faith/Downloads/Compressed', train=True, download=True)
    dataset2 = datasets.MNIST('/home/faith/Downloads/Compressed', train=False)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
