skip_training = True

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from PIL import Image

data_dir = '.'

device = torch.device('cuda')

def plot_classification(support_query, classes):
    fig, axs = plt.subplots(2, 3, figsize=(8, 5))
    axs = axs.flatten()
    for im, ix in zip(support_query.view(6, 28, 28), [0, 3, 1, 4, 2, 5]):
        axs[ix].imshow(im, cmap=plt.cm.Greys)

    colors = ['red', 'green', 'blue']
    for i, ax in enumerate(axs[:3]):
        ax.set_xticks([])
        ax.set_yticks([])
        set_axes_color(ax, colors[i])

    for i, ax in enumerate(axs[3:]):
        ax.set_xticks([])
        ax.set_yticks([])
        set_axes_color(ax, colors[classes[i]])

def save_model(model, filename):
        do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
        if do_save == 'yes':
            torch.save(model.state_dict(), filename)
            print('Model saved to %s.' % (filename))
        else:
            print('Model not saved.')


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()


if skip_training:
    device = torch.device("cpu")

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.Omniglot(root=data_dir, download=True, transform=transform)

x, y = dataset[0]
print(x.shape, y)

fig, ax = plt.subplots(1, figsize=(3, 3))
ax.matshow(1-x[0], cmap=plt.cm.Greys)

class OmniglotFewShot(Dataset):

    def __init__(self, root, n_support, n_query,
                 transform=transforms.Compose([
                     transforms.Resize(28),
                     transforms.ToTensor(),
                     transforms.Lambda(lambda x: 1-x),
                 ]),
                 mix=False,  # Mix support and query examples
                 train=True
                ):

        assert n_support + n_query <= 20, "Omniglot contains only 20 images per character."
        self.n_support = n_support
        self.n_query = n_query
        self.mix = mix
        self.train = train  # training set or test set

        self._omniglot = torchvision.datasets.Omniglot(root=root, download=True, transform=transform)

        self.character_classes = character_classes = np.array([
            character_class for _, character_class in self._omniglot._flat_character_images
        ])

        n_classes = max(character_classes)
        self.indices_for_class = {
            i: np.where(character_classes == i)[0].tolist()
            for i in range(n_classes)
        }

        np.random.seed(1)
        rp = np.random.permutation(n_classes)
        if train:
            self.used_classes = rp[:770]
        else:
            self.used_classes = rp[770:]

    def __getitem__(self, index):
        class_ix = self.used_classes[index]
        indices = self.indices_for_class[class_ix]
        if self.mix:
            indices = np.random.permutation(indices)

        indices = indices[:self.n_support+self.n_query]  # First support, then query
        support_query = torch.stack([self._omniglot[ix][0] for ix in indices])

        return support_query

    def __len__(self):
        return len(self.used_classes)

dataset = OmniglotFewShot(root=data_dir, n_support=1, n_query=3, train=True)
support_query = dataset[0]
print(support_query.shape)

n_way = 5
trainloader = DataLoader(dataset=dataset, batch_size=n_way, shuffle=True, pin_memory=True)

for support_query in trainloader:
    print(support_query.shape)
    # support_query is (n-way, n_support+n_query, 1, 28, 28)
    break


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.bl = nn.Sequential(nn.Conv2d(1,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,2),
                               nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,2),
                               nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,2),
                               nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,2),
                               nn.Flatten(),
                               nn.Linear(64,64))

    def forward(self,x):
        return self.bl(x)


def expanded_pairwise_distances(x, y=None):

    if y is not None:
         differences = x.unsqueeze(1) - y.unsqueeze(0)
    else:
        differences = x.unsqueeze(1) - x.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances

def episode_pn(net, support_query, n_support):
    support_set = support_query[:, :n_support]
    query_set = support_query[:, n_support:]
    #print(support_set.shape)
    emb_1 = net(support_set.reshape(-1,support_set.shape[2],support_set.shape[3],support_set.shape[4]))
    #print(emb_1)
    emb_1 = emb_1.reshape(-1,support_set.shape[1],emb_1.shape[1])
    #print(emb_1)
    prototypes = emb_1.mean(dim=1)
    emb_2 = net(query_set.reshape(-1,query_set.shape[2],query_set.shape[3],query_set.shape[4]))
    emb_2 = emb_2.reshape(-1,query_set.shape[1],emb_2.shape[1])
    accuracy = 0
    dist = []
    for n,t in enumerate(emb_2):
        dist.append(-expanded_pairwise_distances(t,prototypes))
    dist = torch.stack(dist)
    inpu = F.log_softmax(dist,dim=2)
    target = []
    for t in range(query_set.shape[0]):
        for n in range(query_set.shape[1]):
            target.append(t)
    target = torch.tensor(target).to(device)
    x = F.nll_loss(inpu.reshape(-1,inpu.shape[2]),target)
    for n,r in enumerate(inpu):
        for i in r:
            if i.argmax()==n:
                accuracy += 1
    accuracy /=inpu.shape[0]*inpu.shape[1]
    return x, accuracy, inpu

n_support = 1
n_query = 3
n_way = 5
trainset = OmniglotFewShot(root=data_dir, n_support=n_support, n_query=n_query, train=True, mix=True)
trainloader = DataLoader(dataset=trainset, batch_size=n_way, shuffle=True, pin_memory=True, num_workers=3)

testset = OmniglotFewShot(root=data_dir, n_support=n_support, n_query=n_query, train=False, mix=True)
testloader = DataLoader(dataset=testset, batch_size=n_way, shuffle=True, pin_memory=True, num_workers=3)

net = CNN()
net.to(device)

if not skip_training:
    lr = 0.001
    epochs = 3
    opti = optim.Adam(net.parameters(),lr)
    for t in range(epochs):
        net.train()
        for y in trainloader:
            y = y.to(device)
            opti.zero_grad()
            loss,accuracy,output = episode_pn(net,y,n_support)
            loss.backward()
            opti.step()
            print(accuracy)

if not skip_training:
    save_model(net, 'pn.pth')
else:
    net = CNN()
    load_model(net, 'pn.pth', device)

net.eval()
with torch.no_grad():
    support_query = iter(testloader).next()
    _, acc, outputs = episode_pn(net, support_query.to(device), n_support=1)
    print(outputs.argmax(dim=2))
