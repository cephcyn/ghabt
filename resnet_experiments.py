import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import PIL
from torchvision import transforms
import torch.optim as optim
import re
from matplotlib import pyplot as plt
from torch.autograd import Function
from testing import pertube_images
import torchvision.models as models
import torch.nn.init as init
from torch.autograd import Variable
device = torch.device('cuda')
'''
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

def grad_reverse(x):
    return GradReverse.apply(x)
'''
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.fc1 = nn.Linear(3072, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc1(x)

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction, label, reduction=reduction)
        return loss_val

possible_labels = {label.rstrip() : i for i, label in enumerate(open('cifar/labels.txt', 'r'))}
examples = list()
labels = list()
#pertubations = [nn.Parameter(data=torch.zeroes(1, 3, 32, 32), requires_grad=True) for im in len(labels)]
tensorize = transforms.ToTensor()
imagize = transforms.ToPILImage()
i = 0
batch_examples = list()
batch_labels = list()
'''
for example in open('cifar.test', 'r'):
    batch_examples.append(tensorize(Image.open(example.rstrip())).to(device))
    batch_examples[-1].requires_grad = False
    batch_labels.append(torch.LongTensor([possible_labels[example[example.find('_') + 1 : example.find(".")]]]).to(device))
    if i % 8 == 7:
        examples.append(torch.stack(batch_examples))
        labels.append(torch.squeeze(torch.stack(batch_labels)))
        batch_examples = list()
        batch_labels = list()
    i += 1
i = 0
'''
for example in open('cifar.test', 'r'):
    batch_examples.append(tensorize(Image.open(example.rstrip().replace('train', 'train_pertubed_02').replace('test', 'test_pertubed_02'))).to(device))
    batch_examples[-1].requires_grad = False
    batch_labels.append(torch.LongTensor([possible_labels[example[example.find('_') + 1 : example.find(".")]]]).to(device))
    if i % 8 == 7:
        examples.append(torch.stack(batch_examples))
        labels.append(torch.squeeze(torch.stack(batch_labels)))
        batch_examples = list()
        batch_labels = list()
    i += 1
#pertubations = [torch.zeros(1, 3, 32, 32, requires_grad=True).to(device) for im in range(len(labels))]
EPSILON = 0
#model = resnet20()
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('advers_model.pt'))
#model = torch.load(open('/home/kaushman/pytorch_resnet_cifar10/pretrained_models/resnet20-12fca82f.th'))
#import pdb; pdb.set_trace()
model.to(device)
tot = 0
corr = 0
for i in range(len(examples)):
    tot += 8
    outputs = model(examples[i])
    for output, label in zip(outputs, labels[i]):
        if torch.argmax(output).item() == label.item():
            corr += 1
print(corr / tot)
import pdb; pdb.set_trace()
for param in model.parameters():
  param.requires_grad=True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
for epoch in range(1):
    tot = 0
    corr = 0
    batch_examples = list()
    batch_labels = list()
    for i in range(len(examples)):
        if i % 10000 == 0:
            print(i)
        model.zero_grad()
        tot += 8
        outputs = model(examples[i])
        for output, label in zip(outputs, labels[i]):
            if torch.argmax(output).item() == label.item():
                corr += 1
        loss = criterion(outputs, labels[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(corr)
    print(tot)
    print("ACC "+ str(corr / tot))
    torch.save(model.state_dict(), 'advers_model.pt')
print("hi")
import pdb; pdb.set_trace()
criterion = nn.NLLLoss()
tot = 0
corr = 0
for i in range(len(examples)):
    tot += 1
    output = model(examples[i])
    if torch.argmax(output).item() == labels[i].item():
        corr += 1
print(corr / tot)
print("pertube_images")
lr = 10**-2
eps = 0.01
criterion = nn.NLLLoss()
pertubations = [torch.zeros(8, 3, 32, 32).to(device).requires_grad_() for im in range(len(labels))]
for epoch in range(10):
    tot = 0
    corr = 0
    for i in range(len(examples)):
        model.zero_grad()
        tot += 8
        outputs = model(examples[i] + pertubations[i])
        for output, label in zip(outputs, labels[i]):
            if torch.argmax(output).item() == label.item():
                corr += 1
        loss = criterion(outputs, labels[i])
        loss.backward()
        pertubations[i] = torch.clamp(pertubations[i].grad * lr + pertubations[i], -eps, eps).detach().clone()
        pertubations[i].requires_grad = True
    print(corr)
    print(tot)
    print(corr / tot)
names = [img_name.rstrip().replace('train', 'train_pertubed_02').replace('test', 'test_pertubed_02') for img_name in open('cifar.test', 'r')]
idx = 0
for example, pertubation in zip(examples, pertubations):
    for ex, pert in zip(example, pertubation):
        imagize(ex.cpu() + pert.cpu()).save(names[idx], format='PNG')
        idx += 1
import pdb; pdb.set_trace()
#print(pertube_images(model, examples, labels, pertubations, eps, lr, 10))

for i in range(10):
    plt.imshow(imagize(torch.clamp(examples[i].cpu() + pertubations[i].cpu(), 0, 1).squeeze()))
    plt.show()
tot = 0
corr = 0
plt.imshow(imagize(examples[0].squeeze()))
plt.show()
for i in range(10):
    plt.imshow(imagize(torch.clamp(examples[i].cpu() + pertubations[i].cpu(), 0, 1).squeeze()))
    plt.show()
print("fINAL")
for i in range(len(examples)):
    tot += 1
    output = model(examples[i])
    if torch.argmax(output).item() == labels[i].item():
        corr += 1
print(corr / tot)
