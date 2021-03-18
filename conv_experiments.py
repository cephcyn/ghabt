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


class CifarNet4(nn.Module):
    def __init__(self):
        super(CifarNet4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=16384, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=10)
        self.accuracy = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

possible_labels = {label.rstrip() : i for i, label in enumerate(open('cifar/labels.txt', 'r'))}
examples = list()
labels = list()
#pertubations = [nn.Parameter(data=torch.zeroes(1, 3, 32, 32), requires_grad=True) for im in len(labels)]
tensorize = transforms.ToTensor()
imagize = transforms.ToPILImage()
i = 0
batch_examples = list()
batch_labels = list()
for example in open('cifar.test', 'r'):
    batch_examples.append(tensorize(Image.open(example.rstrip())).to(device))
    batch_examples[-1].requires_grad = False
    batch_labels.append(torch.LongTensor([possible_labels[example[example.find('_') + 1 : example.find(".")]]]).to(device))
    if i % 100 == 99:
        examples.append(torch.stack(batch_examples))
        labels.append(torch.squeeze(torch.stack(batch_labels)))
        batch_examples = list()
        batch_labels = list()
    i += 1
i = 0
old_examples = examples
old_labels = labels
examples = list()
labels = list()
for example in open('cifar.test', 'r'):
    batch_examples.append(tensorize(Image.open(example.rstrip().replace('train', 'train_pertubed_02_C4').replace('test', 'test_pertubed_02_C4'))).to(device))
    batch_examples[-1].requires_grad = False
    batch_labels.append(torch.LongTensor([possible_labels[example[example.find('_') + 1 : example.find(".")]]]).to(device))
    if i % 100 == 99:
        examples.append(torch.stack(batch_examples))
        labels.append(torch.squeeze(torch.stack(batch_labels)))
        batch_examples = list()
        batch_labels = list()
    i += 1

#pertubations = [torch.zeros(1, 3, 32, 32, requires_grad=True).to(device) for im in range(len(labels))]
EPSILON = 0
#model = resnet20()
conv_base_model = CifarNet4()
conv_base_model.load_state_dict(torch.load('orig_conv_model.pt'))
conv_model = CifarNet4()
conv_model.load_state_dict(torch.load('pert_conv_model.pt'))
res_base_model = models.resnet18()
res_base_model.fc = nn.Linear(res_base_model.fc.in_features, 10)
res_base_model.load_state_dict(torch.load('orig_model.pt'))
res_pert_model = models.resnet18()
res_pert_model.fc = nn.Linear(res_pert_model.fc.in_features, 10)
res_pert_model.load_state_dict(torch.load('advers_model.pt'))
#model = torch.load(open('/home/kaushman/pytorch_resnet_cifar10/pretrained_models/resnet20-12fca82f.th'))
#import pdb; pdb.set_trace()
conv_base_model.to(device)
conv_model.to(device)
res_base_model.to(device)
res_pert_model.to(device)
tot = 0
corr = 0
j = 0
for i in range(len(examples)):
    tot += 100
    output_1 = conv_base_model(examples[i])
    output_2 = conv_model(examples[i])
    output_3 = res_base_model(examples[i])
    output_4 = res_pert_model(examples[i])
    output_5 = conv_base_model(old_examples[i])
    output_6 = conv_model(old_examples[i])
    output_7 = res_base_model(old_examples[i])
    output_8 = res_pert_model(old_examples[i])
    i = 0
    for o1, o2, o3, o4, o5, o6, o7, o8, label in zip(output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, labels[i]):
        j+= 1
        if torch.argmax(o4).item() == label.item() and torch.argmax(o2).item() != label.item() and torch.argmax(o6).item() == label.item() and torch.argmax(o8).item() == label.item():
            import pdb; pdb.set_trace()
            corr += 1
print(corr / tot)

import pdb; pdb.set_trace()
for param in model.parameters():
  param.requires_grad=True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
for epoch in range(10):
    tot = 0
    corr = 0
    batch_examples = list()
    batch_labels = list()
    for i in range(len(examples)):
        if i % 100 == 0:
            print(i)
        model.zero_grad()
        tot += 100
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
    torch.save(model.state_dict(), 'pert_conv_model.pt')
print("hi")
import pdb; pdb.set_trace()
'''
criterion = nn.NLLLoss()
tot = 0
corr = 0
'''
for i in range(len(examples)):
    tot += 1
    output = model(examples[i])
    if torch.argmax(output).item() == labels[i].item():
        corr += 1
print(corr / tot)

#import pdb; pdb.set_trace()
print("pertube_images")
lr = 5
eps = 0.01
criterion = nn.NLLLoss()
pertubations = [torch.zeros(100, 3, 32, 32).to(device).requires_grad_() for im in range(len(labels))]
for epoch in range(10):
    tot = 0
    corr = 0
    for i in range(len(examples)):
        model.zero_grad()
        tot += 100
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
names = [img_name.rstrip().replace('train', 'train_pertubed_02_C4').replace('test', 'test_pertubed_02_C4') for img_name in open('cifar.test', 'r')]
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
