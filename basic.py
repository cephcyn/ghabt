import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from load import load_possible_labels, load_cifar_train, load_cifar_test, get_files


class CifarNetLinear(nn.Module):
    def __init__(self):
        super(CifarNetLinear, self).__init__()
        self.fc1 = nn.Linear(3072, 10)

    def forward(self, x):
        # if not self.training:
        x = torch.flatten(x, 1)
        return self.fc1(x)

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction, label, reduction=reduction)
        return loss_val


# Train
def trainModel(model):
    possible_labels = load_possible_labels()
    examples, labels = load_cifar_train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

    for epoch in range(1):
        tot = 0
        corr = 0
        for i in range(len(examples)):
            tot += 1
            output = model(examples[i])
            if torch.argmax(output).item() == labels[i].item():
                corr += 1
            loss = criterion(output, labels[i])
            loss.backward()
            optimizer.step()
        print(corr / tot)


# Test
def testModel(model):
    examples, labels = load_cifar_test()
    tot = 0
    corr = 0
    for i in range(len(examples)):
        tot += 1
        output = model(examples[i])
        if torch.argmax(output).item() == labels[i].item():
            corr += 1
    print(corr / tot)

if __name__ == "__main__":
    get_files()
    model = CifarNetLinear()
    trainModel(model)
    testModel(model)
