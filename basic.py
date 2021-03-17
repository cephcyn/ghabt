import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from load import load_possible_labels, load_cifar_train, load_cifar_test, load_perturbations, get_files, pertube_images


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
def trainModel(model, examples, labels, epochs):
    examples, labels = load_cifar_train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

    for epoch in range(epochs):
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
        print(str(epoch) + ': ' + str(corr / tot))


# Test
def testModel(model, examples, labels):
    # examples, labels = load_cifar_test()
    tot = 0
    corr = 0
    for i in range(len(examples)):
        tot += 1
        output = model(examples[i])
        if torch.argmax(output).item() == labels[i].item():
            corr += 1
    print(corr / tot)


if __name__ == "__main__":
    # get_file()
    # first model is trained using the training set for 50
    first_train_examples, first_train_labels = load_cifar_train()
    first_model = CifarNetLinear()
    print('Training first model...')
    trainModel(first_model, first_train_examples, first_train_labels, epochs = 50)

    # perturb the images using the model, then save perturbations
    print('Perturbing images...')
    perturbations = [torch.zeros(1, 3, 32, 32).requires_grad_() for im in range(len(first_train_examples))]
    accuracies = pertube_images(first_model, first_train_examples, first_train_labels, perturbations, 0.01, 10e-4, epochs = 10)
    im = 0
    for name in open('cifar.train', 'r'): # range(len(examples)):
        name = ('cifar/perturbations' + name[len('cifar/train') : len(name)]).rstrip('\n')
        transforms.ToPILImage()((perturbations[im] + first_train_examples[im]).squeeze()).save(name, 'PNG')
        im += 1
    print(accuracies)

    # second model is trained on both original training set and perturbed 
    second_examples, second_labels = load_perturbations()
    second_model = CifarNetLinear()
    print('Training second model...')
    trainModel(second_model, first_train_examples + second_examples, first_train_labels + second_labels, epochs = 25)

    # test models on the vanilla training set (without perturbations)
    first_test_examples, first_test_labels = load_cifar_test()
    print('Testing vanilla set...')
    print('First model:')
    testModel(first_model, first_test_examples, first_test_labels)
    print('Second model:')
    testModel(second_model, first_test_examples, first_test_labels)

    # test both models on the perturbation set (without vanilla)
    print('Testing perturbation set...')
    print('First model:')
    testModel(first_model, second_examples, second_labels)
    print('Second model:')
    testModel(second_model, second_examples, second_labels)
