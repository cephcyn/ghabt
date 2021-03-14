import torch
from subprocess import Popen
from pathlib import Path
from PIL import Image
from torchvision import transforms

def get_files(cifar_location='.'):
    cifar_tar = Path(cifar_location + '/cifar.tgz')
    if not cifar_tar.is_file():
        Popen(['./load_cifar.sh'])

def load_possible_labels():
    possible_labels = {label.rstrip() : i for i, label in enumerate(open('cifar/labels.txt', 'r'))}
    return possible_labels

def load_cifar_train():
    possible_labels = load_possible_labels()
    examples = []
    labels = []
    tensorize = transforms.ToTensor()
    for example in open('cifar.train', 'r'):
        examples.append(torch.unsqueeze(tensorize(Image.open(example.rstrip())), 0))
        labels.append(torch.LongTensor([possible_labels[example[example.find('_') + 1 : example.find(".")]]]))
    return examples, labels

def load_cifar_test():
    possible_labels = load_possible_labels()
    examples = []
    labels = []
    tensorize = transforms.ToTensor()
    for example in open('cifar.test', 'r'):
        examples.append(torch.unsqueeze(tensorize(Image.open(example.rstrip())), 0))
        labels.append(torch.LongTensor([possible_labels[example[example.find('_') + 1 : example.find(".")]]]))
    return examples, labels
