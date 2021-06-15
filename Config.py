import os
from Datasets import CIFAR10, CIFAR100, ImageNet

from models.alexnet_imagenet import alexnet_imagenet
from models.resnet_imagenet import resnet18 as resnet18_imagenet

basedir, _ = os.path.split(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'data')

#print(basedir);

RESULTS_DIR = os.path.join(basedir, 'results')

DEBUG = False
USER_CMD = None
BATCH_SIZE = 128
VERBOSITY = 0

MODELS = {'alexnet_imagenet': alexnet_imagenet,
          'resnet18_imagenet': resnet18_imagenet}

DATASETS = {'cifar10':
                {'ptr': CIFAR10,  'dir': os.path.join(basedir, 'datasets')},
            'cifar100':
                {'ptr': CIFAR100, 'dir': os.path.join(basedir, 'datasets')},
            'imagenet':
                {'ptr': ImageNet, 'dir': '/mnt/ilsvrc2012/'}}
