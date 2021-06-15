import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from QuantConv2d import UnfoldConv2d
from SimModel import SimModel


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(SimModel):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = UnfoldConv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = UnfoldConv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = UnfoldConv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = UnfoldConv2d(256, 256, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.do1 = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.do2 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.update_unfold_list()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.do1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.do2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


def alexnet_imagenet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)

        # Transforming the conventional state_dict mapping
        state_dict['conv1.weight'] = state_dict.pop('features.0.weight')
        state_dict['conv1.bias'] = state_dict.pop('features.0.bias')
        state_dict['conv2.weight'] = state_dict.pop('features.3.weight')
        state_dict['conv2.bias'] = state_dict.pop('features.3.bias')
        state_dict['conv3.weight'] = state_dict.pop('features.6.weight')
        state_dict['conv3.bias'] = state_dict.pop('features.6.bias')
        state_dict['conv4.weight'] = state_dict.pop('features.8.weight')
        state_dict['conv4.bias'] = state_dict.pop('features.8.bias')
        state_dict['conv5.weight'] = state_dict.pop('features.10.weight')
        state_dict['conv5.bias'] = state_dict.pop('features.10.bias')
        state_dict['fc1.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['fc1.bias'] = state_dict.pop('classifier.1.bias')
        state_dict['fc2.weight'] = state_dict.pop('classifier.4.weight')
        state_dict['fc2.bias'] = state_dict.pop('classifier.4.bias')
        state_dict['fc3.weight'] = state_dict.pop('classifier.6.weight')
        state_dict['fc3.bias'] = state_dict.pop('classifier.6.bias')

        model.load_state_dict(state_dict, strict=False)
    return model
