import torch
import torch.nn as nn
from QuantConv2d import UnfoldConv2d
import numpy as np

class SimModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unfold_list = []   # All UnfoldConv2d layers

    def forward(self, x):
        raise NotImplementedError

    def update_unfold_list(self):
        self.apply(self._apply_unfold_list)

    def _apply_unfold_list(self, m):
        if type(m) == UnfoldConv2d:
            self.unfold_list.append(m)

    def set_unfold(self, v):
        for l in self.unfold_list:
            l._unfold = v

    def set_quantize(self, v):
        for l in self.unfold_list:
            l._quantize = v

    def set_custom_matmul(self, v):
        for l in self.unfold_list:
            l._custom_matmul = v

    def set_quantization_bits(self, x_bits, w_bits):
        for l in self.unfold_list:
            l._x_bits = x_bits
            l._w_bits = w_bits

    def set_min_max_update(self, v):
        for l in self.unfold_list:
            l._disable_min_max_update = not v

    #PSUM BITS LIMITATION
    def set_bitsAllowed(self,layer_num, bits_allowed, bits_to_all):
        counter = 0


        if (layer_num == None):
            for l in self.unfold_list:
                l._bits_allowed = bits_allowed
                l._bits_to_all = bits_to_all
        else:
            for l in self.unfold_list:
                if(counter == layer_num):
                    l._bits_allowed = bits_allowed
                    counter = counter + 1
                else:
                    l._bits_allowed = bits_to_all
                    counter = counter + 1


        # #for custom layer sizes
        # for l in self.unfold_list:
        #     if (counter == 0):
        #         l._bits_allowed = 21
        #     elif (counter == 1):
        #         l._bits_allowed = 23
        #     elif (counter == 2):
        #         l._bits_allowed = 21
        #     elif (counter == 3):
        #         l._bits_allowed = 21
        #     counter = counter + 1


    #LAYER NUMBER
    def set_layer_number(self):
        counter = 0
        for l in self.unfold_list:
            l._layer_number = counter
            counter = counter + 1

    #TEST COUNTER
    def set_test_counter(self):
        for l in self.unfold_list:
            l._test_counter = 0

    # #ONLY FOR ALEXNET CHANNELS
    # def set_array(self):
    #     counter = 0
    #     for l in self.unfold_list:
    #         if (counter == 0):
    #             l._arr = [0] * 192
    #         elif (counter == 1):
    #             l._arr = [0] * 384
    #         elif (counter == 2):
    #             l._arr = [0] * 256
    #         elif (counter == 3):
    #             l._arr = [0] * 256
    #         counter = counter + 1

    # #ONLY FOR RESNET CHANNELS
    # def set_array(self):
    #     counter = 0
    #     for l in self.unfold_list:
    #         if (0 <= counter <= 3):
    #             l._arr = [0] * 64
    #         elif (4 <= counter <= 8):
    #             l._arr = [0] * 128
    #         elif (9 <= counter <= 13):
    #             l._arr = [0] * 256
    #         elif (14 <= counter <= 18):
    #             l._arr = [0] * 512
    #         counter = counter + 1