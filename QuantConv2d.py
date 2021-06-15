import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cu_gemm

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class UnfoldConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(UnfoldConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                                           bias=bias, padding_mode=padding_mode)

        # Registering buffers to be saved when calling torch.save
        self.register_buffer('tracked_n', torch.zeros(1))
        self.register_buffer('max_mean', torch.zeros(1))
        self.register_buffer('min_mean', torch.zeros(1))

        self._unfold = False
        self._quantize = False
        self._custom_matmul = False
        self._disable_min_max_update = False
        self._return_inputs = False
        self._x_bits = 8
        self._w_bits = 8
        self._bits_allowed = 0
        self._bits_to_all = 0
        self._layer_number = 0
        self._test_counter = 0
        self.out_clone = None

    def forward(self, x):
        # Prepare activations, weights, and bias
        if self._quantize:

            # Gather statistics during training
            if self.training and not self._disable_min_max_update:
                tracked_n_old = self.tracked_n.clone()
                self.tracked_n += x.size(0)

                max_sum = x.detach().max(dim=3).values.max(dim=2).values.max(dim=1).values.sum()
                min_sum = x.detach().min(dim=3).values.min(dim=2).values.min(dim=1).values.sum()

                self.max_mean = ((self.max_mean * tracked_n_old) + max_sum) / self.tracked_n
                self.min_mean = ((self.min_mean * tracked_n_old) + min_sum) / self.tracked_n

            # These statistics are mandatory for quantization
            assert (self.max_mean != 0 or self.min_mean != 0)

            # Activations quantization
            # Currently only supports unsigned uniform quantization
            if torch.min(x) == 0:
                x_q, x_q_delta = self._uniform_quantization(x, self.max_mean, self._x_bits)
            else:
                raise NotImplementedError

            # Weights quantization
            weight_q, weight_q_delta = \
                self._uniform_symmetric_quantization_per_channel(self.weight,
                                                                 self.weight.data.min(dim=3)[0].min(dim=2)[0].min(dim=1)[0],
                                                                 self.weight.data.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0],
                                                                 self._w_bits)

            # Check ranges
            assert (x_q.max() <= (2 ** self._x_bits - 1) and x_q.min() >= 0)
            assert (weight_q.max() <= (2 ** self._w_bits / 2 - 1) and weight_q.min() >= -2 ** self._w_bits / 2)

            # Bias quantization
            if self.bias is None:
                bias_fp = None
            else:
                bias_q, bias_q_delta = self._uniform_symmetric_quantization(self.bias,
                                                                            torch.min(self.bias.data),
                                                                            torch.max(self.bias.data), self._w_bits)
                bias_fp = bias_q * bias_q_delta

        else:
            # The single scalar movement to CUDA may be bad for performance
            x_q, x_q_delta = x, torch.Tensor([1]).cuda()
            weight_q, weight_q_delta = self.weight, torch.Tensor([1]).cuda()
            bias_fp = self.bias

        if not self._unfold:

            out = nn.functional.conv2d(x_q * x_q_delta,
                                       weight_q * weight_q_delta[:, None, None, None].expand_as(weight_q),
                                       bias=bias_fp,
                                       stride=self.stride[0],
                                       padding=self.padding[0], groups=self.groups)
        else:
            # At the moment, unfold and quantization must go together
            # assert(self._quantize)

            x_unf = nn.functional.unfold(x_q,
                                         kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                         padding=(self.padding[0], self.padding[1]),
                                         stride=(self.stride[0], self.stride[1])).transpose(1, 2)
            w_unf = weight_q.view(self.weight.size(0), -1).t()

            ofmap_height = \
                int((x.size(2) + 2 * self.padding[0] - self.kernel_size[0] + self.stride[0]) / self.stride[0])
            ofmap_width = \
                int((x.size(3) + 2 * self.padding[1] - self.kernel_size[1] + self.stride[1]) / self.stride[1])

            if not self._custom_matmul:
                out_unf = x_unf.matmul(w_unf).transpose(1, 2)
                out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))

                bias_fp = 0 if bias_fp is None else bias_fp[None, :, None, None].expand_as(out)

                out = out * x_q_delta * weight_q_delta[None, :, None, None].expand_as(out) + bias_fp

            else:
                _x_unf = x_unf.reshape(x_unf.size(0) * x_unf.size(1), x_unf.size(2))
                _w_unf = w_unf.t()

                #Custom CUDA kernel
                data_tensor = cu_gemm.forward(_x_unf.contiguous(), _w_unf.contiguous(),self._bits_allowed, self._bits_to_all)
                data_tensor = data_tensor[0]
                #print(data_tensor)
                #data_tensor[0][0] = data_tensor[0][0]*10000



                # #find MAX psum bits for EVERY CHANNEL
                # for i in range(len(data_tensor[1])):
                #     max_tensor = data_tensor[0][i]
                #     min_tensor = data_tensor[0][i]
                #     for k in range (len(data_tensor[0])):
                #         if data_tensor[k][i] > max_tensor:
                #             max_tensor = data_tensor[k][i]
                #         elif data_tensor[k][i] < min_tensor:
                #             min_tensor = data_tensor[k][i]
                #     if abs(min_tensor) > abs(max_tensor):
                #         global_max = abs(min_tensor)
                #     else:
                #         global_max = abs(max_tensor)
                #     if global_max == 0:
                #         max_psum_bits = 1
                #     else:
                #         max_psum_bits = math.ceil(math.log(global_max, 2))
                #         max_psum_bits += 1
                #     # print("max psum bits in channel " , i , ": ")
                #     # print(max_psum_bits)
                #     if self._arr[i] < max_psum_bits:
                #         self._arr[i] = max_psum_bits
                # self._test_counter = self._test_counter + 1 #for update run test number
                # if self._test_counter == 391:
                #     name = f"resnet18_imagenet_8_x_w_max_bits_psum_from_layer_{self._layer_number}_channel_1_to_end.txt"
                #     with open(name, 'a') as f:
                #         for i in range(len(self._arr)):
                #             print(self._arr[i], file=f)
                #         f.close()

                # # find AVG psum bits for EVERY CHANNEL
                # for i in range(len(data_tensor[1])):
                #     sum = 0
                #     for k in range (len(data_tensor[0])):
                #         sum += data_tensor[k][i]
                #     sum /= len(data_tensor[0])
                #     if sum < 0:
                #         sum = -sum
                #     if sum == 0:
                #         avg_psum_bits = 1
                #     else:
                #         avg_psum_bits = math.ceil(math.log(sum, 2))
                #         avg_psum_bits += 1
                #     # print("avg psum bits in channel " , i , ": ")
                #     # print(avg_psum_bits)
                #     self._arr[i] += avg_psum_bits
                # self._test_counter = self._test_counter + 1 #for update run test number
                # if self._test_counter == 391:
                #     name = f"resnet18_imagenet_8_x_w_avg_bits_psum_from_layer_{self._layer_number}_channel_1_to_end.txt"
                #     with open(name, 'a') as f:
                #         for i in range(len(self._arr)):
                #             print(math.ceil(self._arr[i]/391), file=f)
                #         f.close()

                #
                # #find MAX psum bits for EVERY LAYER
                # max_tensor = torch.max(data_tensor)
                # min_tensor = torch.min(data_tensor)
                # if (max_tensor >= min_tensor):
                #     max_global = max_tensor
                # else:
                #     max_global = -min_tensor
                # if max_global == 0:
                #     max_psum_bits = 1
                # else:
                #     max_psum_bits = math.ceil(math.log(max_global,2))
                #     max_psum_bits += 1
                # #print("max psum bits: ")
                # #print(max_psum_bits)
                # name = f"resnet18_imagenet_8_x_w_max_bits_psum_from_layer_{self._layer_number}.txt"
                # with open(name, 'a') as f:
                #      print(max_psum_bits, file=f)
                #      f.close()

                # #find AVG psum for EVERY LAYER
                # avg_tensor = torch.mean(data_tensor).item()
                # if avg_tensor < 0:
                #     avg_tensor = -avg_tensor
                # if avg_tensor == 0:
                #     avg_psum_bits = 1
                # else:
                #     avg_psum_bits = math.ceil(math.log(avg_tensor,2))
                #     avg_psum_bits += 1
                # #print("avg psum bits: ")
                # #print(avg_psum_bits)
                # name = f"restnet18_imagenet_8_x_w_avg_bits_psum_from_layer_{self._layer_number}.txt"
                # with open(name, 'a') as f:
                #      print(avg_psum_bits, file=f)
                #      f.close()

                data_tensor = data_tensor.reshape(x_unf.size(0),
                                                  int(data_tensor.size(0) / x_unf.size(0)),
                                                  data_tensor.size(1))

                out_unf = data_tensor.transpose(1, 2)
                out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))
                out = out * x_q_delta * weight_q_delta[None, :, None, None].expand_as(out) + (0 if bias_fp is None else bias_q[None, :, None, None] * bias_q_delta)

        return out

    @staticmethod
    def _uniform_quantization(x, x_max, bits):
        N = 2 ** bits
        delta = x_max / (N - 1)
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, 0, N - 1)
        return x_q, delta

    @staticmethod
    def _uniform_symmetric_quantization_per_channel(x, x_min, x_max, bits):
        N = 2 ** bits
        delta = torch.where(x_min.abs() > x_max.abs(), x_min.abs(), x_max.abs()) * 2 / (N - 1)
        x_int = RoundSTE.apply(x / delta[:, None, None, None].expand_as(x))
        x_q = torch.clamp(x_int, -N / 2, N / 2 - 1)
        return x_q, delta

    @staticmethod
    def _uniform_symmetric_quantization(x, x_min, x_max, bits):
        N = 2 ** bits
        delta = max(abs(x_min), abs(x_max)) * 2 / (N - 1)
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, -N / 2, N / 2 - 1)
        return x_q, delta
