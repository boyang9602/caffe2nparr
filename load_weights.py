import pickle
import torch
import numpy as np

def process(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    return data_dict['data'].reshape(data_dict['shape']).astype(np.float32)

def load_conv(filename, bias=False, layer=None):
    weights = process(f'{filename}_0.bin')
    if bias:
        bias = process(f'{filename}_1.bin')
    if layer != None:
        layer.weight.data = torch.from_numpy(weights)
        if type(bias) != bool:
            layer.bias.data = torch.from_numpy(bias)
    return weights, bias

def load_bn(filename, scale=True, layer=None):
    """
    PyTorch BN is equivelent to Caffe BN + Scale
    """
    results = []
    for i in range(3):
        results.append(process(f'{filename}_bn_{i}.bin'))
    mean = results[0] / results[2]
    var = results[1] / results[2]
    gamma, beta = None, None
    if scale:
        results = []
        for i in range(2):
            results.append(process(f'{filename}_scale_{i}.bin'))
        gamma = results[0]
        beta = results[1]
    if layer != None:
        layer.running_mean.data = torch.from_numpy(mean)
        layer.running_var.data = torch.from_numpy(var)
        if scale:
            layer.weight.data = torch.from_numpy(gamma)
            layer.bias.data = torch.from_numpy(beta)
    return mean, var, gamma, beta

def load_conv_bn(filename, bias=False, scale=True, layer=None):
    return load_conv(filename, bias, layer.conv), load_bn(filename, scale, layer.bn)

def load_inception_a(filename, layer=None):
    return {
        '1x1': load_conv_bn(f'{filename}_1x1', layer=layer.inception_a_1x1),
        '5x5_reduce': load_conv_bn(f'{filename}_5x5_reduce', layer=layer.inception_a_5x5_reduce),
        '5x5': load_conv_bn(f'{filename}_5x5', layer=layer.inception_a_5x5),
        '3x3_reduce': load_conv_bn(f'{filename}_3x3_reduce', layer=layer.inception_a_3x3_reduce),
        '3x3_1': load_conv_bn(f'{filename}_3x3_1', layer=layer.inception_a_3x3_1),
        '3x3_2': load_conv_bn(f'{filename}_3x3_2', layer=layer.inception_a_3x3_2),
        'pool_proj': load_conv_bn(f'{filename}_pool_proj', layer=layer.inception_a_pool_proj)
    }

def load_inception_b(filename, layer=None):
    return {
        '1x1_2': load_conv_bn(f'{filename}_1x1_2', layer=layer.inception_b_1x1_2),
        '1x7_reduce': load_conv_bn(f'{filename}_1x7_reduce', layer=layer.inception_b_1x7_reduce),
        '1x7': load_conv_bn(f'{filename}_1x7', layer=layer.inception_b_1x7),
        '7x1': load_conv_bn(f'{filename}_7x1', layer=layer.inception_b_7x1),
        '7x1_reduce': load_conv_bn(f'{filename}_7x1_reduce', layer=layer.inception_b_7x1_reduce),
        '7x1_2': load_conv_bn(f'{filename}_7x1_2', layer=layer.inception_b_7x1_2),
        '1x7_2': load_conv_bn(f'{filename}_1x7_2', layer=layer.inception_b_1x7_2),
        '7x1_3': load_conv_bn(f'{filename}_7x1_3', layer=layer.inception_b_7x1_3),
        '1x7_3': load_conv_bn(f'{filename}_1x7_3', layer=layer.inception_b_1x7_3),
        '1x1': load_conv_bn(f'{filename}_1x1', layer=layer.inception_b_1x1)
    }
