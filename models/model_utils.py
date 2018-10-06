from torch.nn import init
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

#-----model initialization-----#
def xavier_uniform(nonlinearity='linear', *modules):
    '''
    xavier uniform initialization, and fill the bias to zero
    :param nonlinearity: string,the non-linear function (nn.functional name), one of ['linear', 'conv1d', 'conv2d',
    'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh', 'relu', 'leaky_relu']
    :param modules: modules which need to be initialized
    :return: no return
    '''
    gain = init.calculate_gain(nonlinearity)
    for module in modules:
        init.xavier_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.fill_(0)


def xavier_normal(nonlinearity='linear', *modules):
    '''
    xavier normal initialization, and fill the bias to zero
    :param nonlinearity: string,the non-linear function (nn.functional name), one of ['linear', 'conv1d', 'conv2d',
    'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh', 'relu', 'leaky_relu']
    :param modules: modules which need to be initialized
    :return: no return
    '''
    gain = init.calculate_gain(nonlinearity)
    for module in modules:
        init.xavier_normal_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.fill_(0)


def kaiming_uniform(nonlinearity, a, *modules):
    '''
    kaiming_uniform initialization, and fill the bias to zero
    :param nonlinearity: one of 'relu' or 'leaky_relu'
    :param a: the negative slope of the rectifier used after this layer (0 for ReLU)
    :param modules: modules which need to be initialized
    :return: no return
    '''
    for module in modules:
        init.kaiming_uniform_(module.weight, a=a, mode='fan_in', nonlinearity=nonlinearity)
        if module.bias is not None:
            module.bias.data.fill_(0)


def kaiming_normal(nonlinearity, a, *modules):
    '''
    kaiming_normal_ initialization, and fill the bias to zero
    :param nonlinearity: one of 'relu' or 'leaky_relu'
    :param a: the negative slope of the rectifier used after this layer (0 for ReLU)
    :param modules: modules which need to be initialized
    :return: no return
    '''
    for module in modules:
        init.kaiming_normal_(module.weight, a=a, mode='fan_in', nonlinearity=nonlinearity)
        if module.bias is not None:
            module.bias.data.fill_(0)


def lstm_init(lstm_Module):
    '''
    orthogonalize the weights in lstm, and zeros the bias in lstm, and the bias of forget gate is set to 1.
    :param lstm_Module: the lstm model to be initialized
    :return: no return
    '''
    hidden_size = lstm_Module.hidden_size
    for name, param in lstm_Module.named_parameters():
        if 'bias_' in name:
            init.constant_(param, 0.0)
            param.data[hidden_size:2 * hidden_size] = 0.5
        elif 'weight_' in name:
            init.orthogonal_(param)



class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, layer_norm=True):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        if self.layer_norm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)
            self.ln_hh = nn.LayerNorm(4 * hidden_size)
            self.ln_ho = nn.LayerNorm(hidden_size)
        else:
            self.ln_ih = self.ln_hh = self.ln_ho = lambda x: x

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.ln_ih(F.linear(input, self.weight_ih)) + self.ln_hh(F.linear(hx, self.weight_hh)) + self.bias_ih + self.bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(self.ln_ho(cy))

        return hy, cy