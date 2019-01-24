from torch.nn import init
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import AttModel

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
    def __init__(self, input_size, hidden_size, linearprodis, drop_p, layer_norm=True, norm_input=True, norm_output=True, norm_hidden=True):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.norm_input = norm_input
        self.norm_output = norm_output
        self.norm_hidden = norm_hidden

        self.ih_linear = nn.Linear(input_size, 4 * hidden_size) if not linearprodis else AttModel.LinearProDis(input_size, 4 * hidden_size, drop_p=drop_p)
        self.hh_linear = nn.Linear(hidden_size, 4 * hidden_size) if not linearprodis else AttModel.LinearProDis(hidden_size, 4 * hidden_size)

        self.weight_ih = self.ih_linear.weight
        self.weight_hh = self.hh_linear.weight
        self.bias_ih = self.ih_linear.bias
        self.bias_hh = self.hh_linear.bias

        if self.layer_norm:
            if self.norm_input:
                self.ln_ih = nn.LayerNorm(4 * hidden_size)
                self.ln_hh = nn.LayerNorm(4 * hidden_size)
            else:
                self.ln_ih = self.ln_hh = lambda x: x

            if self.norm_output:
                self.ln_ho = nn.LayerNorm(hidden_size)
            else:
                self.ln_ho = lambda x: x

            if self.norm_hidden:
                self.ln_ht = nn.LayerNorm(hidden_size)
            else:
                self.ln_ht = lambda x: x

        else:
            self.ln_ih = self.ln_hh = self.ln_ho = self.ln_ht = lambda x: x

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.ln_ih(self.ih_linear(input)) + self.ln_hh(self.hh_linear(hx))

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = self.ln_ht(outgate * F.tanh(self.ln_ho(cy)))

        return hy, cy