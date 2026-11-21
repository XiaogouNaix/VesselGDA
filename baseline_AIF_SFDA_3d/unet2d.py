import torch
from torch import nn
import random
DROPOUT_NONE = 0
DROPOUT_VANILLA = 1
DROPOUT_UD = 2

def get_dropout(dropout_type, dropout_rate=None):
    if dropout_type == DROPOUT_NONE:
        return nn.Identity()
    elif dropout_type == DROPOUT_VANILLA:
        return nn.Dropout(0.5 if dropout_rate is None else dropout_rate)
    elif dropout_type == DROPOUT_UD:
        return UDBasedDropout(0.1 if dropout_rate is None else dropout_rate)
    else:
        raise NotImplementedError('Unknown dropout type: {}'.format(dropout_type))

class UDBasedDropout(nn.Module):

    def __init__(self, dropout_rate=0.1):
        super(UDBasedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if self.training:
            return (1 + random.uniform(-self.dropout_rate, self.dropout_rate)) * x
        return x

    def dropout(self, x):
        return x * (1 - self.dropout_rate)

class MultiConv(nn.Module):

    def __init__(self, input_nc, output_nc, conv_block_num=2, kernel_size=3, use_bias=False, norm_func=nn.BatchNorm2d, activation_func=nn.ReLU, use_dropout=DROPOUT_NONE, do_residual=False, last_activation=True):
        super(MultiConv, self).__init__()
        assert conv_block_num >= 1
        main_conv_list = [nn.Conv2d(input_nc, output_nc, kernel_size, 1, kernel_size // 2, bias=use_bias), norm_func(output_nc), activation_func(inplace=True)] + [get_dropout(use_dropout)]
        for i in range(conv_block_num - 1):
            main_conv_list.append(nn.Conv2d(output_nc, output_nc, kernel_size, 1, kernel_size // 2, bias=use_bias))
            main_conv_list.append(norm_func(output_nc))
            if i == conv_block_num - 2 and last_activation:
                main_conv_list.append(activation_func(inplace=True))
            main_conv_list.append(get_dropout(use_dropout))
        self.main_conv = nn.Sequential(*main_conv_list)
        self.do_residual = do_residual

    def forward(self, x):
        output = self.main_conv(x) + (x if self.do_residual else 0)
        return output

class Unet(nn.Module):
    ALL_OUTPUT = 0
    PART_OUTPUT = 1
    ONE_OUTPUT = 2

    def __init__(self, input_nc, output_nc, depth=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=DROPOUT_NONE, last_layer='Sigmoid', activation_func=nn.ReLU, kernel_size=3, conv_num=2, use_bias=False, output_mode=ONE_OUTPUT, check_input_shape=None):
        super(Unet, self).__init__()
        if isinstance(output_mode, str):
            output_mode = getattr(Unet, output_mode.upper())
        self.output_mode = output_mode
        self.down_down = nn.MaxPool2d(2, 2)
        self.down_conv_list = nn.ModuleList([MultiConv(input_nc if i == 0 else ngf * 2 ** (i - 1), ngf * 2 ** i, conv_num, kernel_size, use_bias, norm_layer, activation_func, use_dropout) for i in range(depth)])
        self.up_conv_list = nn.ModuleList([MultiConv(ngf * 2 ** (i - 1), ngf * 2 ** (i - 2), conv_num, kernel_size, use_bias, norm_layer, activation_func, use_dropout) for i in range(depth, 1, -1)])
        self.up_up_list = nn.ModuleList([nn.ConvTranspose2d(ngf * 2 ** (i - 1), ngf * 2 ** (i - 2), kernel_size=2, stride=2, padding=0, bias=use_bias) for i in range(depth, 1, -1)])
        self.out = nn.Sequential(nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0, bias=use_bias), getattr(nn, last_layer)())
        self.check_input_shape = check_input_shape

    def forward(self, x):
        leak_dic = {}
        leak_dic['input'] = x
        if self.check_input_shape is not None:
            assert x.shape[2] == self.check_input_shape[0] and x.shape[3] == self.check_input_shape[1], f'Input shape {x.shape[2:]} does not match expected shape {self.check_input_shape}.'
        x_list = [self.down_conv_list[0](x)]
        leak_dic['skip_0'] = x_list[0]
        for i in range(1, len(self.down_conv_list)):
            x_list.append(self.down_conv_list[i](self.down_down(x_list[-1])))
            leak_dic[f'skip_{i}'] = x_list[-1]
        y_list = [x_list[-1]]
        for i in range(len(self.up_conv_list)):
            y = torch.cat([x_list[-i - 2], self.up_up_list[i](y_list[-1])], dim=1)
            y_list.append(self.up_conv_list[i](y))
        o = self.out(y_list[-1])
        for i in range(len(y_list)):
            leak_dic[f'map_{i}'] = y_list[i]
        leak_dic['last_feature_map'] = y_list[-1]
        if self.output_mode == Unet.ALL_OUTPUT:
            return (o, leak_dic)
        elif self.output_mode == Unet.PART_OUTPUT:
            return (o, y_list[-1], x_list[0])
        else:
            return o

class UnetDecoder(nn.Module):

    def __init__(self, input_nc, output_nc, depth=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=DROPOUT_NONE, last_layer='Sigmoid', activation_func=nn.ReLU, kernel_size=3, conv_num=2, use_bias=False):
        super(UnetDecoder, self).__init__()
        self.up_conv_list = nn.ModuleList([MultiConv(ngf * 2 ** (i - 1), ngf * 2 ** (i - 2), conv_num, kernel_size, use_bias, norm_layer, activation_func, use_dropout) for i in range(depth, 1, -1)])
        self.up_up_list = nn.ModuleList([nn.ConvTranspose2d(ngf * 2 ** (i - 1), ngf * 2 ** (i - 2), kernel_size=2, stride=2, padding=0, bias=use_bias) for i in range(depth, 1, -1)])
        self.out = nn.Sequential(nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0, bias=use_bias), getattr(nn, last_layer)())

    def forward(self, x, encoder_features, x_map_num=3):
        y_list = [x]
        if x_map_num == 4:
            pass
        else:
            for i in range(x_map_num, len(self.up_conv_list)):
                y = torch.cat([encoder_features[i - x_map_num], self.up_up_list[i](y_list[-1])], dim=1)
                y_list.append(self.up_conv_list[i](y))
        o = self.out(y_list[-1])
        return o
