import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float))

    def __len__(self):
        return len(self.data)

class FTDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        #print(x.size())
        net = x
        #print(net.size())
        # compute pad shape
        in_dim = net.shape[-1]
        #print(in_dim)
        out_dim = (in_dim + self.stride - 1) // self.stride
        #print(out_dim)
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        #print(p)
        pad_left = p // 2
        #print(pad_left)
        pad_right = p - pad_left
        #print(pad_right)
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        #print(net.size())
        #print(net)
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    params:
        kernel_size: kernel size
        stride: the stride of the window. Default value is kernel_size
    
    input: (n_sample, n_channel, n_length)
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        # compute pad shape
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MyConv1dPadSame_dual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyConv1dPadSame_dual, self).__init__()
        self.Conv1d = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            Swish(),
            nn.Conv1d(in_channels, out_channels, 16, 1, 7),
            #nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            Swish(),
            nn.Conv1d(out_channels, out_channels, 16, 1, 7),
            #nn.ReLU(inplace=True)
        )

        self.Conv1d_1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            Swish(),
            nn.Conv1d(in_channels, out_channels, 1, 1)
            # nn.ReLU(inplace=True),
            )

    def forward(self, x):

        out = x

        out = self.Conv1d(out)
        pad = nn.ReplicationPad1d(padding=(1, 1))
        out = pad(out)
        x = self.Conv1d_1(x)
        out += x

        return out


class MyMaxPool1dPadSame_U(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    params:
        kernel_size: kernel size
        stride: the stride of the window. Default value is kernel_size

    input: (n_sample, n_channel, n_length)
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame_U, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        if net.size()[2] % 2 == 0:
            net = self.max_pool(net)
        else:
            # compute pad shape
            p = max(0, self.kernel_size - 1)
            pad_left = p // 2
            pad_right = p - pad_left
            net = F.pad(net, (pad_left, pad_right), "constant", 0)

            net = self.max_pool(net)

        return net

class BasicBlock(nn.Module):
    """
    Basic Block: 
        conv1 -> convk -> conv1

    params:
        in_channels: number of input channels
        out_channels: number of output channels
        ratio: ratio of channels to out_channels
        kernel_size: kernel window length
        stride: kernel step size
        groups: number of groups in convk
        downsample: whether downsample length
        use_bn: whether use batch_norm
        use_do: whether use dropout

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, downsample, is_first_block=False, use_bn=True, use_do=True):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.middle_channels = int(self.out_channels * self.ratio)

        # the first conv, conv1
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=self.in_channels, 
            out_channels=self.middle_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # the second conv, convk
        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=self.middle_channels, 
            out_channels=self.middle_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the third conv, conv1
        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(
            in_channels=self.middle_channels, 
            out_channels=self.out_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # Squeeze-and-Excitation
        r = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels//r)
        self.se_fc2 = nn.Linear(self.out_channels//r, self.out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        out = x
        # the first conv, conv1
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv, convk
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # the third conv, conv1
        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out) # (n_sample, n_channel, n_length)

        # Squeeze-and-Excitation
        se = out.mean(-1) # (n_sample, n_channel)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.sigmoid(se) # (n_sample, n_channel)
        out = torch.einsum('abc,ab->abc', out, se)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out

class BasicStage(nn.Module):
    """
    Basic Stage:
        block_1 -> block_2 -> ... -> block_M
    """
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, i_stage, m_blocks, use_bn=True, use_do=True, verbose=False):
        super(BasicStage, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.i_stage = i_stage
        self.m_blocks = m_blocks
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        self.block_list = nn.ModuleList()
        for i_block in range(self.m_blocks):
            
            # first block
            if self.i_stage == 0 and i_block == 0:
                self.is_first_block = True
            else:
                self.is_first_block = False
            # downsample, stride, input
            if i_block == 0:
                self.downsample = True
                self.stride = stride
                self.tmp_in_channels = self.in_channels
            else:
                self.downsample = False
                self.stride = 1
                self.tmp_in_channels = self.out_channels
            
            # build block
            tmp_block = BasicBlock(
                in_channels=self.tmp_in_channels, 
                out_channels=self.out_channels, 
                ratio=self.ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=self.groups, 
                downsample=self.downsample, 
                is_first_block=self.is_first_block,
                use_bn=self.use_bn, 
                use_do=self.use_do)
            self.block_list.append(tmp_block)

    def forward(self, x):

        out = x

        for i_block in range(self.m_blocks):
            net = self.block_list[i_block]
            out = net(out)
            if self.verbose:
                print('stage: {}, block: {}, in_channels: {}, out_channels: {}, outshape: {}'.format(self.i_stage, i_block, net.in_channels, net.out_channels, list(out.shape)))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv1.in_channels, net.conv1.out_channels, net.conv1.kernel_size, net.conv1.stride, net.conv1.groups))
                print('stage: {}, block: {}, convk: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv2.in_channels, net.conv2.out_channels, net.conv2.kernel_size, net.conv2.stride, net.conv2.groups))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv3.in_channels, net.conv3.out_channels, net.conv3.kernel_size, net.conv3.stride, net.conv3.groups))

        return out

class Net1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    params:
        in_channels
        base_filters
        filter_list: list, filters for each stage
        m_blocks_list: list, number of blocks of each stage
        kernel_size
        stride
        groups_width
        n_stages
        n_classes
        use_bn
        use_do

    """

    def __init__(self, in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes, use_bn=True, use_do=True, verbose=False):
        super(Net1D, self).__init__()
        
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.ratio = ratio
        self.filter_list = filter_list
        self.m_blocks_list = m_blocks_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.n_stages = len(filter_list)
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        # first conv
        self.first_conv = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=self.base_filters, 
            kernel_size=self.kernel_size, 
            stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        # stages
        self.stage_list = nn.ModuleList()
        in_channels = self.base_filters
        for i_stage in range(self.n_stages):

            out_channels = self.filter_list[i_stage]
            m_blocks = self.m_blocks_list[i_stage]
            tmp_stage = BasicStage(
                in_channels=in_channels, 
                out_channels=out_channels, 
                ratio=self.ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=out_channels//self.groups_width, 
                i_stage=i_stage,
                m_blocks=m_blocks, 
                use_bn=self.use_bn, 
                use_do=self.use_do, 
                verbose=self.verbose)
            self.stage_list.append(tmp_stage)
            in_channels = out_channels

        # final prediction
        self.dense = nn.Linear(in_channels, n_classes)
        
    def forward(self, x, fc = True):
        out = x
        #print(out.size())
        # first conv
        out = self.first_conv(out)
        #print(out.size())
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)
        #print(out.size())
        
        # stages
        for i_stage in range(self.n_stages):
            net = self.stage_list[i_stage]
            out = net(out)
            #print(out.size())

        # final prediction
        out = out.mean(-1)
        if fc:
            out = self.dense(out)
        return out



class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = MyConv1dPadSame(in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = MyConv1dPadSame(in_planes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = MyConv1dPadSame(planes, planes, kernel_size=kernel_size, stride=1)
        self.bn2 = nn.BatchNorm1d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                MyConv1dPadSame(in_planes, planes, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = MyConv1dPadSame(in_planes, in_planes, kernel_size=kernel_size, stride=1)
        self.bn2 = nn.BatchNorm1d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = MyConv1dPadSame(in_planes, planes, kernel_size=kernel_size, stride=1,)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=kernel_size, scale_factor=stride)
            self.bn1 = nn.BatchNorm1d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=kernel_size, scale_factor=stride),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self,  kernel_size, channel, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = channel[0]
        self.z_dim = z_dim
        self.conv1 = MyConv1dPadSame(nc, channel[0], kernel_size=kernel_size, stride=2)
        self.bn1 = nn.BatchNorm1d(channel[0])
        self.layer1 = self._make_layer(BasicBlockEnc, channel[0], num_Blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, channel[1], num_Blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, channel[2], num_Blocks[2], kernel_size, stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, channel[3], num_Blocks[3], kernel_size, stride=2)
        self.linear = nn.Linear(channel[3], 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, kernel_size, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, kernel_size, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, fc):
        x = torch.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        # print('Encoder:', x.shape)
        x = self.layer2(x)
        # print('Encoder:', x.shape)
        x = self.layer3(x)
        # print('Encoder:', x.shape)
        x = self.layer4(x)
        # print('Encoder:', x.shape)
        # x = x.mean(-1)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = x.permute(0, 2, 1)
        # x = self.linear(x)
        # print('Encoder:', x.shape)
        return x

class ResNet18Dec(nn.Module):

    def __init__(self, kernel_size, channel, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = channel[0]
        self.linear = nn.Linear(z_dim, self.in_planes)
        self.layer4 = self._make_layer(BasicBlockDec, kernel_size, channel[0], num_Blocks[3], stride=1)
        self.layer3 = self._make_layer(BasicBlockDec, kernel_size, channel[1], num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, kernel_size, channel[2], num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, kernel_size, channel[3], num_Blocks[0], stride=2)
        self.conv1 = ResizeConv2d(channel[3], nc, kernel_size=kernel_size, scale_factor=2)
        self.nc = nc
        # self.out = nn.Linear(1024, self.length)

    def _make_layer(self, BasicBlockDec, kernel_size, planes, num_Blocks, stride, padding):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, kernel_size, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        print(x.shape)
        # print(x.shape)
        # print('encoder', x.shape)
        # x = x.view(z.size(0), 512, 1)
        # x = F.interpolate(x, scale_factor=64)
        x = x.permute(0, 2, 1)
        x = self.layer4(x)
        print(x.shape)
        x = self.layer3(x)

        print(x.shape)
        x = self.layer2(x)

        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = torch.sigmoid(self.conv1(x))
        x = self.out(x)
        print(x.shape)
        x = x.view(x.size(0), self.nc, -1)
        return x


class BasicBlock1dDec(nn.Module):
    def __init__(self, in_planes, stride, kernel_size, sub_padding=0):
        out_planes = in_planes // stride
        super(BasicBlock1dDec, self).__init__()
        self.conv2 = nn.Conv1d(in_planes, in_planes, kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2, stride=1)

        if stride == 1:
            self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                                   padding=(kernel_size - 1) // 2, stride=1)
            self.short_cut = nn.Sequential()
        else:
            self.conv1 = nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_size,
                                            padding=(kernel_size - 1) // 2, stride=stride,
                                            output_padding=(stride - 1 - sub_padding))
            self.short_cut = nn.Sequential(
                nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_size,
                                   padding=(kernel_size - 1) // 2, stride=stride,
                                   output_padding=(stride - 1 - sub_padding)),
                nn.BatchNorm1d(out_planes)
            )

        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(in_planes)
        self.bn1 = nn.BatchNorm1d(out_planes)

    def forward(self, x):
        residual = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        out = out + self.short_cut(residual)

        out = self.relu(out)

        return out


class ResNet1dDec(nn.Module):
    def __init__(self, block, layers, in_channel, out_channel, kernel_size=3):
        super(ResNet1dDec, self).__init__()
        self.layers = layers
        self.kernel_size = kernel_size

        strides = [2, 2, 2, 1]
        sub_paddings = [1, 0, 0, 0]
        channel = in_channel

        self.block_list = nn.ModuleList()
        for stride, layer, sub_padding in zip(strides, layers, sub_paddings):
            self.block_list.append(self._make_blocks(block, channel, layer, stride=stride, sub_padding=sub_padding))
            channel = channel // stride

        self.conv5 = nn.ConvTranspose1d(channel, out_channel, kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2, stride=2, output_padding=1)

    def _make_blocks(self, block, planes, n_layer, stride=1, sub_padding=0):
        strides = [1] * (n_layer - 1) + [stride]
        layers = []
        for stride in strides:
            layers.append(block(planes, stride, self.kernel_size, sub_padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = x.permute(0, 2, 1)
        out = x
        # print(out.shape)
        for block in self.block_list:
            out = block(out)
            # print('Decoder:', out.shape)
            # print(out.shape)
        # out = F.interpolate(out, scale_factor=2, mode="nearest")
        # print(out.shape)
        out = self.conv5(out)
        # print(out.shape)
        out = torch.sigmoid(out)
        return out

class VaeMiddle(nn.Module):
    def __init__(self, f_dim, z_dim):
        """

        :param f_dim: the dimension of feature
        :param z_dim: the dimension of z
        """
        super(VaeMiddle, self).__init__()
        self.f_dim = f_dim
        self.z_dim = z_dim

        self.mean_linear = nn.Linear(self.f_dim, self.z_dim)
        self.vari_linear = nn.Linear(self.f_dim, self.z_dim)
        self.scal_linear = nn.Sequential(nn.Linear(self.z_dim + 1, self.z_dim * 2),
                                         nn.ReLU(),
                                         nn.Linear(self.z_dim * 2, self.f_dim))

    def forward(self, x, vae_denoising):
        """

        :param x:   a tensor shaped of B, C, T
        :param vae_denoising:
        :return: new_feat(B, C, T), mean (B, T, C'), vari (B, T, C')
        """
        B, C, T = x.shape

        x = x.permute(0, 2, 1)  # B, T, C

        # mean and variance
        mean = self.mean_linear(x)  # B, T, C' (表示存在T个latent variable来控制生成)
        log_var = self.vari_linear(x)  # B, T, C'

        noise = torch.randn(B, T, self.z_dim, device=x.device)
        # B, T, C'
        z = mean + noise * torch.exp(0.5 * log_var)  # mean + N(0,1) * (var ** 0.5)

        if vae_denoising:
            mask_prob = 0.3
            mask_tag = torch.rand(B, T, 1, device=x.device, dtype=torch.float)
            mask_tag = mask_tag.gt(mask_prob).float()
            z = z * mask_tag
        else:
            mask_tag = torch.ones(B, T, 1, device=x.device, dtype=torch.float)

        new_feat = self.scal_linear(torch.cat([z, mask_tag], -1))  # B, T, C
        new_feat = new_feat.permute(0, 2, 1)  # B, C, T

        return new_feat, mean, log_var

class VAE(nn.Module):

    def __init__(self, channel, z_dim=20, kernel_size = 3):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim, channel=channel, kernel_size=kernel_size)
        self.decoder = ResNet1dDec(BasicBlock1dDec,[2, 2, 2, 2], channel[-1], 1)
        self.middle = VaeMiddle(channel[-1], z_dim)
        self.fc1 = nn.Linear(channel[-1], z_dim)
        self.fc2 = nn.Linear(channel[-1], z_dim)

    def forward(self, x, fc=True):
        x = self.encoder(x, fc=fc)
        # mu, logvar = self.fc1(x), self.fc2(x)
        z = x
        if not fc:
            return z.mean(-1)
        # z, mu, logvar = self.middle(x, False)
        # z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        # return infonce
        # return x, mu, logvar
        return x

    def loss_function(self, recons, input, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        kld_weight = 5  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))

        print(f'recons:{recons_loss}, kld:{kld_loss}')
        loss = recons_loss + kld_weight * kld_loss
        return loss

    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
