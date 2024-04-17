
#torch
import torch; print('\nPyTorch version in use:', torch.__version__, '\ncuda avail: ', torch.cuda.is_available())
import torch.nn as nn

class ConvBnRelu(nn.Module):
    # Convolutional layer followed by Batch Normalization and ReLU activation
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBnRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

class CNN(nn.Module):
    # CNN network
    def __init__(self, n_classes=10, depth_mult=1.):
        super(CNN, self).__init__()
        # first_conv_channels=int(32*depth_mult)
        self.ConvBnRelu1 = ConvBnRelu(1, 32,  stride=1) # conv3x3: ch_in=1, ch_out=32, in=28x28, out=28x28/2
        self.ConvBnRelu2 = ConvBnRelu(32, 64,  stride=1) # conv3x3: ch_in=1, ch_out=32, in=28x28, out=28x28/2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False) # MaxPool2: in=14x14, out=14x14/2
        self.ConvBnRelu3 = ConvBnRelu(64, 128, stride=2) # conv3x3: ch_in=32, ch_out=64, in=7x7, out=7x7
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        fc_in_size = 7*7*128
        self.fc = nn.Linear(fc_in_size, n_classes, bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.ConvBnRelu1(x)
        x = self.ConvBnRelu2(x)
        x = self.pool(x)
        x = self.ConvBnRelu3(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        output = self.softmax(x)
        return output

def print_summary(net, input_size):
    # From: https://github.com/TylerYep/torchinfo
    from copy import deepcopy
    # REQUIRED: !pip install torchinfo
    from torchinfo import summary
    
    print(summary(deepcopy(net), input_size=input_size)) #use deepcopy to avoid graph modifications by hese function calls
    
def main():
    # Create a CNN network and move it to the device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CNN().to(device)      
    
    # Print output tensor size and the CNN network topology defined:
    input_size = (1, 1, 28, 28)  # (batch_n, img_ch, img_width, img_height)
    input = torch.randn(input_size) # create a random input tensor
    input = input.to(device) # move input tensor to device

    # Forward pass
    output = net(input)
    
    # Print output tensor size and the CNN network topology defined:
    print('Output shape:', output.shape)
    print('Network Topology:\n', net)
    print_summary(net, input_size)
    
if __name__ == '__main__':
    main()
    
    
