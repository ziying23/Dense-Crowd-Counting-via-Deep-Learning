import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

def create_conv2d_block(in_channels, kernel_size, n_filter, dilated_rate=1, batch_norm=True):
    # padding formula  https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
    """
    o = output
    p = padding
    k = kernel_size
    s = stride
    d = dilation
    """
#     o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    k = kernel_size
    d = dilated_rate
    padding_rate = int((k + (k-1)*(d-1))/2)
    conv2d =  nn.Conv2d(in_channels, n_filter, kernel_size, padding=padding_rate, dilation = dilated_rate)
    bn = nn.BatchNorm2d(n_filter)
    relu = nn.ReLU(inplace=True)
    if batch_norm:
        return [conv2d, bn, relu]
    else:
        return [conv2d, relu]

class ScalePyramidModule(nn.Module):
    def __init__(self):
        super(ScalePyramidModule, self).__init__()
        self.a = nn.Sequential(*create_conv2d_block(512, 3, 512, 2))
        self.b = nn.Sequential(*create_conv2d_block(512, 3, 512, 4))
        self.c = nn.Sequential(*create_conv2d_block(512, 3, 512, 8))
        self.d = nn.Sequential(*create_conv2d_block(512, 3, 512, 12))
    def forward(self,x):
        xa = self.a(x)
        xb = self.b(x)
        xc = self.c(x)
        xd = self.d(x)
        return torch.cat((xa, xb, xc, xd), 1)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

def make_layers_by_cfg(cfg, in_channels = 3,batch_norm=False, dilation = True):
    """
    cfg: list of tuple (number of layer, kernel, n_filter, dilated) or 'M'
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # number of layer, kernel, n_filter, dilated
            for t in range(v[0]):
                layers += create_conv2d_block(in_channels, v[1], v[2], v[3], batch_norm = batch_norm)
                in_channels = v[2]
    return nn.Sequential(*layers)

class apdn(nn.Module):
    def __init__(self, load_weights=False):
        super(apdn, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]      
        self.frontend = make_layers(self.frontend_feat)
        self.dropout = nn.Dropout(0.25)
        
        self.planes = self.frontend_feat[-1]
        self.ca = ChannelAttention(self.planes)
        
        self.spm = ScalePyramidModule()
        
        self.conv1 = nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size=3, padding = 1, dilation = 1)
        self.conv2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, padding = 1, dilation = 1)
        self.conv3 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size=3, padding = 1, dilation = 1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, padding = 1, dilation = 1)
        self.output_layer = nn.Conv2d(256, 1, kernel_size=1, padding = 0) 


        
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(list(self.frontend.state_dict().items()))):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self,img_tensor):
        x = self.frontend(img_tensor)
        residual = x
        x = self.ca(x) * x
        x += residual
        #print("x1", x1.shape)
        
        x2 = self.dropout(x)
        x3 = self.spm(x2) 
        x4 = self.dropout(x3)
        #print("x2 ", x2.shape)
        
        x5 = F.relu(self.conv1(x4))
        #print("x3 ", x3.shape)
        x6 = F.relu(self.conv2(x5))
        #print("x4 ", x4.shape)
        x7 = F.relu(self.conv3(x6))
        x8 = F.relu(self.conv4(x7))
        x = self.output_layer(x8)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# net = apdn()
# print(net)
