import torch 
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, kernel_size = (3,3), padding=(1,1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.downsample_bool = in_ch!=out_ch
        if self.downsample_bool:
            stride = (2,2)
        else:
            stride = (1,1)
        self.conv1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,stride=stride,
                               padding=padding,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=(1,1),
                               padding=padding,bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if self.downsample_bool:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=(1,1),stride=stride,bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self,x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.downsample_bool:
            x = self.downsample(x)
        return x + output
    
class ResNet(nn.Module):
    def __init__(self, channels=[3,64,128,256,512], nc=1000,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv2d(in_channels=channels[0],out_channels=channels[1],kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        self.layer1 = nn.Sequential(
            BasicBlock(in_ch=channels[1],out_ch=channels[1]),
            BasicBlock(in_ch=channels[1],out_ch=channels[1])
        )
        
        self.layer2 = nn.Sequential(
            BasicBlock(in_ch=channels[1],out_ch=channels[2]),
            BasicBlock(in_ch=channels[2],out_ch=channels[2])
        )
        
        self.layer3 = nn.Sequential(
            BasicBlock(in_ch=channels[2],out_ch=channels[3]),
            BasicBlock(in_ch=channels[3], out_ch=channels[3])
        )

        self.layer4 = nn.Sequential(
            BasicBlock(in_ch=channels[3],out_ch=channels[4]),
            BasicBlock(in_ch=channels[4], out_ch=channels[4])
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=channels[4], out_features=nc)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x    
    
    
if __name__=="__main__":
    model = ResNet()
    input_tensor = torch.ones(size=(1,3,224,224))
    output = model(input_tensor)