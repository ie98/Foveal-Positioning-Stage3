""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torchvision.models import resnet18

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # self.down_add = nn.Sequential(
        #     nn.MaxPool2d((3,3),3),
        #     nn.Conv2d(64,16,3),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(11)
        #
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(11*11*16,2)
        # )

        self.down_add = nn.Sequential(
            nn.AdaptiveAvgPool2d(7)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 2)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        p_fc = self.down_add(x)
        p_fc = torch.flatten(p_fc, 1)
        p_fc = self.fc(p_fc)
        logits = self.outc(x)
        return logits,p_fc
