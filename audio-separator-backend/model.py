import torch
import torch.nn as nn

class AudioSeparatorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=6):  # ปรับเพิ่มเป็น 6 เครื่องดนตรี
        super(AudioSeparatorUNet, self).__init__()
        
        # ฝั่งบีบอัดข้อมูล (Encoder)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ฝั่งขยายข้อมูลกลับ (Decoder)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # ชั้นสุดท้ายส่งผลลัพธ์กระจายออกเป็น 6 ชิ้นดนตรี
        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        
        d3 = self.up3(e4)
        # ปรับขนาดภาพความถี่ให้แมตช์กันหากมีมิติเศษเกินเล็ดลอดมา
        if d3.shape != e3.shape:
            d3 = nn.functional.interpolate(d3, size=(e3.shape[2], e3.shape[3]), mode='bilinear', align_corners=False)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = nn.functional.interpolate(d2, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = nn.functional.interpolate(d1, size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=False)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final_layer(d1)
