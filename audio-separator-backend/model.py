# model.py
import torch
import torch.nn as nn

class UNetAudio(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(UNetAudio, self).__init__()
        # ตัวอย่างโครงสร้างย่อของ U-Net เพื่อให้เห็นทิศทาง Data Flow
        # Down-sampling (วิเคราะห์ภาพ)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Up-sampling (ขยายภาพกลับเพื่อทำนาย)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # 🌟 สำคัญมาก: บังคับค่าให้อยู่ระหว่าง 0.0 - 1.0 เพื่อเป็นแผ่นกรอง (Mask)
        )

    def forward(self, x):
        # x คือภาพ Magnitude Spectrogram [Batch, 1, Freq, Time]
        d1 = self.down1(x)
        mask = self.up1(d1)
        return mask # Output: [Batch, 4, Freq, Time]
