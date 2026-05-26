# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import AudioConfig
from model import UNetAudio
from dataset import MUSDB18Dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetAudio(in_channels=1, out_channels=AudioConfig.NUM_STEMS).to(device)
    
    dataset = MUSDB18Dataset(data_dir="./data")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss() # โครงสร้างการคำนวณระยะห่างคลื่นความถี่แบบพื้นฐาน
    
    print("🎬 เริ่มลูปฝึกสอนโมเดลแยกเสียง...")
    for epoch in range(10):
        for mix_mag, target_mags in loader:
            mix_mag, target_mags = mix_mag.to(device), target_mags.to(device)
            
            optimizer.zero_grad()
            pred_masks = model(mix_mag)
            
            # นำหน้ากากไปกรองเพลงรวมเพื่อให้ได้มิติเสียงของชิ้นดนตรี
            pred_mags = pred_masks * mix_mag
            
            loss = criterion(pred_mags, target_mags)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} สำเร็จ - Loss: {loss.item():.6f}")
        torch.save(model.state_dict(), f"./checkpoints/unet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
