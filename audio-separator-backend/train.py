import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import numpy as np

# ดึงโครงสร้างโมเดล 6 แทร็กเข้ามาประมวลผล
from model import AudioSeparatorUNet  

class MusicianDataset(Dataset):
    def __init__(self, data_dir, n_fft=512):
        self.data_dir = data_dir
        self.n_fft = n_fft
        self.song_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, f))]
        # กำหนดขนาดขนาดรูปภาพความถี่ให้คงที่ [257, 256] เพื่อไม่ให้คณิตศาสตร์ของ U-Net เพี้ยน
        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=512, power=None)

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, idx):
        song_path = self.song_folders[idx]
        instruments = ["mixture.wav", "vocals.wav", "drums.wav", "bass.wav", 
                       "acoustic_guitar.wav", "electric_guitar.wav", "keyboard.wav"]
        waves = []
        sample_rate = 44100
        max_samples = sample_rate * 30  # ล็อกสเปกเวลาที่ 30 วินาทีพอดี
        
        for inst in instruments:
            file_path = os.path.join(song_path, inst)
            
            # 🛡️ ระบบเซฟตี้กันตาย: ถ้าหาไฟล์ไม่เจอ หรือไฟล์ขนาดเป็น 0 KB ให้สร้างเสียงเงียบ 30 วิ แทนทันที
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                data = np.zeros((max_samples, 1))
            else:
                try:
                    data, sr = sf.read(file_path)
                except Exception:
                    data = np.zeros((max_samples, 1))
            
            # แปลงและจัดมิติให้อยู่ในรูปแบบสากลของ PyTorch [Channels, Time]
            tensor_wave = torch.tensor(data, dtype=torch.float32)
            if tensor_wave.ndim == 1:
                tensor_wave = tensor_wave.unsqueeze(0)
            elif tensor_wave.ndim == 2:
                tensor_wave = tensor_wave.t()
                
            # รวมมิติสเตอริโอให้กลายเป็นโมโน (Mono)
            if tensor_wave.shape[0] > 1:
                tensor_wave = torch.mean(tensor_wave, dim=0, keepdim=True)
                
            # จัดความยาวให้นิ่งสนิทที่ 30 วินาทีเป๊ะๆ (ขลิบปลาย หรือ ถ่างเสียงเงียบเพิ่ม)
            if tensor_wave.shape[1] < max_samples:
                padding = max_samples - tensor_wave.shape[1]
                tensor_wave = nn.functional.pad(tensor_wave, (0, padding))
            else:
                tensor_wave = tensor_wave[:, :max_samples]
                
            waves.append(tensor_wave)

        # แปลงคลื่นเสียงเป็นภาพความถี่ตัดเอาเฉพาะสัญญาณจริง (.real)
        # บังคับขนาดภาพให้คงที่ตายตัว เพื่อให้ U-Net บีบอัดและขยายกลับได้ไม่ติดเศษทศนิยม
        specs = []
        for w in waves:
            spec = self.stft(w).real
            # บังคับตัดมิติเวลาของภาพความถี่ให้อยู่ที่ 256 เฟรมพอดีเป๊ะ
            if spec.shape[2] != 256:
                spec = nn.functional.interpolate(spec.unsqueeze(0), size=(257, 256), mode='bilinear', align_corners=False).squeeze(0)
            specs.append(spec)
        
        # specs[0] คือภาพเพลงรวม (Input) มีมิติ [1, 257, 256]
        # target_spec คือเฉลยดนตรี 6 แทร็กมัดรวมกัน มีมิติ [6, 257, 256]
        target_spec = torch.cat(specs[1:], dim=0) 
        return specs[0], target_spec

def run_train(data_path, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MusicianDataset(data_dir=data_path)
    
    if len(dataset) == 0:
        print(f"❌ Error: ไม่พบโฟลเดอร์เพลงข้างใน {data_path} กรุณาสร้างโฟลเดอร์ย่อยก่อนครับ")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = AudioSeparatorUNet(in_channels=1, out_channels=6).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🤖 ระบบเริ่มรันเทรนด้วยอุปกรณ์: {device}")
    print("🤖 กำลังคำนวณแยก 6 แทร็ก จัดเวลาคงที่ 30 วินาทีแบบสมบูรณ์...")
    
    for epoch in range(epochs):
        total_loss = 0
        for mix, target in dataloader:
            mix, target = mix.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(mix)
            
            # ตรวจเช็คเกราะป้องกันมิติผิดพลาดก่อนส่งเข้าห้องตรวจ Loss
            if output.shape != target.shape:
                output = nn.functional.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bilinear', align_corners=False)
                
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Error Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "model_weights.pth")
    print("🎉 [สำเร็จเด็ดขาด] เทรนเสร็จสมบูรณ์! ได้ไฟล์น้ำหนัก 'model_weights.pth' ไปใช้งานยาวๆ แล้วครับ")

if __name__ == "__main__":
    run_train("./musdb18_data")
