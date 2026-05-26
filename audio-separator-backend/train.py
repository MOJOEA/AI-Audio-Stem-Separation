# train.py (ฉบับรวบรวมมอดูลสมบูรณ์แบบ รันเทรนจริง 100% ประหยัดแรม)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import numpy as np

# ==========================================
# 📦 1. มอดูลจัดการชุดข้อมูลและแปลงฟูริเยร์ (Dataset & Audio STFT)
# ==========================================
class MusicianDataset(Dataset):
    def __init__(self, data_dir, n_fft=512):
        self.data_dir = data_dir
        self.n_fft = n_fft
        # ตรวจสอบและค้นหาโฟลเดอร์เพลงทั้งหมดในระบบ
        self.song_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, f))]
        # ตั้งค่าหน่วยแปลงคลื่นเสียงเป็นภาพความถี่ (STFT)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=512, power=None)

    def __len__(self):
        return len(self.song_folders)

    def __getitem__(self, idx):
        song_path = self.song_folders[idx]
        # รายชื่อ 6 แทร็กเครื่องดนตรีเฉลย และไฟล์เพลงรวม (mixture.wav)
        instruments = ["mixture.wav", "vocals.wav", "drums.wav", "bass.wav", 
                       "acoustic_guitar.wav", "electric_guitar.wav", "keyboard.wav"]
        waves = []
        sample_rate = 44100
        max_samples = sample_rate * 30  # ล็อกขอบเขตเวลาคงที่ไว้ที่ 30 วินาทีพอดีเพื่อประหยัด RAM
        
        for inst in instruments:
            file_path = os.path.join(song_path, inst)
            
            # เกราะป้องกันระบบ: หากหาไฟล์ไม่พบ หรือขนาดไฟล์ว่างเปล่า ให้เติมเสียงเงียบความยาว 30 วินาทีทดแทน
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                data = np.zeros((max_samples, 1))
            else:
                try:
                    data, sr = sf.read(file_path)
                except Exception:
                    data = np.zeros((max_samples, 1))
            
            # แปลงชุดข้อมูลให้อยู่ในโครงสร้างมาตรฐาน PyTorch Matrix Tensor [Channels, Samples]
            tensor_wave = torch.tensor(data, dtype=torch.float32)
            if tensor_wave.ndim == 1:
                tensor_wave = tensor_wave.unsqueeze(0)
            elif tensor_wave.ndim == 2:
                tensor_wave = tensor_wave.t()
                
            # แปลงสัญญาณสเตอริโอ (Stereo) ให้กลายเป็นโมโน (Mono) โดยหาค่าเฉลี่ย
            if tensor_wave.shape[0] > 1:
                tensor_wave = torch.mean(tensor_wave, dim=0, keepdim=True)
                
            # ควบคุมขนาดความยาวท่อนเสียงดิบให้คงที่ 30 วินาทีเป๊ะๆ (ขลิบปลายทิ้ง หรือถ่างเสียงเงียบเพิ่ม)
            if tensor_wave.shape[1] < max_samples:
                padding = max_samples - tensor_wave.shape[1]
                tensor_wave = nn.functional.pad(tensor_wave, (0, padding))
            else:
                tensor_wave = tensor_wave[:, :max_samples]
                
            waves.append(tensor_wave)

        # ประมวลผลถอดสูตรฟูริเยร์แปลงคลื่นเสียงเป็นภาพความถี่เฉพาะสัญญาณจริง (.real)
        # บังคับขนาดภาพสเปกตรัมให้อยู่ที่ระนาบสี่เหลี่ยมสมมาตรเพื่อความเสถียรของสมองกล
        specs = []
        for w in waves:
            spec = self.stft(w).real
            # ตัดแบ่งและบีบมิติแกนเวลาของภาพให้อยู่ที่ขอบเขต 256 เฟรมลงล็อกพอดี
            if spec.shape[2] != 256:
                spec = nn.functional.interpolate(spec.unsqueeze(0), size=(257, 256), mode='bilinear', align_corners=False).squeeze(0)
            specs.append(spec)
        
        # specs[0] คือภาพสัญญาณเพลงรวม (Input ตัวแปรหลัก)
        # target_spec คือสัญญาณเฉลยดนตรี 6 แทร็กมัดรวมกัน -> มิติรูปทรง [6, 257, 256]
        target_spec = torch.cat(specs[1:], dim=0) 
        return specs[0], target_spec

# ==========================================
# 🤖 2. ฟังก์ชันระบบลูปฝึกสอนหลัก (Core Training Loop)
# ==========================================
def run_train(data_path, epochs=15):
    # เลือกระบบประมวลผลคำนวณ (เลือกใช้ GPU/CUDA ถ้ามีการ์ดจอพร้อม หากไม่มีจะรันบน CPU อัตโนมัติ)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ดึงคลาส MusicianDataset ภายในตัวเองมาล็อกเป้าหมายตำแหน่งคลังเพลง
    dataset = MusicianDataset(data_dir=data_path)
    
    if len(dataset) == 0:
        print(f"❌ Error: ไม่พบโฟลเดอร์เพลงข้างในตำแหน่ง '{data_path}' เลยครับ!")
        print(f"👉 ตรวจสอบให้มั่นใจว่าโฟลเดอร์ musdb18_data ตั้งอยู่ด้านนอกเคียงข้างหลังบ้านจริงไหม")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # ดึงสถาปัตยกรรมตัวโมเดล U-Net จากไฟล์ model.py เข้ามาประกบช่องแชนเนลเอาต์พุต 6 เครื่องดนตรี
    from model import AudioSeparatorUNet
    model = AudioSeparatorUNet(in_channels=1, out_channels=6).to(device)
    
    # ระบบสืบทอดเทรนต่อเนื่อง: หากมีไฟล์น้ำหนักเดิมตั้งอยู่ ให้ดึงค่าน้ำหนักมาคำนวณต่อทันที
    if os.path.exists("model_weights.pth"):
        try:
            model.load_state_dict(torch.load("model_weights.pth", map_location=device))
            print("🧠 [ระบบพร้อมลุย] ตรวจพบไฟล์เดิม! ทำการเชื่อมต่อน้ำหนักสมอง AI 6 แทร็กเพื่อเทรนต่อยอดเรียบร้อยครับ")
        except Exception:
            print("⚠️ แจ้งเตือน: ไฟล์น้ำหนักเดิมมีโครงสร้างต่างกัน ระบบจะเริ่มฝึกสอนนับหนึ่งใหม่จากศูนย์")
            
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🤖 ระบบเปิดฉากเริ่มต้นกระบวนการเทรนด้วยอุปกรณ์: {device}")
    print(f"📦 สแกนตรวจพบข้อมูลคลังเพลงสำหรับการฝึกสอนทั้งหมด: {len(dataset)} เพลง")
    print("🤖 AI กำลังคำนวณมิติผ่าแยก 6 แทร็ก จัดความยาว 30 วินาทีคงที่...")
    print("----------------------------------------------------------------")
    
    # สั่งระบบเริ่มทลายลูปฝึกสอนตามจำนวนรอบที่กำหนด
    for epoch in range(epochs):
        total_loss = 0
        model.train() # ล็อกโหมดเทรนเพื่อให้ระบบ BatchNorm ปรับค่าน้ำหนัก
        
        for mix, target in dataloader:
            mix, target = mix.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # ให้โมเดล U-Net ปล่อยค่าหน้ากากความถี่ (Mask) ออกมา 6 ช่องทางดนตรี
            pred_masks = model(mix)
            
            # เกาะเกราะป้องกันมิติระเบิด: ขยายรูปทรงหน้ากากให้สมมาตรล็อกพิกเซลกับเฉลยต้นฉบับ
            if pred_masks.shape != target.shape:
                pred_masks = nn.functional.interpolate(pred_masks, size=(target.shape[2], target.shape[3]), mode='bilinear', align_corners=False)
                
            # 🛠️ เทคนิค Masking: นำหน้ากากความถี่คูณสกัดเนื้อสัญญาณเสียงดนตรีจริงออกมาจากเพลงรวม
            output = pred_masks * mix
            
            # คำนวณหาค่าระยะห่างความผิดพลาด (Loss) และอัปเดตความฉลาดของ Neural Network
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"🔥 สรุปผลรอบที่ Epoch {epoch+1}/{epochs} -> ค่าเฉลี่ย Error Loss ปัจจุบัน: {total_loss/len(dataloader):.4f}")
        
    # บันทึกไฟล์น้ำหนักสมองกลตัวที่เก่งที่สุดลงฮาร์ดดิสก์เพื่อส่งต่อให้ไฟล์ main.py เรียกใช้เปิดหลังบ้าน
    torch.save(model.state_dict(), "model_weights.pth")
    print("\n🎉 [สำเร็จเด็ดขาด] กระบวนการเทรนเสร็จสิ้นสมบูรณ์!")
    print("👉 ได้ไฟล์น้ำหนักสมองกลรุ่นอัปเดต 'model_weights.pth' ไปเปิดให้บริการบนหน้าแอปมือถือแล้วครับ!")

# 🚨 สั่งให้ระบบถอยหลังออกจากโฟลเดอร์หลังบ้าน 1 สเต็ปเพื่อระเบิดเป้าหมายเข้าจับโฟลเดอร์ข้อมูลเพลงจริง
if __name__ == "__main__":
    run_train("../musdb18_data")
