# main.py
from fastapi import FastAPI, UploadFile, File
import torch
import torchaudio
import io
from config import AudioConfig
from model import UNetAudio
from audio_utils import FourierProcessor

app = FastAPI(title="AI Audio Separation API Server")

# โหลดโมเดลรอไว้บนหน่วยความจำทันทีที่เปิดเซิร์ฟเวอร์
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetAudio(in_channels=1, out_channels=AudioConfig.NUM_STEMS).to(device)
# ในสภาวะใช้งานจริง: model.load_state_dict(torch.load("./checkpoints/unet_best.pth", map_location=device))
model.eval()

@app.post("/api/v1/unmix")
async def unmix_audio(file: UploadFile = File(...)):
    """
    เส้นทาง API สำหรับรับไฟล์เพลงเดี่ยว (.mp3/.wav) จากมือถือ 
    ประมวลผลผ่าแยก 4 แทร็กดนตรีสด แล้วบันทึกส่งกลับผลลัพธ์
    """
    # 1. อ่านไฟล์เสียงที่ส่งมาจากโมบายแอป
    audio_bytes = await file.read()
    wave, sr = torchaudio.load(io.BytesIO(audio_bytes))
    
    # ปรับแต่ง Sample Rate ให้ตรงตามที่ AI ฝึกฝนมา
    if sr != AudioConfig.SAMPLE_RATE:
        wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AudioConfig.SAMPLE_RATE)(wave)
    wave = torch.mean(wave, dim=0, keepdim=True).to(device) # แปลงเป็น Mono เข้า GPU
    
    # 2. 🛠️ ถอดสูตรฟูริเยร์แยก Magnitude และความต่อเนื่องของเฟส (Phase)
    with torch.no_grad():
        mix_mag, mix_phase = FourierProcessor.wav_to_spectrogram(wave)
        
        # ส่งภาพความถี่ให้ U-Net ทำนายหน้ากากแยกเสียง (ส่งมิติจำลอง Batch เข้าไปด้วย)
        pred_masks = model(mix_mag.unsqueeze(0)).squeeze(0) # [4, Freq, Time]
        
        # 3. นำหน้ากากความถี่ (Mask) คูณกับสเปกตรัมรวม และนำเฟสดั้งเดิม (Phase) มาใช้ประกอบกลับ
        # สร้างเป็น Dictionary เก็บผลลัพธ์เสียงที่คลีนที่สุด
        output_stems = {}
        for idx, stem_name in enumerate(AudioConfig.STEM_NAMES):
            stem_mag = pred_masks[idx] * mix_mag
            
            # 🛠️ แปลงกลับเป็นคลื่นเสียงสัญญาณสมบูรณ์ด้วยอินเวอร์สฟูริเยร์ (ISTFT)
            stem_wave = FourierProcessor.spectrogram_to_wav(stem_mag, mix_phase)
            
            # ในขั้นตอนนี้ โค้ดสามารถนำ stem_wave ไปเขียนลงไฟล์ (.wav) บน Cloud Storage (S3) 
            # และจัดส่งลิงก์ดาวน์โหลดทั้ง 4 ลิงก์กลับไปให้ UI มิกเซอร์บนมือถือของผู้ใช้ได้ทันที
            output_stems[stem_name] = f"https://cloud-storage.com{file.filename}_{stem_name}.wav"
            
    return {"status": "success", "separated_stems": output_stems}
