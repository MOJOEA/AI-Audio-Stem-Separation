from fastapi import FastAPI, UploadFile, File
import torch
import torchaudio
import os
import shutil
from model import AudioSeparatorUNet

app = FastAPI(title="6-Stem Audio Separation API (30 Seconds Fix)")

UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./separated_stems"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ติดตั้งโหลดสถาปัตยกรรมและค่าน้ำหนักจริง
model = AudioSeparatorUNet(in_channels=1, out_channels=6)
WEIGHT_FILE = "model_weights.pth"

if os.path.exists(WEIGHT_FILE):
    model.load_state_dict(torch.load(WEIGHT_FILE, map_location=torch.device('cpu')))
    print("✅ ระบบทำการเชื่อมต่อน้ำหนักสมอง AI 6 แทร็กเรียบร้อยแล้ว!")
else:
    print("⚠️ แจ้งเตือน: ระบบรันด้วยสมองเปล่า เนื่องจากยังไม่พบไฟล์ model_weights.pth")
model.eval()

@app.post("/separate-audio/")
async def separate_audio(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    waveform, sample_rate = torchaudio.load(input_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 🎯 ตัดสเปกคุมความยาวไฟล์เพลงของคนที่อัปโหลดมาให้ประมวลผลนิ่งๆ ที่ 30 วินาที
    max_samples = sample_rate * 30
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    elif waveform.shape[1] < max_samples:
        padding = max_samples - waveform.shape[1]
        waveform = nn.functional.pad(waveform, (0, padding))

    stft_transform = torchaudio.transforms.Spectrogram(n_fft=512, power=None)
    spectrogram = stft_transform(waveform).real
    ai_input = spectrogram.unsqueeze(0)

    with torch.no_grad():
        ai_output = model(ai_input)

    istft_transform = torchaudio.transforms.InverseSpectrogram(n_fft=512)
    song_title = os.path.splitext(file.filename)[0]
    song_folder = os.path.join(OUTPUT_DIR, song_title)
    os.makedirs(song_folder, exist_ok=True)

    # 🎸 รายชื่อ 6 สเต็มเครื่องดนตรีตามเป้าหมายหลักของคุณ
    instruments = ["vocals", "drums", "bass", "acoustic_guitar", "electric_guitar", "keyboard"]
    file_links = {}

    for idx, name in enumerate(instruments):
        # ดึงช่องสัญญาณของแต่ละเครื่องดนตรีออกมา (มิติที่ 1 ช่อง 0 ถึง 5)
        spec_channel = ai_output[0, idx, :, :].unsqueeze(0)
        
        audio_recovered = istft_transform(spec_channel)
        file_out_path = os.path.join(song_folder, f"{name}.wav")
        
        torchaudio.save(file_out_path, audio_recovered, sample_rate)
        file_links[name] = f"/download/{song_title}/{name}.wav"

    return {
        "status": "success",
        "message": "AI แยกสัญญาณดนตรี 6 ชิ้น ความยาว 30 วินาทีเสร็จสมบูรณ์!",
        "song_name": song_title,
        "download_urls": file_links
    }
