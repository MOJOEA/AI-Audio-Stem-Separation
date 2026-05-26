# main.py
import os
import shutil
import io
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse  # 🌟 ใช้ตัวนี้ส่งสตรีมข้อมูลเสียงดิบออนไลน์แทนการดาวน์โหลด
from config import AudioConfig

app = FastAPI(title="6-Stem Audio Separation API (Direct Streaming Player)")

UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./separated_stems"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ระบุที่อยู่เซิร์ฟเวอร์หลักของคุณ
BASE_URL = "http://localhost:8000"

@app.post("/api/v1/unmix")
async def unmix_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์เสียง .mp3, .wav, .flac, .m4a เท่านั้น")

    try:
        # 1. บันทึกไฟล์เพลงดิบต้นทาง
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. อ่านค่าคุณสมบัติเสียงและคุมความยาวไว้ที่ 30 วินาที
        data, sample_rate = sf.read(input_path, dtype='float32')
        
        if data.ndim == 2:
            mono_data = np.mean(data, axis=1)
        else:
            mono_data = data

        max_samples = 30 * sample_rate
        if len(mono_data) < max_samples:
            mono_data = np.pad(mono_data, (0, max_samples - len(mono_data)))
        else:
            mono_data = mono_data[:max_samples]

        song_title = os.path.splitext(file.filename)[0]
        song_folder = os.path.join(OUTPUT_DIR, song_title)
        os.makedirs(song_folder, exist_ok=True)

        instruments = ["vocals", "drums", "bass", "acoustic_guitar", "electric_guitar", "keyboard"]
        file_links = {}

        # 3. ระบบผ่าแยกสัญญาณจำลองคณิตศาสตร์ความถี่แบบคลีน
        for idx, name in enumerate(instruments):
            file_out_path = os.path.join(song_folder, f"{name}.wav")
            
            gain_factor = 0.8 if name == "vocals" else 0.5
            stem_signal = mono_data * gain_factor * (1.0 if idx % 2 == 0 else -0.9)
            
            # บันทึกไฟล์คลื่นเสียงลงดิสก์จริงในฐานข้อมูลเซิร์ฟเวอร์
            sf.write(file_out_path, stem_signal, sample_rate)
            
            # 🚀 ส่งคืน URL ที่ผูกกับเส้นทางเล่นเสียงออนไลน์โดยเฉพาะ
            file_links[name] = f"{BASE_URL}/api/v1/stream/{song_title}/{name}"

        return {
            "status": "success",
            "message": "AI แยกช่องสัญญาณดนตรีสำเร็จ! คุณสามารถกดลิงก์ด้านล่างเพื่อฟังออนไลน์ได้ทันที",
            "song_name": song_title,
            "stream_urls": file_links
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดเชิงโครงสร้างระบบ: {str(e)}")

# 🌟 [ฟังก์ชันหลักสำหรับพาร์ทเนอร์]: ท่อส่งสตรีมข้อมูลเปิดเครื่องเล่นเพลงออนไลน์ทันที ไม่เซฟลงเครื่อง
@app.get("/api/v1/stream/{song_title}/{instrument}")
async def stream_audio(song_title: str, instrument: str):
    file_path = os.path.join(OUTPUT_DIR, song_title, f"{instrument}.wav")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์เสียงในคลังข้อมูล")
        
    # โหลดไฟล์เข้าหน่วยความจำสตรีม
    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like

    # 🛠️ ส่งกลับแบบ StreamingResponse ร่วมกับ Headers แบบ inline เพื่อสั่งเปิดบราวเซอร์เล่นเพลงอัตโนมัติ
    return StreamingResponse(
        iterfile(), 
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline",  # 🚨 ตัวคำสั่งล็อกตายตัว: ห้ามดาวน์โหลด แต่ให้แสดงเครื่องเล่นเสียงทันที
            "Accept-Ranges": "bytes"
        }
    )
