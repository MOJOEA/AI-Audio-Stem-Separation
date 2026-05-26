# dataset.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from config import AudioConfig
from audio_utils import FourierProcessor

class MUSDB18Dataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.num_samples = int(AudioConfig.SAMPLE_RATE * AudioConfig.DURATION)
        self.track_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.track_folders)

    def __getitem__(self, idx):
        track_path = self.track_folders[idx]
        stem_waves = []
        
        # โหลดแทร็กแยกดนตรีทั้ง 4 ชิ้น
        for stem in AudioConfig.STEM_NAMES:
            wave, sr = torchaudio.load(os.path.join(track_path, f"{stem}.wav"))
            # จัดการตัดความยาวเพลงให้ตรงตาม Config
            if wave.shape[1] > self.num_samples:
                start = torch.randint(0, wave.shape[1] - self.num_samples, (1,)).item() if os.path.basename(track_path).startswith("train") else 0
                wave = wave[:, start:start + self.num_samples]
            wave = torch.mean(wave, dim=0, keepdim=True) # แปลงเป็น Mono
            stem_waves.append(wave)
            
        # รวมเสียงเป็นเพลงมิกซ์ (Mixture)
        mix_wave = sum(stem_waves)
        
        # 🛠️ เรียกใช้ FourierProcessor แปลงเป็นสเปกตรัมเฉพาะความดัง (Magnitude) ไปใช้เทรน
        mix_mag, _ = FourierProcessor.wav_to_spectrogram(mix_wave)
        
        target_mags = []
        for sw in stem_waves:
            smag, _ = FourierProcessor.wav_to_spectrogram(sw)
            target_mags.append(smag)
        target_mags = torch.cat(target_mags, dim=0) # [4, Freq, Time]
        
        return mix_mag, target_mags
