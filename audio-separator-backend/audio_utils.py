# audio_utils.py
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from config import AudioConfig

class FourierProcessor:
    @staticmethod
    def wav_to_spectrogram(wave: torch.Tensor) -> tuple:
        """
        แปลงคลื่นเสียงเป็น Spectrogram ด้วยระบบแปลงสำเร็จรูปของ torchaudio 
        เสถียรสูง ป้องกันปัญหา Python แครชบน Windows
        """
        # ใช้มอดูลแปลงความถี่สำเร็จรูปที่คำนวณผ่านหน่วยความจำที่ปลอดภัย
        stft_transform = T.Spectrogram(
            n_fft=AudioConfig.N_FFT,
            hop_length=AudioConfig.HOP_LENGTH,
            win_length=AudioConfig.WIN_LENGTH,
            power=None # คืนค่าเป็น Complex Tensor สำหรับดึง Phase ได้
        ).to(wave.device)
        
        stft_complex = stft_transform(wave)
        magnitude = torch.abs(stft_complex)
        phase = torch.angle(stft_complex)
        
        # ปรับขนาดแกนเวลาเป็นเลขคู่ให้เข้าคู่กับเลเยอร์ U-Net
        pad_time = (2 - (magnitude.shape[-1] % 2)) % 2
        if pad_time > 0:
            magnitude = F.pad(magnitude, (0, pad_time))
            phase = F.pad(phase, (0, pad_time))
            
        return magnitude, phase

    @staticmethod
    def spectrogram_to_wav(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        แปลงสเปกตรัมกลับเป็นคลื่นเสียงดิบ ด้วยระบอินเวอร์สสำเร็จรูป (InverseSpectrogram)
        """
        stft_complex = torch.polar(magnitude, phase)
        
        istft_transform = T.InverseSpectrogram(
            n_fft=AudioConfig.N_FFT,
            hop_length=AudioConfig.HOP_LENGTH,
            win_length=AudioConfig.WIN_LENGTH
        ).to(magnitude.device)
        
        wave = istft_transform(stft_complex)
        return wave
