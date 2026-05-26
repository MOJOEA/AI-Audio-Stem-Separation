# audio_utils.py
import torch
import torchaudio
from config import AudioConfig

class FourierProcessor:
    @staticmethod
    def wav_to_spectrogram(wave: torch.Tensor) -> tuple:
        """
        แปลงคลื่นเสียง (Waveform) เป็น Spectrogram ด้วยสูตร STFT
        ส่งออกสองค่า: Magnitude (ความดัง) และ Phase (เฟสเสียง)
        """
        # คำนวณ STFT จะได้ค่า Complex Number (มิติ: [Channels, Freq, Time])
        stft_complex = torch.stft(
            wave,
            n_fft=AudioConfig.N_FFT,
            hop_length=AudioConfig.HOP_LENGTH,
            win_length=AudioConfig.WIN_LENGTH,
            window=torch.hann_window(AudioConfig.WIN_LENGTH).to(wave.device),
            return_complex=True
        )
        magnitude = torch.abs(stft_complex)
        phase = torch.angle(stft_complex)
        return magnitude, phase

    @staticmethod
    def spectrogram_to_wav(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        
        # ประกอบร่างความดังและเฟสกลับเป็น Complex Number
        stft_complex = torch.polar(magnitude, phase)
        
        wave = torch.istft(
            stft_complex,
            n_fft=AudioConfig.N_FFT,
            hop_length=AudioConfig.HOP_LENGTH,
            win_length=AudioConfig.WIN_LENGTH,
            window=torch.hann_window(AudioConfig.WIN_LENGTH).to(magnitude.device)
        )
        return wave
