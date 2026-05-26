# config.py
class AudioConfig:
    SAMPLE_RATE = 44100      # ความละเอียดเสียงมาตรฐาน HQ
    DURATION = 4.0           # ความยาวท่อนเพลงที่ใช้เทรน (วินาที)
    
    # การตั้งค่าสำหรับ Short-Time Fourier Transform (STFT)
    N_FFT = 2048             # ขนาดหน้าต่างวิเคราะห์ความถี่
    HOP_LENGTH = 512         # ระยะขยับของหน้าต่าง (Overlap 75%)
    WIN_LENGTH = 2048        # ความยาวหน้าต่างเสียง
    
    NUM_STEMS = 4            # จำนวนช่องสัญญาณดนตรีที่จะแยก (Vocals, Drums, Bass, Other)
    STEM_NAMES = ['vocals', 'drums', 'bass', 'other']
