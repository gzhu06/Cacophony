from dataclasses import dataclass

@dataclass
class VGGSoundConfig:
    data_dir: str = '/storageHDD/ge/audio_sfx_raw/vggsound'
    sampling_rate: int = 48000
        
@dataclass
class AudioCaps16kConfig:
    data_dir: str = '/storageHDD/ge/audio_sfx_wav/audiocaps'
    sampling_rate: int = 16000
        
@dataclass
class Clotho16kConfig:
    data_dir: str = '/storageHDD/ge/audio_sfx_wav/clothov2'
    sampling_rate: int = 16000

@dataclass
class TUTAS2017Config:
    data_dir: str = '/storageNVME/ge/TUT_Acoustic_scenes_2017'
    sampling_rate: int = 44100

@dataclass
class ESC50Config:
    data_dir: str = '/storageHDD/ge/audio_sfx_raw/esc50/ESC-50-master'
    sampling_rate: int = 44100

@dataclass
class US8KConfig:
    data_dir: str = '/storageHDD/ge/audio_sfx_raw/us8k/UrbanSound8K'
    sampling_rate: int = 44100
