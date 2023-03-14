import os
from rich.progress import track
import numpy as np
import librosa.core
import scipy
from glob import glob
from os.path import basename

SRC = 'data/raw/wav'
DST = 'data/wav'

    
if __name__=='__main__':
    os.makedirs(DST, exist_ok=True)
    wavs = sorted(glob(f"{SRC}/**/*.wav", recursive=True))

    for wav in track(wavs):
        audio, sr = librosa.core.load(wav, sr=16000)
        audio = audio * (0.7079 / np.max(np.abs(audio)))
        maxv = np.iinfo(np.int16).max
        audio = (audio * maxv).astype(np.int16)
        scipy.io.wavfile.write(f'{DST}/{basename(wav)}', sr, audio)


           


