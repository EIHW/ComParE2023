#!/usr/bin/env python
import argparse
import json
import os
import pandas as pd
import torch
import torchaudio
import librosa
import yaml

from os import makedirs
from os.path import basename
from tqdm import tqdm
from glob import glob
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
)


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["wav2vec"]
    device = "cpu"
    INPUT_DIR = "./data/wav"
    OUTPUT_DIR = "./data/features/wav2vec"
    wavs = sorted(glob(f"{INPUT_DIR}/*.wav"))
    index = list(map(basename, wavs))

    vocab_dict = {}
    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json")
    tokenizer.save_pretrained("./tokenizer")

    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            params["model"]
        )
    except OSError:
        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )

    processor = Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)
    model = Wav2Vec2Model.from_pretrained(params["model"]).to(
        device
    )
    model.eval()


    embeddings = torch.zeros(len(index), 1024)
    for counter, wav in tqdm(enumerate(wavs)):
        audio, fs = librosa.core.load(wav)
        audio = torch.from_numpy(audio)
        if fs != 16000:
            audio = torchaudio.transforms.Resample(fs, 16000)(audio)
        if len(audio.shape) == 2:
            audio = audio.mean(0)
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            embeddings[counter, :] = (
                model(
                    inputs.input_values.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                )[0]
                .cpu()
                .mean(1)
                .squeeze(0)
            )
    makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(
        data=embeddings.numpy(),
        columns=[f"Neuron_{x}" for x in range(embeddings.shape[1])],
        index=index,
        index_label="name"
    ).reset_index().to_csv(f"{OUTPUT_DIR}/features.csv", index=False)
