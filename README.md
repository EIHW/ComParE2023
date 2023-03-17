# ComParE23 - The Hume-Prosody Corpus (HP-C)
This repository provides the code for running the official baselines for the The Hume-Prosody Corpus (HP-C) subchallenge of ComParE2023 (excluding feature extraction).


## Getting the code
Clone this repository and checkout the correct branch:
```bash
git clone --branch HP-C https://github.com/EIHW/ComParE2023
```

## Adding the data
Drop the data into `./data` (~40GB), creating this directory structure:
```console
data
├── features
│  ├── audeep
│  ├── deepspectrum
│  └── opensmile
├── lab
├── raw
│  └── wav
└── wav
```

## Installing the dependencies
Install [poetry](https://python-poetry.org/docs/)

`poetry install`

## Reproducing the baseline
To reproduce the baseline for `wav2vec2` (excluding feature extraction), run:

```console
poetry run dvc repro
```

## Notes 
If you want to reproduce the feature extraction follow the steps in the HC-C branch, using  HP-C. To avoid platform requirements, use the provided docker image. All parameters for feature extraction are given in `params.yaml`

As it is only `wav2vec2` will run, running all experiements will take some time, but you can uncomment the `dvc.yaml` to do this. 


