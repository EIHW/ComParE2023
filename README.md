# ComParE23 - Complaint
This repository provides the code for running the official baselines for the Complaint subchallenge of ComParE2023.

<!-- > Björn W. Schuller, Anton Batliner, Shahin Amiriparian, Christian Bergler, Maurice Gerczuk, Natalie Holz, Pauline Larrouy-Maestri, Sebastian Bayerl, Korbinian Riedhammer, Adria Mallol-Ragolta, Maria Pateraki, Harry Coppock, Ivan Kiskin, Marianne Sinka, Stephen Roberts, "The ACM Multimedia 2022 Computational Paralinguistics Challenge: Vocalisations, Stuttering, Activity, & Mosquitos," in *Proceedings of the 30th International Conference on Multimedia*, (Lisbon, Portugal), ACM, 2022. -->

<!-- ```bibtex
@inproceedings{Schuller21-TAM,
author = {Bj\”orn W.\ Schuller and Anton Batliner and Shahin Amiriparian and Christian Bergler and Maurice Gerczuk and Natalie Holz and Pauline Larrouy-Maestri and Sebastian Bayerl and Korbinian Riedhammer and Adria Mallol-Ragolta and Maria Pateraki and Harry Coppock and Ivan Kiskin and Marianne Sinka and Stephen Roberts},
title = {{The ACM Multimedia 2022 Computational Paralinguistics Challenge: Vocalisations, Stuttering, Activity, \& Mosquitos}},
booktitle = {{Proceedings of the 30th International Conference on Multimedia}},
year = {2022},
address = {Lisbon, Portugal},
publisher = {ACM},
month = {October},
note = {to appear},
}
``` -->

## Adding the data
Drop the data into `./data`, creating this directory structure:
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
You can use either [devenv](https://devenv.sh) or the pre-built [docker image](https://hub.docker.com/repository/docker/mauricege/compare23-shell/).

### devenv
Just run `devenv shell` after you followed the [installation instructions](https://devenv.sh/getting-started/).

### Docker
Run the docker container and mount the repository:
```console
docker run -it v /path/to/ComParE2023:/ComParE2023 mauricege/compare23-shell
```

## Reproducing the baseline
To reproduce the complete baseline (including feature extraction), run:
```console
dvc repro
```

If you want to skip feature extraction and instead use the provided features as they are, commit them to dvc:
```console
dvc commit features
```
and then run `dvc repro` as above.

