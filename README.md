# ComParE23 - The HealthCall30 Corpus (HC-C)
This repository provides the code for running the official baselines for the The HealthCall30 Corpus (HC-C) subchallenge of ComParE2023.


## Getting the code
Clone this repository together with its submodules and checkout the correct branch:
```bash
git clone --recurse-submodules --branch HC-C https://github.com/EIHW/ComParE2023
```

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
docker run -it -v /path/to/ComParE2023:/ComParE2023 mauricege/compare23-shell
cd ComParE2023
```

## Reproducing the baseline
To reproduce the complete baseline (including feature extraction), run:
```console
dvc repro
```

If you want to skip feature extraction and instead use the provided features as they are, commit them to dvc:
```console
dvc commit -d features audeep_export
```
and then run `dvc repro` as above.

By default, this will generate results for "complaint" classification. You can change `target` in `params.yaml` to `request` and run `dvc repro` for request classification.

