# Implementation for the paper "[Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation](https://neurips.cc/virtual/2023/poster/70795)"

## Prerequisite
You will need to have a host machine with gpu, and have a docker with `nvidia-container-runtime` enabled.

> [!TIP]
> - This repo provide an easy to use script to install docker and nvidia-container-runtime, in `./MolFM/docker` run `sudo ./setup_docker_for_host.sh` to setup your host machine.
> - You can also refer to [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if you don't have them installed.

## Quick start

### Environment setup
Clone the repo with `git clone`,
```bash
git clone https://github.com/AlgoMole/MolFM.git
```

setup environment with docker,

```bash
cd ./MolFM/docker

make # a make is all you need
```

> [!NOTE]
> - The `make` will automatically build the docker image and run the container. with your host home directory mounted to the `${HOME}/home` directory inside the container. **highly recommended**
> 
> - If you need to setup the environment manually, please refer to files `docker/Dockerfile`, `docker/asset/requirements.txt` and `docker/asset/apt_packages.txt`. 
