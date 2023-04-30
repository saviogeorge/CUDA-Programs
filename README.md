[WIP]

# CUDA-Programs
Repository for CUDA Study applications

## Pre-installation Actions
Uninstall previous nvidia cuda tool kit
```
sudo apt-get remove nvidia-cuda-toolkit
sudo apt-get remove --auto-remove nvidia-cuda-toolkit
sudo apt-get purge nvidia-cuda-toolkit
```

Please follow the steps specified under [Pre-Installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#pre-installation-actions) to verify if the system is CUDA - capable.


## Set up and Installation on Linux

Please follow [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation)

## Post Installation Actions

Please follow [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)

In addition i also added the path to bashrc

```
echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
```
