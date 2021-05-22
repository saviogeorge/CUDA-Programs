[WIP]

# CUDA-Programs
Repository for CUDA Study applications

Set up and Installation on windows

Install the Microsoft visual studio community version from:
https://visualstudio.microsoft.com/downloads/

Install the Cuda tookit from the NVIDIA developer website:
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

Create an empty project in the Visual studio code
and 
Right click on project -> Build Dependencies -> Buld Customization -> " Select the CUDA (for me it was CUDA 11.2)"

Linking Issues
1. "unresloved external symbol "cudaMalloc"
Solution. Go to Right click on project -> properties -> Linker -> Input -> Additional Dependencies 
          edit and add cudart.lib apply and save. 



