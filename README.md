[WIP]

# CUDA-Programs
Repository for CUDA Study applications

## Set up and Installation on windows

1. Install the Microsoft visual studio community version from: [microsoft visual studio](https://visualstudio.microsoft.com/downloads/)

2. Install the Cuda tookit from the NVIDIA developer website: [CUDA 11.3](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

3. Create an empty project in the Visual studio code

          Right click on project -> Build Dependencies -> Buld Customization -> " Select the CUDA (for me it was CUDA 11.2)"

4. Nsight tools for Debugging CUDA kernel and profiling :
  
   From Visual studio [market place](https://marketplace.visualstudio.com/items?itemName=NVIDIADevToolsTeam.nvnsighttoolsvsintegration)
   
   Installation tips from [Nvidia](https://developer.nvidia.com/nvidia-nsight-integration-install-tips)
   
   Debugging a CUDA Application using
   [Nsight-1](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Debugging_CUDA_Application.htm)
   [Nsight-2](https://docs.nvidia.com/nsight-visual-studio-edition/index.html)
   
   Nsight compute helps tp profile your kernels
          
          


## Build errors/Linking Issues
1. "unresloved external symbol "cudaMalloc"

         Solution. 
         
         Go to Right click on project -> properties -> Linker -> Input -> Additional Dependencies 
         edit and add cudart.lib apply and save.
         
2: Issue when launching Nsight compute and profiling the kernel. "ERR_NVGPUCTRPERM The user running the target application does not have permission to access NVIDIA GPU        Performance Counters on the target device"
   [link](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute)

          Solution.
          
          As stated in the above link start your Nsight compute application as administrator
          
 
         




