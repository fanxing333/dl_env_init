# Deep Learning Hello World
This is an easy setup before the entry of Deep Learning and Pytorch, 
including installing related packages and testing your GPU environment.

## Windows 10
1. 安装相应显卡驱动 [NVDIA官网](https://www.nvidia.com/download/index.aspx?lang=en-us)
2. 选择cuda驱动版本 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
   1. 配置环境变量：在系统变量中的Path中加入
   ```bash
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\libnvvp
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64
   ```
   2. 检查显卡驱动及 CUDA
   ```bash
   nvidia-smi
   nvcc -V
   ```
3. 选择cuDNN版本 [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-download)
   1. 解压缩包，并将`bin`、`include`及`lib`文件夹拷贝至`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7`目录下
4. 选择Pytorch版本

5. 安装Anaconda
进入Power Shell创建新Conda环境
   ```bash
   $ conda create -n torch-gpu python=3.8
   $ conda activate torch-gpu
   # 选择相应的Pytorch版本安装 例如
   $ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
   # 安装Jupyterlab
   $ conda install -c conda-forge jupyter jupyterlab
   ```
   设置jupyterlab远程登录

6. 安装Git

7. 运行测试程序