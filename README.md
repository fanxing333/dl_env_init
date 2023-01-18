# Deep Learning Hello World
This is an easy setup before the entry of Deep Learning and Pytorch, 
including installing related packages and testing your GPU environment.

## Windows 10
1. 选择cuda驱动版本
2. 选择显卡驱动版本
3. 选择cuDNN版本
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