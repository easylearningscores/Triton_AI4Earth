 # <p align=center> Advanced long-term earth system forecasting by learning the small-scale nature</p>

<p align="left">
<a href="https://arxiv.org/abs/2312.08403" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2306.11249-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/blob/master/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<!-- <a href="https://huggingface.co/OpenSTL" alt="Huggingface">
    <img src="https://img.shields.io/badge/huggingface-OpenSTL-blueviolet" /></a> -->
<a href="https://openstl.readthedocs.io/en/latest/" alt="docs">
    <img src="https://readthedocs.org/projects/openstl/badge/?version=latest" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/chengtan9907/SimVPv2?color=%23FF9600" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="resolution">
    <img src="https://img.shields.io/badge/issue%20resolution-1%20d-%23B7A800" /></a>
<a href="https://img.shields.io/github/stars/chengtan9907/OpenSTL" alt="arXiv">
    <img src="[https://arxiv.org/abs/2312.08403](https://github.com/easylearningscores/DGODE_ood" /></a>
</p>

[📘Documentation](https://arxiv.org/abs/2312.08403) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](https://arxiv.org/abs/2312.08403) |
[🤗Huggingface](https://huggingface.co/easylearning/Triton_Earth_V1/tree/main) |
[👀Visualization](https://arxiv.org/abs/2312.08403) |
[🆕News](docs/en/changelog.md)


</div>
<div align=center>
<img src="figure/Figure1.jpg" width="1080">
</div>


## News 🚀🚀🚀
- `2025/05/10`: We release all weights [Triton_AI4Earth_V1](https://huggingface.co/easylearning/Triton_Earth_V1/tree/main), training, inference, and other raw files, and upload the draft of the paper.



## Documents

### 🌟 **Get Started**

#### 🤖 Environment installation

```bash
conda create -n triton_earth python=3.10.15 -y && \
conda activate triton_earth && \
conda install -c nvidia cuda-cudart=12.1.105 cuda-libraries=12.1.0 cuda-nvrtc=12.1.105 cuda-nvtx=12.1.105 cuda-opencl=12.6.77 -y && \
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y && \
pip install -r requirements.txt && \
conda install ffmpeg=4.3 libjpeg-turbo=2.0.0 -c pytorch -y
```

#### 📄 Example Usage

```bash
from Triton_model import Triton
import torch.nn.functional as F
import torch

inputs = torch.randn(1, 10, 2, 256, 256)
model = Triton(
        shape_in=(10, 2, 256, 256),
        spatial_hidden_dim=32,
        output_channels=1,
        temporal_hidden_dim=64,
        num_spatial_layers=4,
        num_temporal_layers=8)
output = model(inputs)
print(output.shape)
target = torch.rand((1, 10, 2, 256, 256))
loss = F.mse_loss(output, target)
loss.backward()
```

####  ✨ Run the train code

We currently provide the code for single-machine multi-GPU runs, such as the Kuroshio experiment. We conduct experiments on a single machine with 8 GPUs and 40GB A100. The training command is as follows:

```bash
torchrun --nnodes=1 --nproc_per_node=8 train_Kuro_triton.py
```

or
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=25641 train_Kuro_triton.py
```



