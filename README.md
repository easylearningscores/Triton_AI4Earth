
 
 # <p align=center> Advanced long-term earth system forecasting by learning the small-scale nature</p>
<p align="center" width="100%">
  <img src='figure/tritongpt.png' width="40%">
</p>


This repo is the official PyTorch implementation of Triton_Earth, which advances long-term Earth system forecasting by learning the small-scale nature.

<p align="left">
<a href="https://arxiv.org/abs/2505.19432" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2306.11249-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/easylearningscores/Triton_AI4Earth/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
</p>

[📘Documentation](https://arxiv.org/abs/2312.08403) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](https://arxiv.org/abs/2312.08403) |
[🤗Huggingface](https://huggingface.co/easylearning/Triton_Earth_V1/tree/main) |
[👀Visualization](https://arxiv.org/abs/2312.08403) |
[🆕News](docs/en/changelog.md)


## 📑 Open-source Plan
- [ ] Project Page
- [x] Github Page
- [x] Paper

## Architecture 🌟🌟🌟

</div>
<div align=center>
<img src="figure/Figure1.jpg" width="1080">
</div>


## News 🚀🚀🚀

- `2025/05/27`: After two weeks on hold, our paper finally appeared on arxiv.
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

####  ✨ Run the train code

We currently provide the code for single-machine multi-GPU runs, such as the Kuroshio experiment. We conduct experiments on a single machine with 8 GPUs and 40GB A100. The training command is as follows:

```bash
torchrun --nnodes=1 --nproc_per_node=8 train_Kuro_triton.py
```

or
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=25641 train_Kuro_triton.py
```

#### 🛠️ Quick Start Guide

This guide will walk you through environment setup, model configuration, and training workflow.

##### Project Structure Initialization 📂 

Run the following command to create the basic directory structure:

```bash
mkdir -p {checkpoints,dataset,logs,model,results}

Initialized project structure:
├── checkpoints/    # Model weights storage
├── dataset/        # Experimental datasets
├── logs/           # Training logs
├── model/          # Triton model directory
├── results/        # Experiment outputs
├── config.yaml     # Global configuration
├── dataloader_ns.py # Data loader implementation
├── inference_all.py # Inference interface
├── train_api.py    # Training entry point
```

##### Model Preparation ⬇️

1. Download the Triton model from official source
2. Place model files in `model/` directory
3. (Optional) Download pretrained weights to `checkpoints/`

##### Data Specification 📊
The data loader expects the following dimensions:
```bash
# Dimension verification
sample_input, sample_target = next(iter(train_loader))
print(f"📥 Input tensor shape: {sample_input.shape}")   # [B, T, C, H, W]
print(f"📤 Target tensor shape: {sample_target.shape}")  # [B, T, C, H, W]

# Expected output:
Input shape: torch.Size([32, 10, 2, 128, 128])
Target shape: torch.Size([32, 10, 2, 128, 128])

```
💡 Dimension legend: B-Batch size, T-Time steps, C-Channels, H-Height, W-Width

##### Model Initialization & Training 🧠 
Reference code for model instantiation and training:
```bash
from Triton_model import Triton
import torch.nn.functional as F
import torch

# Generate sample data
inputs = torch.randn(1, 10, 2, 256, 256)  # (B, T, C, H, W)
target = torch.rand((1, 10, 2, 256, 256))

# Initialize Triton model
model = Triton(
    shape_in=(10, 2, 256, 256),      # Input dimensions (T, C, H, W)
    spatial_hidden_dim=32,           # Spatial encoder hidden dim
    output_channels=1,               # Output channels
    temporal_hidden_dim=64,          # Temporal encoder hidden dim
    num_spatial_layers=4,            # Spatial encoder layers
    num_temporal_layers=8            # Temporal encoder layers
)

# Forward pass
output = model(inputs)
print(f"🎯 Output shape: {output.shape}")  # Expected: torch.Size([1, 10, 1, 256, 256])

# Loss calculation & backpropagation
loss = F.mse_loss(output, target)
loss.backward()
print("✅ Backpropagation completed!")
```

## Forecast Visualization 🏆🏆🏆 

### Weather forecasting 👀

<div align="center">
  <img src="figure/temperature_evolution.gif" alt="364-day Temperature Forecast Evolution" width="500"/>
  
  <br>
  
  <em>Figure: Dynamic evolution of predicted (red) versus observed (blue) global average temperatures over one year (365 days). 
  <br>Shaded region shows the absolute difference between prediction and observation.</em>
</div>

##### 🔥Note: The results can be replicated by training for 1,000 epochs on 8 40GB-A100 GPUs using the hyperparameters we have released.

----------



### Ocean simulation 👀

<div align="center">
  <img src="figure/SSTa_forecast (2).gif" alt="60-day MHW simulation" width="800"/>
  <br>
  <em>Figure: 60-day MHW simulation</em>
</div>


<div align="center">
  <img src="figure/SSSa_forecast (1).gif" alt="60-day SSSa simulation" width="800"/>
  <br>
  <em>Figure: 60-day SSSa simulation</em>
</div>

<div align="center">
  <img src="figure/SSHa_forecast (1) (1).gif" alt="60-day SSHa simulation" width="800"/>
  <br>
  <em>Figure: 60-day SSHa simulation</em>
</div>

----------


### Kuroshio forecasting 👀

<div align="center">
  <img src="figure/forecast_comparison_02.gif" alt="120-day Kuroshio Current Forecast Comparison" width="800"/>
  <br>
  <em>Figure: 120-day forecast comparison showing initial conditions (left, 2021-09-03), ground truth (middle-left), model prediction (middle-right), and absolute error (right)</em>
</div>

<div align="center">
  <img src="figure/forecast_comparison.gif" alt="120-day Kuroshio Current Forecast Comparison" width="800"/>
  <br>
  <em>Figure: 120-day forecast comparison showing initial conditions (left, 2021-10-18), ground truth (middle-left), model prediction (middle-right), and absolute error (right)</em>
</div>





### Turbulence forecasting 👀

<div align="center">
  <img src="figure/model_comparison.gif" alt="Model Comparison Animation" width="1000"/>
  
  <br>
  
  <em>Figure: Temporal evolution comparing Ground Truth with Triton, FNO, SimVP and U-Net predictions over 99 timesteps</em>
</div>


<div align="center">
  <img src="figure/model_comparison_1.gif" alt="Model Comparison Animation" width="1000"/>
  
  <br>
  
  <em>Figure: Temporal evolution comparing Ground Truth with Triton, FNO, SimVP and U-Net predictions over 99 timesteps</em>
</div>

<div align="center">
  <img src="figure/supply_main_ns_esd.png" alt="Model Comparison Animation" width="1000"/>
  <br>
  <em>Figure: Enstrophy spectra comparison at the final forecast step.</em>
</div>


## Citation

```
@article{wu2025triton,
  title={Advanced long-term earth system forecasting by learning the small-scale nature},
  author={Hao Wu and Yuan Gao and Ruiqi Shu and Kun Wang and Ruijian Gou and Chuhan Wu and Xinliang Liu and Juncai He and Shuhao Cao and Junfeng Fang and Xingjian Shi and Feng Tao and Qi Song and Shengxuan Ji and Yanfei Xiang and Yuze Sun and Jiahao Li and Fan Xu and Huanshuo Dong and Haixin Wang and Fan Zhang and Penghao Zhao and Xian Wu and Qingsong Wen and Deliang Chen and Xiaomeng Huang},
  journal={arXiv preprint arXiv:2505.19432},
  year={2025}
}
```


## Acknowledgments  🤗🤗🤗

This project draws partial inspiration from and/or incorporates code from the following open-source works: [NVIDIA Physics NeMo](https://github.com/NVIDIA/physicsnemo), [UniFormer](https://github.com/Sense-X/UniFormer), [OpenSTL](https://github.com/chengtan9907/OpenSTL), [EarthFarseer](https://github.com/easylearningscores/EarthFarseer), and [Torch-CFD](https://github.com/scaomath/torch-cfd). We gratefully acknowledge their contributions to our work.

## License 📂📂📂
This project is released under the Apache-2.0 license. Please see the [LICENSE](https://github.com/easylearningscores/Triton_AI4Earth/blob/main/LICENSE) file for more information.



## Contact ✉️✉️✉️
If you have any questions about our paper or code, please contact wuhao2022@mail.ustc.edu.cn, yuangao24@mails.tsinghua.edu.cn, srq24@mails.tsinghua.edu.cn


## Github Star History
<a href="https://star-history.com/#easylearningscores/Triton_AI4Earth&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=easylearningscores/Triton_AI4Earth&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=easylearningscores/Triton_AI4Earth&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=easylearningscores/Triton_AI4Earth&type=Date" />
 </picture>
</a>

