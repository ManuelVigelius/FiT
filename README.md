![Figure](assets/figure.png)

# FiT: Flexible Vision Transformer for Diffusion Model

<p align="center">
📃 <a href="https://arxiv.org/pdf/2402.12376.pdf" target="_blank">FiT Paper</a> • 
📦 <a href="https://huggingface.co/InfImagine/FiT" target="_blank">FiT Checkpoint</a> <br> • 
📃 <a href="https://arxiv.org/pdf/2410.13925" target="_blank">FiTv2 Paper</a> • 
📦 <a href="https://huggingface.co/InfImagine/FiTv2" target="_blank">FiTv2 Checkpoint</a> <br> 
</p>

This is the official repo which contains PyTorch model definitions, pre-trained weights and sampling code for our flexible vision transformer (FiT).
FiT is a diffusion transformer based model which can generate images at unrestricted resolutions and aspect ratios.

The core features will include:
* Pre-trained class-conditional FiT-XL-2-16 (2000K) model weight trained on ImageNet ($H\times W \le 256\times256$).
* Pre-trained class-conditional FiTv2-XL-2-16 (2000K) and FiTv2-3B-2-16 (1000K) model weight trained on ImageNet ($H\times W \le 256\times256$).
* High-resolution Fine-tuned FiTv2-XL-2-32 (400K) and FiTv2-3B-2-32 (200K) model weight trained on ImageNet ($H\times W \le 512\times512$).
* A pytorch sample code for running pre-trained FiT and FiTv2 models to generate images at unrestricted resolutions and aspect ratios.

Why we need FiT?
* 🧐 Nature is infinitely resolution-free. FiT, like <a href="https://openai.com/sora" target="_blank">Sora</a>, was trained on the unrestricted resolution or aspect ratio. FiT is capable of generating images at unrestricted resolutions and aspect ratios.
* 🤗 FiT exhibits remarkable flexibility in resolution extrapolation generation.

Stay tuned for this project! 😆


## Setup
First, download and setup the repo:
```
git clone https://github.com/whlzy/FiT.git
cd FiT
```
## Installation
```
conda create -n fit_env python=3.10
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

## Sample

Image generation is done with `generate_images.py`, which uses the noise-field
sampler in `fit/noise_field_sampler/`. All configuration lives in the CONFIG
block at the top of the script — checkpoints, schedules, CFG scale and output
directory — so it takes no CLI arguments:

```
python generate_images.py
```

FID against a reference batch is computed with `measure_fid.py`.

> The original upstream DDP sampling script (`sample_fitv2_ddp.py`) and its
> RoPE-interpolation modes (NTK / YaRN) have been removed from this fork.
> For those, see the [upstream FiT repository](https://github.com/whlzy/FiT).

## Flexible Imagenet Latent Datasets

We use [SD-VAE-FT-EMA](https://huggingface.co/stabilityai/sd-vae-ft-ema) to encode an image into the latent codes.

Accordingly to our flexible training pipeline, we can deal with images with arbitrary resolutions and aspect ratios.
So we preprocess the ImageNet1k dataset according to the original height and width of an image.
Conventionally, we set patch size $p$ to $2$ and the downsampling scale $s$ of the VAE encoder is $8$, 
so an image with $H=256$ and $W=256$ will lead to $\frac{H\times W}{p^2\times s^2} = 256$ tokens. 

For our pre-training, we set the maximum token length to $256$, which corresponds image resolution size $S=H=W=256$. 
For the high-resolution fine-tuning, the token length is $1024$, which corresponds image resolution size $S=H=W=512$. 
Given an input image $I\in \mathbb{R}^{3\times H \times W}$, and target resolution size $S=256/512$, the preprocessing is:
```
If H > S and W > S:
  img_resize = Resize(I)
  latent_resize = VAE_Encode(img_resize)
  save(latent_resize)
  img_crop = CenterCrop(Resize(I))
  latent_crop = VAE_Encode(img_crop)
  save(latent_resize)
else:
  img_resize = Resize(I)
  latent_resize = VAE_Encode(img_resize)
  save(latent_resize)
```

### Dataset for Pretraining

All the image latent codes with maximum token length $256$ can be downloaded from [here](https://huggingface.co/datasets/InfImagine/imagenet1k_features_256_sd_vae_ft_ema/tree/main).

```
bash tools/download_in1k_latents_256.sh
```



### Dataset for High-resolution Fine-tuning
All the image latent codes with maximum token length $1024$ can be downloaded from [here](https://huggingface.co/datasets/InfImagine/imagenet_features_1024_sd_vae_ft_ema).


```
bash tools/download_in1k_latents_1024.sh
```

### Dataset Architecture

- imagenet1k_latents_256_sd_vae_ft_ema
  - less_than_16
    - xxxxxxx.safetensors
    - xxxxxxx.safetensors
  - from_16_to_256
    - xxxxxxx.safetensors
    - xxxxxxx.safetensors
  - greater_than_256_crop
    - xxxxxxx.safetensors
    - xxxxxxx.safetensors
  - greater_than_256_resize
    - xxxxxxx.safetensors
    - xxxxxxx.safetensors
- imagenet1k_latents_1024_sd_vae_ft_ema
  - less_than_16
    - xxxxxxx.safetensors
    - xxxxxxx.safetensors
  - from_16_to_1024
    - xxxxxxx.safetensors
    - xxxxxxx.safetensors
  - greater_than_1024_crop
    - xxxxxxx.safetensors
    - xxxxxxx.safetensors
  - greater_than_1024_resize
    - xxxxxxx.safetensors
    - xxxxxxx.safetensors

## Train
You need to determine the number of node and GPU for your training.

Train FiT and FiTv2 models:
```
bash tools/train_fit_xl.sh

bash tools/train_fitv2_xl.sh

bash tools/train_fitv2_3B.sh
```

High-resolution Fine-tuning:
```
bash tools/train_fitv2_hr_xl.sh

bash tools/train_fitv2_hr_3B.sh
```


## BibTeX
```bibtex
@article{Lu2024FiT,
  title={FiT: Flexible Vision Transformer for Diffusion Model},
  author={Zeyu Lu and Zidong Wang and Di Huang and Chengyue Wu and Xihui Liu and Wanli Ouyang and Lei Bai},
  year={2024},
  journal={arXiv preprint arXiv:2402.12376},
}
```
```bibtex
@article{wang2024fitv2,
  title={Fitv2: Scalable and improved flexible vision transformer for diffusion model},
  author={Wang, ZiDong and Lu, Zeyu and Huang, Di and Zhou, Cai and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2410.13925},
  year={2024}
}
```