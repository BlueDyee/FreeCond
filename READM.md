# FreeCond: Free Lunch in the input conditons of text-guided inpainting

### FreeCond is a more general version of original inpainting noise prediction function, which can boosting the existing methods with no cost.

## 0. Prepariation
```
conda create -n freecond python=3.9 -y
conda activate freecond
pip install -r requirements.txt

# (optional) SAM dependency for IoU Score computation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P data/ckpt
```
## 1. Run