# GRiT

[Original repository](https://github.com/JialianW/GRiT)

### Installation

The authors experiments are based on Python 3.8, PyTorch 1.9.0, and CUDA 11.1. 
Newer version of PyTorch may also work.

Download and install Pytorch:

```bash
conda create --name grit python=3.8 -y
conda activate grit
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Download and install Detectron2 and checkout commit ID cc87e7ec:

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout cc87e7ec
pip install -e .
cd ..
```


Download and install GRiT:

```bash
git clone https://github.com/JialianW/GRiT.git
cd GRiT
pip install -r requirements.txt
pip install protobuf==3.19.6
```

Download weights:

```bash
mkdir models && cd models
wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
cd ..
```

### Usage:

```bash
python3 demo.py --test-task DenseCap --config-file configs/GRiT_B_DenseCap.yaml --input ../../images --output visualization --opts MODEL.WEIGHTS models/grit_b_densecap_objectdet.pth
```

This will save the results in the folder ```visualization```

### Acknowledgments 

```
@article{wu2022grit,
  title={GRiT: A Generative Region-to-text Transformer for Object Understanding},
  author={Wu, Jialian and Wang, Jianfeng and Yang, Zhengyuan and Gan, Zhe and Liu, Zicheng and Yuan, Junsong and Wang, Lijuan},
  journal={arXiv preprint arXiv:2212.00280},
  year={2022}
}
```