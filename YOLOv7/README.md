# Yolo v7

[Original repository](https://github.com/WongKinYiu/yolov7)

### Installation

Download Pytorch:

```bash
conda create --name yolo
conda activate yolo
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install requirements:

```bash
pip install -r requirements.txt
```

Download the weights:

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

Run the script: 

```bash
python3 yolov7.py
```

The annotated images will be stored in a folder named ```results```

### Acknowledgments 

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

