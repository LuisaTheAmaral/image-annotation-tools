# CLIP

[Original repository](https://github.com/rmokady/CLIP_prefix_caption)

### Installation

Prepare the environment:

```bash
conda env create -f environment.yml
conda activate clip
```

Download the weights:

[COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view)

[Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view)

Run the detection script:

```bash
python3 caption_prediction.py
```


### Run it in the browser

Alternatively, run it in the [browser](https://replicate.ai/rmokady/clip_prefix_caption):


### Acknowledgments 

```
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}
```