import torch
import io
from PIL import Image
import pandas as pd
import time
import os

IMGS_PATH = "../images"

model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True, verbose=False)
model.eval()

images = os.listdir(IMGS_PATH)

for img_name in images:
    img = Image.open(IMGS_PATH + '/' + img_name)

    start = time.time()
    with torch.no_grad():
        results = model(img)

    torch.cuda.empty_cache()

    t = round(time.time() - start, 4)

    print(f"\nImage: {img_name}")
    for idx, label in results.pandas().xyxy[0].iterrows():
        x1 = label['xmin']
        y1 = label['ymin']
        x2 = label['xmax']
        y2 = label['ymax']
        print(f"Object: {label['name']}, xmin: {x1}, ymin: {y1}, xmax: {x2}, ymax: {y2}, Confidence: {label['confidence']}")

    print(f"Predictions generated in {t} seconds")
    results.save(save_dir='results/')