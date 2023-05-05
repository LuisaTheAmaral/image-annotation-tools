from character_recognition import recognition
from word_cropper import detection, crop_images
import os
import shutil
import numpy as np
from skimage import io
import cv2
from io import BytesIO
import json

IMGS_PATH = "../images"

images = os.listdir(IMGS_PATH)

for img_path in images:
    img = io.imread(IMGS_PATH + '/' + img_path)

    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    bbox_score = detection.run_detection(trained_model=os.path.join('craft_mlt_25k.pth'), image=img)
    crop_images.crop_words(img, bbox_score)
    predictions = recognition.recognize_characters()


    with open('OCR-settings.json') as f:
        data = json.load(f)

    folder = data["image_folder"]

    #clean up folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    print(f"Image: {img_path}")
    if predictions:
        for pred in predictions:
            print(f"Detected '{pred[0]}' with coordinates xmin: {pred[1]} ymin: {pred[2]} xmax: {pred[3]} ymax: {pred[4]} and score of {pred[5]}")
    else:
        print("No OCR detected")
    print("------------------------------\n")



