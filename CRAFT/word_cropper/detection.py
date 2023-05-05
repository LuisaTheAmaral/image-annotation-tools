"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import sys
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from . import craft_utils
from . import imgproc
from . import file_utils
import json
import zipfile
import pandas as pd

from .craft import CRAFT

from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, canvas_size, mag_ratio, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys, det_scores = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    print("infer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, det_scores

'''
    :param trained_model: path of the weights file
    :param image:
    :param text_threshold: text confidence threshold
    :param low_text: text low-bound score
    :param link_threshold: link confidence threshold
    :param cuda: Use cuda for inference
    :param canvas_size: image size for inference
    :param mag_ratio: image magnification ratio
    :param poly: enable polygon type
    :param refine: enable link refiner
    :param refiner_model: pretrained refiner model
    :return: array of boxes that contain detected words
'''
def run_detection(trained_model: str, image, text_threshold: float = 0.7, low_text: float = 0.4, link_threshold: float = 0.4, cuda: bool = False, 
canvas_size: int = 1280, mag_ratio: float = 1.5, poly: bool = False, refine: bool = False, refiner_model: str = './weights/craft_refiner_CTW1500.pth'):

    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + trained_model + ')')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

        refine_net.eval()
        poly = True

    t = time.time()

    bboxes, polys, score_text, det_scores = test_net(net, image, text_threshold, link_threshold, low_text, cuda, canvas_size, mag_ratio, poly, refine_net)

    bbox_score={}

    for box_num in range(len(bboxes)):
        key = str (det_scores[box_num])
        item = bboxes[box_num]
        bbox_score[key]=item

    # save score text
    #filename, file_ext = os.path.splitext(os.path.basename(image_path))
    #mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    #cv2.imwrite(mask_file, score_text)

    #file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
    print("elapsed time : {}s".format(time.time() - t))

    return bbox_score
