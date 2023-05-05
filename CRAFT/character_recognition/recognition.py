import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate
from .model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import json

def process_cropped_words(opt):

    """ model configuration """
    if 'CTC' in opt["Prediction"]:
        converter = CTCLabelConverter(opt["character"])
    else:
        converter = AttnLabelConverter(opt["character"])
    opt["num_class"] = len(converter.character)

    if opt["rgb"]:
        opt["input_channel"] = 3
    model = Model(opt)

    print(
        f'model input parameters {opt["imgH"]}, {opt["imgW"]}, {opt["num_fiducial"]}, {opt["input_channel"]}, {opt["output_channel"]}, '
        f'{opt["hidden_size"]}, {opt["num_class"]}, {opt["batch_max_length"]}, {opt["Transformation"]}, {opt["FeatureExtraction"]}, '
        f'{opt["SequenceModeling"]}, {opt["Prediction"]}')
    model = torch.nn.DataParallel(model).to(device)

    # load model
    model_path = opt["saved_model"]
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(os.path.join(model_path), map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt["imgH"], imgW=opt["imgW"], keep_ratio_with_pad=opt["PAD"])
    demo_data = RawDataset(root=opt["image_folder"], opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt["batch_size"],
        shuffle=False,
        num_workers=int(opt["workers"]),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    predictions = []
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt["batch_max_length"]] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt["batch_max_length"] + 1).fill_(0).to(device)

            if 'CTC' in opt["Prediction"]:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                
                
                start = opt["image_folder"]
                path = os.path.relpath(img_name, start)

                image_name=os.path.basename(path)

                image_name = image_name[:-4]
                coords = image_name.split('_')[1:]
                
                if 'Attn' in opt["Prediction"]:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                # print(f'{image_name:25s}\t {pred:25s}\t {confidence_score:0.4f}')

                xmin, xmax, ymin, ymax = float(coords[0]), float(coords[2]), float(coords[1]), float(coords[5])
                tpl = (pred, xmin, ymin, xmax, ymax, round(float(confidence_score), 3))
                predictions.append(tpl)                

    return predictions
  

def recognize_characters():
    with open('OCR-settings.json') as f:
        data = json.load(f)
    
    """ vocab / character number configuration """
    if data["sensitive"]:
        data["character"] = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    data["num_gpu"] = torch.cuda.device_count()

    return process_cropped_words(data)