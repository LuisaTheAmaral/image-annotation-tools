import clip
import os
import torch
from transformers import GPT2Tokenizer
import skimage.io as io
import PIL.Image
import time
import os

from model import ClipCaptionModel
from prediction_generator import generate_beam, generate_without_beam

IMGS_PATH = "../images"
CPU = torch.device('cpu')

def get_device(device_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')

def predict_caption(img, pretrained_model):
    
    start = time.time()
    CUDA = get_device
    is_gpu = False
    device = CUDA(0) if is_gpu else "cpu"

    if pretrained_model == 'Conceptual captions':
        model_path = os.path.join('conceptual_weights.pt')
    else:
        model_path = os.path.join('coco_weights.pt')

    #Load model weights
    prefix_length = 10

    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=CPU))
    model = model.eval()
    model = model.to(device)

    #CLIP model + GPT2 tokenizer
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    #Inference
    use_beam_search = True

    image = io.imread(img)
    pil_image = PIL.Image.fromarray(image)

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate_without_beam(model, tokenizer, embed=prefix_embed)

    print(f"Time taken for CLIP to generate caption: {time.time()-start} seconds")
    return generated_text_prefix


images = os.listdir(IMGS_PATH)
for img in images:
    img = IMGS_PATH + '/' + img
    print(f"Image: {img}")
    cap1 = predict_caption(img, pretrained_model='Conceptual captions')
    print(f"Caption generated using Conceptual captions dataset: {cap1}")
    cap2 = predict_caption(img, pretrained_model='COCO')
    print(f"Caption generated using COCO dataset: {cap2}\n")
    