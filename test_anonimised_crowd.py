
from deep_privacy import cli
import argparse
import datetime
import random
import time
from pathlib import Path
from deep_privacy.build import available_models
import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-s", "--source_path",
        help="Target path to infer. Can be video or image, or directory",
        default="test_examples/images"
    )
    parser.add_argument(
        "-t", "--target_path",
        help="Target path to save anonymized result.\
                Defaults to subdirectory of config file."
    )


    return parser

class P2pConfigs:

    weight_path = './weights/SHTechA.pth'
    img_path = "./input/img_27.jpg"
    threshold = 0.7
    output_dir = './output/'
    row = 2
    line = 2
    backbone = 'vgg16_bn'
    gpu_id = 0

    def config_printer(self):
        return (
                "weight_path: " + str(self.weight_path) +
                " img_path: " + str(self.img_path) +
                " threshold: " + str(self.threshold) +
                " output_dir: " + str(self.output_dir) +
                " row: " + str(self.row) +
                " line: " + str(self.line) +
                " backbone: " + str(self.backbone) +
                " gpu_id: " + str(self.gpu_id)
                )

class DeepPrivacyConfig:
    model = available_models[0]
    opts = None
    config_path = None
    source_path = "./input/"
    target_path = "./output/"
    start_time = None
    end_time = None


def p2pNetInference(input_image, debug=False):

    config = P2pConfigs()
    config.img_path = input_image
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(config.gpu_id)
    print(":::printing config:::")
    print(config.config_printer())
    if torch.cuda.is_available():
        # Disable CUDA
        device = torch.device('cuda')
        torch.cuda.set_device(config.gpu_id)
    else:
        device = torch.device('cpu')
    # get the P2PNet
    model = build_model(config)
    # model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if config.weight_path is not None:
        checkpoint = torch.load(config.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    img_path = config.img_path
    #  Enter Deep Privacy!!!
    deepprivacyconfig = DeepPrivacyConfig()
    deepprivacyconfig.source_path = config.img_path
    ann_imgs = list(cli.get_anonimized(deepprivacyconfig))  # Got the annomized Image will take the first one

    
    # img_path = "./vis/img_27.jpg"
    # load the images
    # for ann_img in ann_imgs:
    
    # img_raw = Image.open(ann_img).convert('RGB')
    img_raw = Image.fromarray(ann_imgs[0], 'RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = config.threshold
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    # text writing features
    text = "Total Crowd: "+ str(predict_cnt)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    thickness = 4
    font_scale = 1.0

    # draw the predictions
    size = 2
    # img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    img_to_draw = np.array(img_raw)


    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size


    padding_x, padding_y = 10, 10

    


    (width, height) = img_to_draw.shape[:2]
    x = width - text_width - padding_x
    y = text_height + padding_y
    # (x, y) = (-10, height+10)
    # print(x)
    # print(y)
    cv2.putText(img_to_draw, text, (x, y), font, 1, color, thickness, cv2.LINE_AA)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (255, 0, 255), -1)
    # save the visualized image
    cv2.imwrite(os.path.join(config.output_dir, '{}-pred{}.jpg'.format((img_path.split('/')[-1]).split('.')[0],predict_cnt)), img_to_draw)





import  os

if __name__ == '__main__':
    file_path = "./input/"
    for path in os.listdir(file_path):
        f_path = file_path + path
        p2pNetInference(f_path)
    # parser = argparse.ArgumentParser('Merged evaluation script', parents=[get_parser()])
    # args = parser.parse_args()
    # p2pNetInference(args)
    # p2pNetInference()