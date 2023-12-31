#############################################################

import argparse
import datetime
import random
import time
from pathlib import Path
from torch.utils.data import DataLoader
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

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='D:/wtg/interpreter/CrowdCounting-P2PNet/',
                        help='path where to save')
    parser.add_argument('--weight_path', default='./checkpoints/best_mae.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    parser.add_argument('--dataset_file', default='SHHA')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    if torch.cuda.is_available():
        # Disable CUDA
        device = torch.device('cuda') 
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device('cpu')


    # print("device :::" + device)

    # device = torch.device('cpu')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    # model.eval()
    # create the pre-processing transform
    # transform = standard_transforms.Compose([
    #     standard_transforms.ToTensor(), 
    #     standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    ##########################################################################################

    # Adam is used by default
    # optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = loading_data(args.data_root)
    # create the sampler used during training
    # sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    # data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_test = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=8)
    




    print("Start testing")
    # start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    # the logger writer
    # writer = SummaryWriter(args.tensorboard_dir)

    t1 = time.time()
    result = evaluate_crowd_no_overlap(model, data_loader_test, 'cuda:0')
    t2 = time.time()

    mae.append(result[0])
    mse.append(result[1])
    # print the evaluation results
    print('=======================================test=======================================')
    print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )
    # with open(run_log_name, "a") as log_file:
    #     log_file.write("mae:{}, mse:{}, time:{}, best mae:{}".format(result[0],
    #                     result[1], t2 - t1, np.min(mae)))
    print('=======================================test=======================================')
    
    
    # # run inference
    # outputs = model(samples)
    # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    # outputs_points = outputs['pred_points'][0]

    # threshold = 0.7
    # # filter the predictions
    # points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    # predict_cnt = int((outputs_scores > threshold).sum())

    # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    # outputs_points = outputs['pred_points'][0]

    
    # text = str(predict_cnt)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # color = (0, 0, 255)
    # thickness = 5
    
    # # draw the predictions
    # size = 2
    # img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)


    # (width, height) = img_to_draw.shape[:2]
    # (x, y) = (width//2, height//2)
    # print(x)
    # print(y)
    # cv2.putText(img_to_draw, text, (x, y), font, 5, color, thickness, cv2.LINE_AA)
    # for p in points:
    #     img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # # save the visualized image
    # cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)