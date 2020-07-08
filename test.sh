#!/usr/bin/env bash

python test.py --param_path /home/lart/Coding/HDFFile/output/HDFNet/PretrainedParams/HDFNet_VGG16/HDFNet_VGG16_7Datasets.pth \
               --model HDFNet \
               --testset /home/lart/Datasets/Saliency/RGBDSOD/LFSD/ \
               --has_masks True \
               --save_pre False \
               --save_path output/HDFNet/pre/test \
               --data_mode RGBD \
               --use_gpu True
