#!/bin/zsh

img_path="/home/william/Downloads/ot_data/l_00004.png"
model_path="/home/william/Downloads/depth_anything_v2_vitl.pth"
encoder="vitl"
output_dir="/home/william/Downloads/ot_data"
device="cuda:0"

python cli.py --img_path $img_path --model_path $model_path --encoder $encoder --outdir $output_dir --device $device
