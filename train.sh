#!/bin/bash

echo "SIRA Training Script"
name="SIRA_Train"
description="Description"
trainset="flickr"
testset="flickr"
train_data_path="/path_to_trainset"
test_data_path="/path_to_testset"
gt_path="/path_to_ground_truth"
image_size=224
epochs=100
batch_size=128
freeze_vision=1
lr=0.0001

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --name "$name" --trainset "$trainset" --testset "$testset" --train_data_path "$train_data_path" --test_data_path "$test_data_path" --image_size "$image_size" --epochs "$epochs" --batch_size "$batch_size" --gt_path "$gt_path" --freeze_vision "$freeze_vision" --lr "$lr" --description "$description"
