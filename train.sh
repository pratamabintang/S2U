#!/usr/bin/env bash
python train.py \
  --hiera_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/2.MODIFICATION_SAM2-UNet/SAM2-UNet/sam2_hiera_large.pt" \
  --train_image_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/3.TRAINING/IMAGE" \
  --train_mask_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/3.TRAINING/LABEL" \
  --train_dtm_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/3.TRAINING/DTM_NORM" \
  --val_image_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/IMAGE" \
  --val_mask_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/LABEL" \
  --val_dtm_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/4.VALIDATION/DTM_NORM" \
  --save_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/6.TRAINING RESULT/2.MODIFY_CODE/IMAGE_DTM/0%_MODIFICATION3" \
  --trainsize 512 \
  --dtm_availability 0.0 \
  --lr 0.0001 \
  --epoch 50 \
  --batch_size 4 \
  --pos_weight 12.0 \
  --patience 0 \
  --overwrite