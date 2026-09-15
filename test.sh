CUDA_VISIBLE_DEVICES="0" \
python test.py \
--hiera_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/2.MODIFICATION_SAM2-UNet/SAM2-UNet/sam2_hiera_large.pt" \
--checkpoint "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/6.TRAINING RESULT/2.MODIFY_CODE/IMAGE_DTM/0%_MODIFICATION3/best_by_val_iou.pth" \
--test_image_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/5.TESTING/IMAGE/" \
--test_dtm_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/5.TESTING/DTM_NORM/" \
--test_gt_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/5.TESTING/LABEL/" \
--save_path "/content/drive/MyDrive/MULTI MODAL SEGMENTATION/End-To-End/ALIGNED_512_16BIT/7.TESTING RESULT/2.MODIFY_CODE/IMAGE_DTM/0%_MODIFICATION3/" \
--threshold 0.8 \
--min_component_area 20