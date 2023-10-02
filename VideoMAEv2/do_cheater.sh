#!/bin/bash

python train_cheater_pred.py \
        --model vit_base_patch16_224 \
        --data_set cheater \
        --nb_classes 1 \
        --data_path /home/waldo/code/models/utils/cheater_data/ \
		--data_root /home/waldo/code/models/utils/cheater_data/ \
        --finetune ./killshot2/checkpoint-999.pth \
        --log_dir cheat_rican2 \
        --output_dir cheat_rican2 \
        --batch_size 8 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 200 \
        --num_frames 16 \
        --sampling_rate 1 \
        --num_sample 1 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 40 \
        --epochs 1000 \
        --test_num_segment 5 \
        --test_num_crop 3 $1
		
		
	