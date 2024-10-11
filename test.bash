#!/bin/bash

python main_train_test.py \
                        --model 'tfcnn' \
                        --load_checkpoint logs/tfcnn/tf-cnn-clean-ds-shuffled/Best_model/checkpoints/epoch=36-val_loss=1.48.ckpt