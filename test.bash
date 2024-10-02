#!/bin/bash

python main_train_test.py \
                        --model 'tfcnn' \
                        --load_checkpoint logs/tfcnn/tf-cnn-clean-ds-shuffled/run_20240929083956/checkpoints/epoch=39-val_loss=1.47.ckpt