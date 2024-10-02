
# # Define arguments
# PREFIX='tsformer-clean-ds-shuffled-rm_spatial'
# NUM_EPOCHS=300
# BATCH_SIZE=1
# LR=0.001
# MODEL='tsformer'
# DEPTH=2
# HEADS=3
# DIM_HEAD=64
# MLP_DIM=64

# # Construct log file name based on arguments
# LOG_FILE="${PREFIX}_epochs${NUM_EPOCHS}_bs${BATCH_SIZE}_lr${LR}_model${MODEL}_depth${DEPTH}_heads${HEADS}_dimhead${DIM_HEAD}_mlpdim${MLP_DIM}.log"

# # Run the command with nohup and redirect output to the constructed log file
# nohup \
# python main_train_test.py \
#                         --train \
#                         --prefix "$PREFIX" \
#                         --num_epochs "$NUM_EPOCHS" \
#                         --batch_size "$BATCH_SIZE" \
#                         --lr "$LR" \
#                         --model "$MODEL" \
#                         --depth "$DEPTH" \
#                         --heads "$HEADS" \
#                         --dim_head "$DIM_HEAD" \
#                         --mlp_dim "$MLP_DIM" \
#                         >> "$LOG_FILE" 2>&1 &

# Define arguments
# PREFIX='cnn-clean-ds-shuffled_weighted'
# NUM_EPOCHS=300
# BATCH_SIZE=1
# LR=0.0001
# MODEL='cnn'
# NOTES='a'

# # Construct log file name based on arguments
# LOG_FILE="${PREFIX}_epochs${NUM_EPOCHS}_bs${BATCH_SIZE}_lr${LR}_model${MODEL}_${NOTES}.log"

# nohup \
# python main_train_test.py \
#                         --train \
#                         --prefix "$PREFIX" \
#                         --num_epochs "$NUM_EPOCHS" \
#                         --batch_size "$BATCH_SIZE" \
#                         --lr "$LR" \
#                         --model "$MODEL" \
#                         >> "$LOG_FILE" 2>&1 &


# Define arguments
PREFIX='tf-cnn-clean-ds-shuffled'
NUM_EPOCHS=300
BATCH_SIZE=1
LR=0.0001
MODEL='tfcnn'
DEPTH=1
HEADS=3
DIM_HEAD=64
MLP_DIM=64
NOTE='diff_weight_2_resume_2'
# Construct log file name based on arguments
LOG_FILE="${PREFIX}_epochs${NUM_EPOCHS}_bs${BATCH_SIZE}_lr${LR}_model${MODEL}_depth${DEPTH}_heads${HEADS}_dimhead${DIM_HEAD}_mlpdim${MLP_DIM}_${NOTE}.log"

# Run the command with nohup and redirect output to the constructed log file
nohup \
python main_train_test.py \
                        --train \
                        --prefix "$PREFIX" \
                        --num_epochs "$NUM_EPOCHS" \
                        --batch_size "$BATCH_SIZE" \
                        --lr "$LR" \
                        --model "$MODEL" \
                        --depth "$DEPTH" \
                        --heads "$HEADS" \
                        --dim_head "$DIM_HEAD" \
                        --mlp_dim "$MLP_DIM" \
                        --load_checkpoint logs/tfcnn/tf-cnn-clean-ds-shuffled/run_20240929083956/checkpoints/epoch=77-val_loss=1.48.ckpt \
                        >> "$LOG_FILE" 2>&1 &