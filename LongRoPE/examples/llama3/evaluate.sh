#!/bin/bash

# Evaluate Perplexity on Proof-Pile test dataset and Passkey on Llama-3-8B model with LongRoPE rescale factors.

export CUDA_VISIBLE_DEVICES=0,2
# export CUDA_LAUNCH_BLOCKING=1

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export TARGET_LENGTH=$((32 * 1024))
MODEL_PATH=/data/model/Meta-Llama-3-8B
DATASETS_PATH=$(pwd)/datasets

export ROPE_METHOD=longrope
# export ROPE_METHOD=none
export LONGROPE_RESCALE_FACTOR=$(pwd)/results/search/llama3-8b/$TARGET_LENGTH/result_final.csv
export LONGROPE_SCALING_POLICY=su

export OUTPUT_DIR=$(pwd)/results/eval/llama3-8b/$TARGET_LENGTH
mkdir -p $OUTPUT_DIR

# For finetuned model, LongRoPE rescale factor is compatible for shorter input lengths.

LENGTH_LIST=""
TMP_LENGTH=$((48 * 1024))
while (($TMP_LENGTH >= 4096))
do
    LENGTH_LIST="$TMP_LENGTH,$LENGTH_LIST"
    TMP_LENGTH=$((TMP_LENGTH / 2))
done
TARGET_LENGTH_LIST=${LENGTH_LIST::-1}
TARGET_LENGTH_LIST='49152' # 49152,65536
# python evaluation/perplexity.py \
#     --model $MODEL_PATH \
#     --tokenized $DATASETS_PATH/proof-pile-test-llama-tokenized \
#     --num-tokens $TARGET_LENGTH_LIST \
#     --dataset-min-tokens 131072 \
#     --samples 10 \
#     --truncate \
#     --attn-implementation flash_attention_2 \
#     --output-file $OUTPUT_DIR/proof-pile.csv

python evaluation/passkey.py \
    --model $MODEL_PATH \
    --num-tokens $TARGET_LENGTH_LIST \
    --samples 10 \
    --attn-implementation flash_attention_2 \
    --output-file $OUTPUT_DIR/passkey.csv \
    --log-file $OUTPUT_DIR/passkey.log
