#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
export MODEL_PATH=$HOME/USERS/Shubham/MRL/models/Qwen3-0.6B-Base
export TRAIN_DATA=$HOME/USERS/Shubham/MRL/data/kalm-sample-1152k
C=32; N=100
for cfg in base:0:0 blk:1.0:0.1; do
  for m in rawMRL SpecMRL; do
    IFS=: read tag bc ol <<< "$cfg"
    name=${m}-${tag}-${N}
    echo "[$(date +%T)] TRAIN $name"
    BLOCK_COEF=$bc ORTH_LAMBDA=$ol NPROC_PER_NODE=1 PER_DEVICE_BS=384 \
      bash train_custom.sh $m --max_steps $N --save_steps $N --gc_chunk_size $C \
      --output_dir ./trained_models/$name > logs/train_$name.txt 2>&1 \
      || { echo "[$(date +%T)] TRAIN FAILED $name"; continue; }
    flag=""; [ "$m" = "rawMRL" ] && flag="--mrl"
    echo "[$(date +%T)] EVAL $name"
    python run4mteb_custom.py --model_name $MODEL_PATH \
      --trained_model_path ./trained_models/$name $flag --subset15 \
      --pool mean --batch_size 256 --dims 1024 512 256 128 64 \
      --output_dir ./output/$name > logs/eval_$name.txt 2>&1 \
      || echo "[$(date +%T)] EVAL FAILED $name"
  done
done
echo "[$(date +%T)] ALL DONE"
