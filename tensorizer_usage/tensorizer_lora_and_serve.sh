#!/bin/bash

# Needs:
# S3_ACCESS_KEY_ID
# S3_SECRET_ACCESS_KEY
# HF_TOKEN

export MODEL_REF=meta-llama/Llama-2-7b-hf
export LORA_REF=yard1/llama-2-7b-sql-lora-test
export SUFFIX=v1
export BUCKET_PREFIX=s3://tensorized-ssteel/lora-tests

export BUCKET_DIR=$BUCKET_PREFIX/vllm/$MODEL_REF/$SUFFIX

cat > model_loader_config.json <<EOF
{
  "tensorizer_uri": "$BUCKET_PREFIX/model.tensors",
  "lora_dir": "$BUCKET_PREFIX"
}
EOF

# First, tensorize a model and lora adapter
python3 ../examples/other/tensorize_vllm_model.py \
   --model "$MODEL_REF" \
   --lora-path "$LORA_REF" \
   serialize \
   --serialized-directory "$BUCKET_PREFIX" \
   --suffix "$SUFFIX"

echo "Now trying to serve it..."
sleep 10

LOGFILE="lora_tensorizer_serve.log"

MODEL_LOADER_EXTRA_CONFIG=$(jq -c -n \
  --arg tensorizer_uri "$BUCKET_DIR/model.tensors" \
  --arg lora_dir "$BUCKET_DIR" \
  '{tensorizer_uri: $tensorizer_uri, lora_dir: $lora_dir}')

# The `-c` ensures it's compact (no line breaks)

vllm serve "$MODEL_REF" \
  --load-format tensorizer \
  --enable-lora \
  --model-loader-extra-config "$MODEL_LOADER_EXTRA_CONFIG" \
  > "$LOGFILE" 2>&1 &


VLLM_PID=$!

retries=0
finished=0

while [ "$finished" -ne 1 ]; do
  if tail -n 10 "$LOGFILE" | grep -q "Application startup complete"; then
    finished=1
  else
    tail -n 10 "$LOGFILE"
    echo "===== Checking for completion again in 10 seconds.. ====="
    sleep 10
    retries=$((retries + 1))
  fi
done

echo "Success."

kill "$VLLM_PID"
exit 0