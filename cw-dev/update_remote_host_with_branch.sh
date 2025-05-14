#!/bin/bash
POD_NAME=$1
REMOTE_USER=root
REMOTE_HOST=localhost
PORT=8889
LOCAL_PATH=/Users/ssteel/dev/vllm-fork/vllm
REMOTE_PATH=/vllm-workspace

POD_READY=0


while [ $POD_READY -ne "1" ]; do
  recent_output=$(kubectl logs $POD_NAME | tail -n 10)
  echo "Recent logs: $recent_output"
  if echo "$recent_output" | grep -q "Starting SSH server..."; then
    ((POD_READY++))
  else
    echo "Pod still starting up. Sleeping 10 seconds."
    sleep 10
  fi
done

echo "Killing existing port-forwards..."
ps aux | grep "kubectl port-forward" | grep ":$PORT" | awk '{print $2}' | xargs kill -9
sleep 3

kubectl port-forward pod/$POD_NAME $PORT:22 &

# Wait until the port is open
while ! nc -z localhost $PORT; do
  echo "Waiting for port $PORT to be ready..."
  sleep 0.5
done

rclone copy "$LOCAL_PATH" :sftp:"$REMOTE_PATH" \
  --sftp-host=localhost \
  --sftp-port="$PORT" \
  --sftp-user="$REMOTE_USER" \
  --filter="+ /.buildkite/**" \
  --filter="- /.*/**" \
  --filter="- /docs/**" \
  --progress \
  --no-traverse \
  --verbose

# Finally, install vLLM on the dev container with the remote filesystem synced-up.
kubectl exec -it "$POD_NAME" -- /bin/bash -c "\
  export VLLM_TARGET_DEVICE=cuda; \
  export VLLM_USE_PRECOMPILED=1; \
  pip install -r requirements/build.txt && \
  pip install --no-build-isolation --editable ."