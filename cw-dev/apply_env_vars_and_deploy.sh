#!/bin/bash
FILE=$1
set -a
source vars.env
set +a

applied=$(envsubst < $FILE | kubectl apply -f -)

if echo "$applied" | grep -q "unchanged"; then
    echo "Unchanged, so won't wait."
else
    echo "Waiting 1 minute to let the old pod spin down..."
    sleep 60
fi


POD_NAME=$(k get pods | grep $APP_NAME | cut -d ' ' -f 1)

echo "Pod name is $POD_NAME"
./update_remote_host_with_branch.sh $POD_NAME