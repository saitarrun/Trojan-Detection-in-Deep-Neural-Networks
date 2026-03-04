#!/bin/bash

# Configuration
POD_NAME="trojan-detection-dev"
LOCAL_PORT=2222
SSH_KEY_PATH="$HOME/.ssh/id_ed25519.pub"

# Check if pod is running
while [[ $(kubectl get pod $POD_NAME -o 'jsonpath={..status.phase}') != "Running" ]]; do
  echo "Waiting for pod $POD_NAME to be running..."
  sleep 5
done

# Add SSH key to the pod
echo "Adding local public key to remote pod..."
kubectl exec $POD_NAME -- mkdir -p /root/.ssh
kubectl cp "$SSH_KEY_PATH" "$POD_NAME":/root/.ssh/authorized_keys
kubectl exec $POD_NAME -- chmod 700 /root/.ssh
kubectl exec $POD_NAME -- chmod 600 /root/.ssh/authorized_keys

# Start port forwarding
echo "Starting port-forwarding on port $LOCAL_PORT..."
echo "To terminate, press Ctrl+C"
kubectl port-forward $POD_NAME $LOCAL_PORT:22
