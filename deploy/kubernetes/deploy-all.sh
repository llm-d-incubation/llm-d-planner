#!/bin/bash
set -e

echo "Deploying Planner..."

# Apply base infrastructure (everything except backend, which needs
# service-ca and NetworkPolicy to be ready first)
oc apply -f deploy/kubernetes/namespace.yaml \
         -f deploy/kubernetes/secrets.yaml \
         -f deploy/kubernetes/configmap.yaml \
         -f deploy/kubernetes/service-ca-configmap.yaml \
         -f deploy/kubernetes/gpu-reader-rbac.yaml \
         -f deploy/kubernetes/postgres.yaml \
         -f deploy/kubernetes/ollama.yaml \
         -f deploy/kubernetes/ui.yaml \
         -f deploy/kubernetes/route.yaml

# Cross-namespace NetworkPolicy (allows planner backend -> Model Catalog)
BENCHMARK_SOURCE=$(oc get configmap planner-config -n planner -o jsonpath='{.data.PLANNER_BENCHMARK_SOURCE}') || {
  echo "Warning: Failed to read planner-config configmap, skipping Model Catalog network policy"
  BENCHMARK_SOURCE=""
}
if [ "$BENCHMARK_SOURCE" = "model_catalog" ]; then
  echo "Applying Model Catalog network policy..."
  oc apply -f deploy/kubernetes/networkpolicy-model-catalog.yaml

  echo "Waiting for service-ca certificate injection..."
  for i in $(seq 1 30); do
    if oc get configmap planner-service-ca -n planner -o jsonpath='{.data.service-ca\.crt}' 2>/dev/null | grep -q "BEGIN CERTIFICATE"; then
      echo "Service CA certificate is ready."
      break
    fi
    if [ "$i" -eq 30 ]; then
      echo "Error: Timed out waiting for service-ca certificate injection" >&2
      exit 1
    fi
    sleep 2
  done
else
  echo "Skipping Model Catalog network policy (benchmark source: ${BENCHMARK_SOURCE:-postgresql})"
  oc delete -f deploy/kubernetes/networkpolicy-model-catalog.yaml --ignore-not-found
fi

# Apply backend after prerequisites are ready
echo "Deploying backend..."
oc apply -f deploy/kubernetes/backend.yaml

echo "Waiting for PostgreSQL to be ready..."
oc wait --for=condition=ready pod -l app.kubernetes.io/name=postgres -n planner --timeout=120s

echo "Running database initialization job..."
# Delete previous job if it exists (jobs are immutable)
oc delete job db-init -n planner --ignore-not-found
oc apply -f deploy/kubernetes/db-init-job.yaml

echo "Waiting for db-init job to complete..."
oc wait --for=condition=complete job/db-init -n planner --timeout=300s

echo "Database initialized. Deployment complete."
