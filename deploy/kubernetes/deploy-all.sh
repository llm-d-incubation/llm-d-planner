#!/bin/bash
set -e

echo "Deploying NeuralNav..."

oc apply -f deploy/kubernetes/namespace.yaml \
         -f deploy/kubernetes/secrets.yaml \
         -f deploy/kubernetes/postgres.yaml \
         -f deploy/kubernetes/ollama.yaml \
         -f deploy/kubernetes/backend.yaml \
         -f deploy/kubernetes/ui.yaml \
         -f deploy/kubernetes/route.yaml

echo "Waiting for PostgreSQL to be ready..."
oc wait --for=condition=ready pod -l app.kubernetes.io/name=postgres -n neuralnav --timeout=120s

echo "Running database initialization job..."
# Delete previous job if it exists (jobs are immutable)
oc delete job db-init -n neuralnav --ignore-not-found
oc apply -f deploy/kubernetes/db-init-job.yaml

echo "Waiting for db-init job to complete..."
oc wait --for=condition=complete job/db-init -n neuralnav --timeout=300s

echo "Database initialized. Deployment complete."
