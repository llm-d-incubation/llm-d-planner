#!/bin/bash

# NeuralNav - Kubernetes Cluster Management Script
#
# This script manages the KIND cluster lifecycle for NeuralNav.
# It handles cluster creation, KServe installation, and cluster teardown.

set -e  # Exit on error

# Configuration
CLUSTER_NAME="neuralnav"
KSERVE_VERSION="v0.13.0"
CERT_MANAGER_VERSION="v1.14.4"
CLUSTER_CONFIG="config/kind-cluster.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    printf "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}\n"
    printf "${BLUE}║  $1${NC}\n"
    printf "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    printf "${GREEN}▶ $1${NC}\n"
}

print_error() {
    printf "${RED}✗ ERROR: $1${NC}\n"
}

print_warning() {
    printf "${YELLOW}⚠ WARNING: $1${NC}\n"
}

print_success() {
    printf "${GREEN}✓ $1${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."

    local missing_deps=()

    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    elif ! docker info &> /dev/null; then
        print_error "Docker is installed but not running. Please start Docker Desktop."
        exit 1
    fi

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        missing_deps+=("kubectl")
    fi

    # Check kind
    if ! command -v kind &> /dev/null; then
        missing_deps+=("kind")
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        echo ""
        echo "Install missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  brew install $dep"
        done
        exit 1
    fi

    print_success "All prerequisites satisfied"
}

# Check if cluster exists
cluster_exists() {
    kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"
}

# Start cluster
start_cluster() {
    print_header "Starting NeuralNav Kubernetes Cluster"

    check_prerequisites

    if cluster_exists; then
        print_warning "Cluster '${CLUSTER_NAME}' already exists!"
        echo "Use 'scripts/kind-cluster.sh restart' to recreate, or 'scripts/kind-cluster.sh status' to check status"
        exit 1
    fi

    # Check for cluster config
    if [ ! -f "$CLUSTER_CONFIG" ]; then
        print_error "Cluster config file not found: $CLUSTER_CONFIG"
        exit 1
    fi

    # Create KIND cluster
    print_step "Creating KIND cluster with GPU node labels..."
    kind create cluster --config "$CLUSTER_CONFIG"
    print_success "KIND cluster created"

    # Wait a moment for cluster to stabilize
    sleep 5

    # Install cert-manager
    print_step "Installing cert-manager ${CERT_MANAGER_VERSION}..."
    kubectl apply -f "https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"

    print_step "Waiting for cert-manager to be ready (this may take 2-3 minutes)..."
    kubectl wait --for=condition=available --timeout=300s \
        -n cert-manager \
        deployment/cert-manager \
        deployment/cert-manager-webhook \
        deployment/cert-manager-cainjector
    print_success "cert-manager ready"

    # Install KServe
    print_step "Installing KServe ${KSERVE_VERSION}..."
    kubectl apply -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"

    print_step "Waiting for KServe controller to be ready (this may take 2-3 minutes)..."
    kubectl wait --for=condition=available --timeout=300s \
        -n kserve \
        deployment/kserve-controller-manager
    print_success "KServe controller ready"

    # Wait for webhook to be fully functional before applying cluster resources
    print_step "Waiting for KServe webhook to be ready..."
    sleep 10  # Give webhook server time to register
    kubectl wait --for=condition=ready --timeout=60s \
        -n kserve \
        pod -l control-plane=kserve-controller-manager 2>/dev/null || true
    print_success "KServe webhook ready"

    # Now apply cluster resources (requires webhook to be functional)
    print_step "Installing KServe cluster resources..."
    kubectl apply -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-cluster-resources.yaml"
    print_success "KServe cluster resources installed"

    # Configure KServe for RawDeployment mode
    print_step "Configuring KServe for RawDeployment mode..."
    kubectl patch configmap/inferenceservice-config -n kserve --type=strategic \
        -p '{"data": {"deploy": "{\"defaultDeploymentMode\": \"RawDeployment\"}"}}'
    print_success "KServe configured"

    # Load vLLM simulator image into cluster
    print_step "Loading vLLM simulator image into cluster..."
    if docker images vllm-simulator:latest --format "{{.Repository}}" | grep -q vllm-simulator; then
        kind load docker-image vllm-simulator:latest --name "$CLUSTER_NAME"
        print_success "vLLM simulator image loaded"
    else
        print_warning "vLLM simulator image not found locally - skipping"
        echo "  Build the simulator: cd simulator && docker build -t vllm-simulator:latest ."
    fi

    echo ""
    print_header "Cluster Ready!"
    echo ""
    print_success "Cluster Name: ${CLUSTER_NAME}"
    print_success "KServe Version: ${KSERVE_VERSION}"
    print_success "cert-manager Version: ${CERT_MANAGER_VERSION}"
    echo ""
}

# Stop/delete cluster
stop_cluster() {
    print_header "Stopping NeuralNav Kubernetes Cluster"

    if ! cluster_exists; then
        print_warning "Cluster '${CLUSTER_NAME}' does not exist"
        exit 0
    fi

    print_step "Deleting KIND cluster '${CLUSTER_NAME}'..."
    kind delete cluster --name "$CLUSTER_NAME"
    print_success "Cluster deleted"
    echo ""
    echo "Cluster '${CLUSTER_NAME}' has been removed."
    echo "Run 'scripts/kind-cluster.sh start' to create a new cluster."
    echo ""
}

# Restart cluster (delete and recreate)
restart_cluster() {
    print_header "Restarting NeuralNav Kubernetes Cluster"

    if cluster_exists; then
        print_step "Deleting existing cluster..."
        kind delete cluster --name "$CLUSTER_NAME"
        print_success "Old cluster deleted"
        echo ""
    fi

    # Wait a moment before recreating
    sleep 2

    # Start fresh cluster
    start_cluster
}

# Show cluster status
show_status() {
    print_header "NeuralNav Kubernetes Cluster Status"

    echo ""
    echo "Cluster Name: ${CLUSTER_NAME}"
    echo ""

    if ! cluster_exists; then
        print_error "Cluster does not exist"
        echo ""
        echo "Run 'scripts/kind-cluster.sh start' to create the cluster."
        exit 1
    fi

    print_success "Cluster exists"
    echo ""

    # Check cluster connectivity
    if kubectl cluster-info &> /dev/null; then
        print_success "kubectl can connect to cluster"
    else
        print_error "kubectl cannot connect to cluster"
        echo ""
        echo "Try: kubectl config use-context kind-${CLUSTER_NAME}"
        exit 1
    fi

    echo ""
    print_step "Cluster Nodes:"
    kubectl get nodes

    echo ""
    print_step "KServe Status:"
    if kubectl get namespace kserve &> /dev/null; then
        kubectl get pods -n kserve
        echo ""
        echo "KServe ClusterServingRuntimes:"
        kubectl get clusterservingruntimes 2>/dev/null | head -5
    else
        print_warning "KServe not installed"
    fi

    echo ""
    print_step "cert-manager Status:"
    if kubectl get namespace cert-manager &> /dev/null; then
        kubectl get pods -n cert-manager
    else
        print_warning "cert-manager not installed"
    fi

    echo ""
    print_step "InferenceServices in default namespace:"
    local isvc_count=$(kubectl get inferenceservices -n default --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [ "$isvc_count" -eq 0 ]; then
        echo "  (none)"
    else
        kubectl get inferenceservices -n default
    fi

    echo ""
    print_step "Cluster Context:"
    kubectl config current-context

    echo ""
    print_header "Cluster is Ready"
    echo ""
}

# Show usage
show_usage() {
    cat << EOF
NeuralNav - Cluster Management

Usage: scripts/kind-cluster.sh <command>

Commands:
  start     Create and configure a new KIND cluster with KServe
  stop      Delete the existing KIND cluster
  restart   Delete and recreate the cluster (fresh start)
  status    Show cluster status and components
  help      Show this help message

Examples:
  scripts/kind-cluster.sh start      # Create new cluster
  scripts/kind-cluster.sh status     # Check cluster status
  scripts/kind-cluster.sh restart    # Reset cluster to fresh state
  scripts/kind-cluster.sh stop       # Remove cluster

Configuration:
  Cluster Name: ${CLUSTER_NAME}
  KServe Version: ${KSERVE_VERSION}
  cert-manager Version: ${CERT_MANAGER_VERSION}
  Config File: ${CLUSTER_CONFIG}

Prerequisites:
  - Docker Desktop (running)
  - kubectl
  - kind

EOF
}

# Main script logic
case "${1:-}" in
    start)
        start_cluster
        ;;
    stop)
        stop_cluster
        ;;
    restart)
        restart_cluster
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: ${1:-}"
        echo ""
        show_usage
        exit 1
        ;;
esac
