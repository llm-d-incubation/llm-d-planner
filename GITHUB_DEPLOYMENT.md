# GitHub Actions Deployment Guide - Demo Mode

This guide explains how to deploy **lightweight frontend demos** of NeuralNav using GitHub Actions.

## Overview - Demo Mode

This workflow is optimized for **frontend development work**. It deploys:

✅ **UI** (Streamlit frontend)
✅ **Backend API** (with static demo data)
❌ **PostgreSQL** (uses JSON files instead)
❌ **Ollama** (LLM features limited to demo data)

**Benefits:**
- 🚀 Fast deployments (~3 minutes)
- 💰 Low resource usage (no database, no LLM service)
- 🔗 Easy sharing - just send the URL
- 🧹 Auto-cleanup (manual or on-demand)

## Quick Start

### 1. Configure GitHub Secret

**Required:** Add `KUBE_CONFIG` secret:

1. **Get your kubeconfig:**
   ```bash
   oc login --token=<your-token> --server=<your-server>
   cat ~/.kube/config | base64 | tr -d '\n'
   ```
   Copy the entire output.

2. **Add to GitHub:**
   - Go to your repo: **Settings → Secrets and variables → Actions**
   - Click **New repository secret**
   - **Name**: `KUBE_CONFIG`
   - **Secret**: Paste the base64 string
   - Click **Add secret**

### 2. Deploy Your Branch

**Option A: Automatic (for demo/ or feature/ branches)**

```bash
# Push to a branch starting with demo/ or feature/
git checkout -b demo/new-ui-design
git push origin demo/new-ui-design

# Workflow runs automatically!
# Check: Actions tab → Deploy Demo workflow
```

**Option B: Manual (any branch)**

```bash
# Push your branch
git push origin your-branch-name

# Go to GitHub: Actions → Deploy Demo → Run workflow
# Select your branch
# Click "Run workflow"
```

### 3. Get Your URL

1. Go to **Actions** tab
2. Click on the running workflow
3. Click on the **Deploy to OpenShift** job
4. Scroll to bottom of logs
5. Copy the URL (looks like `https://llm-planner-demo-...apps.cluster.com`)
6. **Share with teammates!** 🎉

## What Gets Deployed

```
┌─────────────────────────────────────┐
│  Namespace: llm-planner-demo-       │
│             {your-branch-name}      │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │   Backend    │  │     UI      │ │
│  │   (API)      │  │ (Streamlit) │ │
│  │              │  │             │ │
│  │ • Demo data  │◄─┤ • Your UI   │ │
│  │ • JSON files │  │   changes   │ │
│  └──────────────┘  └─────────────┘ │
│                           │         │
│                           ▼         │
│                    ┌─────────────┐  │
│                    │   Route     │  │
│                    │  (Public)   │  │
│                    └─────────────┘  │
└─────────────────────────────────────┘
```

**Images pushed to:**
- `ghcr.io/daniel-warner-x/llm-d-planner-dw/backend:${branch}-${sha}`
- `ghcr.io/daniel-warner-x/llm-d-planner-dw/ui:${branch}-${sha}`

**Resource Usage:**
- Backend: 512Mi RAM, 0.25 CPU
- UI: 256Mi RAM, 0.1 CPU
- Total: ~768Mi RAM, 0.35 CPU (very light!)

## Workflow Triggers

The workflow runs when:

1. **Automatic:** Push to branches starting with `demo/` or `feature/`
2. **Manual:** Run workflow from Actions tab (any branch)
3. **Pull Requests:** (Optional - currently disabled)

## Common Workflows

### Scenario 1: Share UI Mockup with Designers

```bash
git checkout -b demo/new-layout
# Make your UI changes
git add ui/
git commit -s -m "feat(ui): redesign recommendation cards"
git push origin demo/new-layout

# Workflow runs automatically
# Go to Actions tab to get URL
# Share URL with design team
```

### Scenario 2: Manual Deploy from Any Branch

```bash
git checkout -b my-work
# Make changes
git push origin my-work

# Go to: Actions → Deploy Demo → Run workflow
# Select branch: my-work
# Click "Run workflow"
```

### Scenario 3: Test Before Merging

```bash
# Create PR
gh pr create --title "New UI feature"

# Deploy to test
# Go to Actions → Run workflow on your PR branch
# Test the live URL
# If good, merge
```

## Viewing Logs and Status

### View Workflow Run

1. **GitHub → Actions tab**
2. Click **Deploy Demo** workflow
3. Click on a specific run
4. See jobs:
   - **Build and Push Images** - Docker builds
   - **Deploy to OpenShift** - K8s deployment

### View Deployment Logs

```bash
# Set namespace (replace branch name)
NAMESPACE=llm-planner-demo-demo-new-layout

# Check pod status
oc get pods -n $NAMESPACE

# View UI logs
oc logs -n $NAMESPACE deployment/ui -f

# View Backend logs
oc logs -n $NAMESPACE deployment/backend -f

# Get route URL
oc get route -n $NAMESPACE
```

## Container Images

Images are published to **GitHub Container Registry** (ghcr.io):

- **Registry**: `ghcr.io`
- **Repository**: `daniel-warner-x/llm-d-planner-dw`
- **Images**: `backend`, `ui`
- **Tags**: `{branch}-{commit-sha}`, `{branch}-latest`

**View your images:**
- Go to repo → **Packages** (right sidebar)
- See all published versions
- Pull locally: `docker pull ghcr.io/daniel-warner-x/llm-d-planner-dw/ui:demo-new-layout-latest`

## Cleanup

### Manual Cleanup via OpenShift CLI

```bash
# List demo namespaces
oc get namespaces | grep llm-planner-demo

# Delete specific deployment
oc delete namespace llm-planner-demo-demo-new-layout

# Delete all demo deployments
oc get namespaces -l app=llm-planner-demo -o name | xargs oc delete
```

### Cleanup Old Images

GitHub Container Registry keeps all pushed images. To clean up:

1. Go to repo → **Packages**
2. Click on `backend` or `ui`
3. Click on specific versions
4. Click **Delete package version**

Or use CLI:
```bash
# Install GitHub CLI extension for packages
gh extension install actions/gh-actions-cache

# List old images (manual review needed)
# Delete via GitHub UI or API
```

## Troubleshooting

### Workflow Fails: "KUBE_CONFIG secret not set"

**Cause**: Missing GitHub secret
**Fix**: Add `KUBE_CONFIG` in Settings → Secrets → Actions

### Workflow Fails: "permission denied"

**Cause**: GitHub Actions doesn't have package write permissions
**Fix**: Should work automatically. Check Settings → Actions → General → Workflow permissions → "Read and write permissions"

### Deployment Succeeds but URL Shows 404

**Cause**: Route creation failed or pods not ready
**Fix**:
```bash
# Check pod status
oc get pods -n llm-planner-demo-{your-branch}

# Check route
oc get route -n llm-planner-demo-{your-branch}

# Check logs
oc logs -n llm-planner-demo-{your-branch} deployment/ui
```

### Backend API Errors in UI

**Cause**: Backend using JSON files instead of full database
**Expected**: Some features may be limited in demo mode
**Solution**: This is normal for demo deployments

### Workflow is Slow

**Cause**: Docker layer caching, OpenShift resource limits
**Fix**:
- First build takes ~5 minutes (subsequent builds ~2-3 min due to caching)
- Check OpenShift cluster status

### Image Pull Errors

**Cause**: GitHub Container Registry authentication issues
**Fix**:
```bash
# In OpenShift, create image pull secret (if needed)
oc create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<your-github-username> \
  --docker-password=<github-token> \
  -n llm-planner-demo-{branch}

# Link to serviceaccount
oc secrets link default ghcr-secret --for=pull -n llm-planner-demo-{branch}
```

Note: By default, GitHub packages from public repos should be pullable without auth.

## Customizing the Workflow

### Change Auto-Deploy Branches

Edit `.github/workflows/deploy-demo.yml`:

```yaml
on:
  push:
    branches:
      - 'staging/**'  # Auto-deploy staging/ branches
      - 'preview/**'  # Auto-deploy preview/ branches
```

### Add Environment Variables

Edit the ConfigMap in the workflow:

```yaml
data:
  NEURALNAV_BENCHMARK_SOURCE: "json"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  MY_CUSTOM_VAR: "value"  # Add custom vars here
```

### Increase Resources

Edit the Deployment resources:

```yaml
resources:
  requests:
    memory: "1Gi"   # Increase from 512Mi
    cpu: "500m"     # Increase from 250m
```

### Change Namespace Pattern

Edit the workflow:

```yaml
- name: Generate namespace name
  run: |
    NAMESPACE="my-app-${BRANCH_NAME}"  # Custom pattern
```

## Security Notes

✅ **Good practices:**
- `KUBE_CONFIG` stored as encrypted GitHub secret
- Images stored in GitHub Container Registry (private by default)
- Route uses TLS (edge termination)
- Namespaces isolated from each other

⚠️ **Important:**
- Demo deployments are publicly accessible (no authentication)
- Don't deploy sensitive data in demo mode
- Rotate your OpenShift token regularly
- Clean up old deployments

## GitHub vs GitLab Differences

| Feature | GitHub Actions | GitLab CI |
|---------|---------------|-----------|
| Config file | `.github/workflows/*.yml` | `.gitlab-ci.yml` |
| Secrets | Settings → Secrets | Settings → CI/CD Variables |
| Registry | `ghcr.io` | `registry.gitlab.com` |
| Trigger | Workflow dispatch | Manual job |
| Caching | Built-in layer cache | Needs configuration |

## Next Steps

1. ✅ Set up `KUBE_CONFIG` secret in GitHub
2. ✅ Push a `demo/` or `feature/` branch
3. ✅ Check Actions tab for workflow run
4. ✅ Share the URL with teammates!

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [OpenShift CLI](https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html)
- [Local Development](README.md#quick-start) - For full feature testing
