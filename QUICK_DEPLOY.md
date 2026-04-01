# Quick Deploy Cheat Sheet - GitHub Actions

## One-Time Setup (5 minutes)

```bash
# 1. Get your kubeconfig
oc login --token=<your-token> --server=<your-server>
cat ~/.kube/config | base64 | tr -d '\n'

# 2. Add to GitHub
# Go to: Settings → Secrets and variables → Actions
# Click: New repository secret
# Name: KUBE_CONFIG
# Secret: <paste the base64 string>
# Click: Add secret
```

## Deploy Your Branch

### Option A: Automatic (Recommended)

```bash
# Use demo/ or feature/ branch prefix for auto-deploy
git checkout -b demo/my-feature
git push origin demo/my-feature

# ✨ Workflow runs automatically!
# Check: Actions tab → Deploy Demo
```

### Option B: Manual

```bash
# Push any branch
git push origin your-branch

# Go to: Actions → Deploy Demo → Run workflow
# Select: your-branch
# Click: Run workflow
```

## Get Your URL

```
1. GitHub → Actions tab
2. Click: Latest "Deploy Demo" run
3. Click: "Deploy to OpenShift" job
4. Scroll down → Copy URL
5. Share! 🎉
```

## Quick Commands

```bash
# Set your namespace (replace with your branch name)
NS=llm-planner-demo-demo-my-feature

# View your demo URL
oc get route -n $NS

# Check if it's running
oc get pods -n $NS

# View UI logs
oc logs -n $NS deployment/ui -f

# View backend logs
oc logs -n $NS deployment/backend -f

# Port-forward for debugging
oc port-forward -n $NS svc/ui 8501:8501

# Manual cleanup
oc delete namespace $NS
```

## URLs You'll Need

- **GitHub Repo**: https://github.com/Daniel-Warner-X/llm-d-planner-DW
- **Actions**: https://github.com/Daniel-Warner-X/llm-d-planner-DW/actions
- **Packages**: https://github.com/Daniel-Warner-X?tab=packages

## What Gets Deployed

- ✅ Your UI changes (Streamlit app)
- ✅ Backend API (with demo data)
- ❌ No PostgreSQL (uses JSON files)
- ❌ No Ollama (limited AI features)

**Resources**: ~768MB RAM, 0.35 CPU (very light!)

## Auto-Deploy Triggers

Workflow runs automatically when you push to:
- `demo/*` branches (e.g., `demo/new-design`)
- `feature/*` branches (e.g., `feature/button-fix`)

Or manually via Actions tab (any branch).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "KUBE_CONFIG not set" | Add secret in Settings → Secrets → Actions |
| Build fails | Check Actions logs, retry workflow |
| 404 on URL | Check pod status: `oc get pods -n $NS` |
| Demo is slow | First build ~5min, then cached ~2-3min |
| Can't find URL | Check workflow logs (Deploy step, bottom) |

## Tips

💡 **Branch naming**: Use `demo/` or `feature/` prefix for auto-deploy

💡 **Testing**: Deploy → Test → Fix → Push → Auto-redeploys

💡 **Sharing**: URLs are public - share with anyone on the network

💡 **Cleanup**: Delete namespace when done, or let it run

💡 **Images**: View in Packages tab, tagged by branch + commit

💡 **Local dev**: For full features, use `make start` locally

## Quick Workflow Example

```bash
# Day 1: New feature
git checkout -b demo/redesign
# ... make UI changes ...
git push origin demo/redesign
# ✨ Auto-deploys! Copy URL from Actions

# Day 2: Updates
# ... more changes ...
git push origin demo/redesign
# ✨ Auto-redeploys with new changes

# Day 3: Done
git checkout main
git merge demo/redesign
git push origin main
oc delete namespace llm-planner-demo-demo-redesign
```

## Security Note

🔒 **Your OpenShift token is public in this chat!**

After setup, regenerate it:
1. OpenShift Console → Username → Copy login command
2. Display Token → Get new token
3. Use new token for future logins

## Full Documentation

- [GITHUB_DEPLOYMENT.md](GITHUB_DEPLOYMENT.md) - Complete guide
- [README.md](README.md) - Project overview and local development
- [.github/workflows/deploy-demo.yml](.github/workflows/deploy-demo.yml) - Workflow source
