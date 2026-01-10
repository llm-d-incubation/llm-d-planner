# Sprint 3 Testing Guide - Streamlit UI

This guide covers testing the Streamlit UI for NeuralNav.

## Prerequisites

1. **Ollama running** with llama3.1:8b model
2. **FastAPI backend running** on http://localhost:8000
3. **Streamlit and dependencies installed** in virtual environment

## Setup

```bash
# Activate virtual environment
cd backend
source venv/bin/activate

# Install UI dependencies (if not already installed)
pip install streamlit requests
```

## Running the UI

### Option 1: Using the startup script (Recommended)

```bash
# From project root
scripts/run_ui.sh
```

### Option 2: Manual start

```bash
# From project root
cd backend
source venv/bin/activate
streamlit run ../ui/app.py
```

**Note**: On first run, Streamlit may prompt for an email address. Just press Enter to skip.

The UI will start on http://localhost:8501

## Testing Checklist

### 1. UI Loads Successfully

- [ ] Streamlit app starts without errors
- [ ] Header displays "NeuralNav" with app icon
- [ ] Sidebar shows app title and navigation
- [ ] Two-column layout visible (Conversation | Recommendation)
- [ ] Chat input field is present

### 2. Backend Connectivity

- [ ] No connection errors in UI
- [ ] Health check passes (check backend logs)

### 3. Example Prompts

Test each example prompt button in the sidebar:

- [ ] **Example 1**: "Customer service chatbot for 5000 users, low latency critical"
  - Should recommend: Mistral 7B on A100-80GB
  - Cost: ~$3,285/month
  - SLO Status: ✅ MEETS SLO

- [ ] **Example 2**: "Code generation assistant for 500 developers, quality over speed"
  - Should recommend: Mistral 7B on A10G
  - Cost: ~$730/month
  - SLO Status: ✅ MEETS SLO

- [ ] **Example 3**: "Document summarization pipeline, high throughput, cost efficient"
  - Should recommend: Granite 8B on L4
  - Cost: ~$365/month
  - SLO Status: ✅ MEETS SLO

### 4. Chat Interface

- [ ] User messages appear in chat
- [ ] Assistant responses appear with recommendation summary
- [ ] Spinner shows during API call
- [ ] Messages persist in chat history

### 5. Recommendation Display

Test all tabs when a recommendation is displayed:

#### Overview Tab
- [ ] SLO status badge displays correctly (✅ MEETS SLO or ⚠️ DOES NOT MEET SLO)
- [ ] Model name and ID shown
- [ ] GPU configuration displayed (count, type, tensor parallel, replicas)
- [ ] Key metrics cards show: TTFT, TPOT, E2E, Throughput
- [ ] Reasoning section displays

#### Specifications Tab
- [ ] Use case and requirements shown
- [ ] Traffic profile values displayed
- [ ] SLO targets displayed
- [ ] "✏️ Enable Editing" button works
- [ ] Fields become editable when editing mode enabled
- [ ] "💾 Save Changes" and "❌ Cancel" buttons work in edit mode

#### Performance Tab
- [ ] TTFT metrics shown (p50, p90, p99)
- [ ] TPOT metrics shown (p50, p90, p99)
- [ ] E2E latency metrics shown (p50, p90, p99)
- [ ] Throughput metrics shown (QPS, tokens/sec)
- [ ] Delta values vs targets displayed

#### Cost Tab
- [ ] Hourly cost displayed
- [ ] Monthly cost displayed
- [ ] GPU configuration details shown
- [ ] Cost assumptions info box visible
- [ ] "Generate Deployment YAML" button shows (Sprint 4 placeholder)
- [ ] "Deploy to Kubernetes" button shows (Sprint 6 placeholder)

### 6. Navigation & UX

- [ ] "🔄 New Conversation" button clears state and reloads
- [ ] Switching between tabs works smoothly
- [ ] Custom CSS styling applied (professional theme, metric cards)
- [ ] Responsive layout works on different screen sizes

### 7. Error Handling

Test error scenarios:

- [ ] Backend not running → Shows connection error message
- [ ] Invalid prompt → Shows appropriate error
- [ ] API timeout → Shows timeout error

## Custom Test Scenarios

Try these additional prompts to test the system:

1. **High-volume scenario**:
   - "I need to serve 10,000 concurrent users with a recommendation engine. Latency is very important - users expect results in under 300ms."
   - Should recommend higher-end GPU configuration

2. **Cost-sensitive scenario**:
   - "Small team of 50 developers need a coding assistant. Budget is very limited, quality can be moderate."
   - Should recommend cost-effective configuration (L4 GPUs)

3. **Quality-focused scenario**:
   - "Building a medical diagnosis assistant for 200 doctors. Accuracy is critical, budget is flexible, latency can be 1-2 seconds."
   - Should recommend larger model with better quality

## Known Limitations (Sprint 3)

- [ ] Editing specifications doesn't trigger re-recommendation (Sprint 4)
- [ ] YAML generation not implemented (Sprint 4)
- [ ] Kubernetes deployment not implemented (Sprint 6)
- [ ] Conversation history not persisted across sessions
- [ ] No authentication/user management

## Success Criteria

Sprint 3 is successful if:

1. ✅ UI loads without errors
2. ✅ All 3 example scenarios work end-to-end
3. ✅ Recommendation details display correctly in all tabs
4. ✅ Edit mode toggles properly
5. ✅ Chat interface is intuitive and functional
6. ✅ No connection errors when backend is running

## Troubleshooting

### Streamlit won't start

```bash
# Check virtual environment
which python  # Should point to backend/venv

# Reinstall streamlit
pip install --upgrade streamlit
```

### Backend connection fails

```bash
# Check FastAPI is running
curl http://localhost:8000/health

# Should return: {"status":"healthy","service":"ai-pre-deployment-assistant"}
```

### Recommendation request fails

1. Check backend logs for errors
2. Verify Ollama is running: `curl http://localhost:11434/api/tags`
3. Test backend directly:
   ```bash
   curl -X POST http://localhost:8000/api/recommend \
     -H "Content-Type: application/json" \
     -d '{"message": "test chatbot for 100 users"}'
   ```

### UI shows old cached data

```bash
# Clear Streamlit cache
streamlit cache clear
```
