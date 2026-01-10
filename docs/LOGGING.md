# NeuralNav Logging Guide

This document describes the logging features in NeuralNav and how to use them for debugging and monitoring.

## Overview

NeuralNav implements comprehensive logging at every stage of the recommendation workflow:

1. **User Request Logging** - Captures every user message
2. **LLM Interaction Logging** - Logs prompts sent to the LLM and responses received
3. **Intent Extraction Logging** - Tracks the conversion of natural language to structured intent
4. **Workflow Logging** - Monitors each step of the recommendation generation process

## Log Levels

### INFO (Default)
Standard operational logging that includes:
- User requests and messages
- Workflow step progress
- LLM request/response metadata (length, model used)
- Extracted intent summaries
- Recommendation results
- Deployment operations

### DEBUG (Verbose)
Detailed logging for troubleshooting that adds:
- **Full LLM prompts** (complete text sent to the model)
- **Full LLM responses** (complete responses from the model)
- **Complete extracted intent** (full JSON structure)
- Request/response timing details

## Enabling Debug Mode

### Option 1: Environment Variable (Recommended)
Set the `COMPASS_DEBUG` environment variable before starting the backend:

```bash
# Enable debug logging
export COMPASS_DEBUG=true
make start-backend

# Or inline:
COMPASS_DEBUG=true make start-backend
```

### Option 2: Script Modification
Edit `scripts/run_api.sh` and add:

```bash
export COMPASS_DEBUG=true
```

## Log Output Locations

### Console Output
All logs are written to stdout/stderr and captured by uvicorn.

### Log Files
Logs are also written to:
- `logs/backend.log` - Main application logs (when using make commands)
- `logs/compass.log` - Structured logs with full details

## Log Format

Each log entry follows this format:

```
YYYY-MM-DD HH:MM:SS - module.name - LEVEL - message
```

Example:
```
2025-10-16 16:45:12 - src.api.routes - INFO - [USER MESSAGE] I need a chatbot for 1000 users
```

## Understanding the Logs

### User Request Flow

When a user makes a request, you'll see logs like:

```
================================================================================
[USER REQUEST] New recommendation request
[USER MESSAGE] I need a chatbot for 1000 users with low latency
[CONVERSATION HISTORY] 0 previous messages
================================================================================
```

### LLM Interaction

For each LLM call, you'll see:

**INFO level:**
```
[LLM REQUEST] Role: user, Content length: 450 chars
[LLM RESPONSE] Model: llama3.1:8b, Response length: 220 chars
```

**DEBUG level (includes full content):**
```
[LLM PROMPT] You are an expert assistant helping users deploy...
[LLM RESPONSE CONTENT] {"use_case": "chatbot", "user_count": 1000, ...}
```

### Intent Extraction

```
[INTENT EXTRACTION] Sending prompt to LLM for intent extraction
[EXTRACTED INTENT] {'use_case': 'chatbot', 'user_count': 1000, ...}
```

### Workflow Steps

```
Step 1: Extracting deployment intent
Intent extracted: chatbot, 1000 users, high latency
Step 2: Generating traffic profile and SLO targets
Traffic profile: 50 QPS
SLO targets: TTFT=200ms, TPOT=50ms
Step 3: Recommending models
Found 3 model candidates
Step 4: Planning GPU capacity
```

## Searching and Filtering Logs

### Find all user requests
```bash
grep "\[USER MESSAGE\]" logs/backend.log
```

### Find all LLM prompts (DEBUG mode only)
```bash
grep "\[LLM PROMPT\]" logs/backend.log
```

### Find all extracted intents
```bash
grep "\[EXTRACTED INTENT\]" logs/backend.log
```

### View logs for a specific request session
```bash
# Find the request start, then tail from there
grep -A 100 "\[USER MESSAGE\] your message here" logs/backend.log
```

## Log Rotation

To prevent log files from growing indefinitely, consider setting up log rotation:

### Using logrotate (Linux)

Create `/etc/logrotate.d/compass`:

```
/path/to/compass/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 user group
}
```

### Manual Rotation
```bash
# Archive old logs
mv logs/backend.log logs/backend.log.$(date +%Y%m%d)
gzip logs/backend.log.$(date +%Y%m%d)

# Restart to create new log file
make restart
```

## Troubleshooting Tips

### Not seeing DEBUG logs?
1. Check `COMPASS_DEBUG` is set: `echo $COMPASS_DEBUG`
2. Restart the backend after setting the variable
3. Verify log level in startup message: `NeuralNav API starting with log level: DEBUG`

### Logs too verbose?
1. Unset `COMPASS_DEBUG`: `unset COMPASS_DEBUG`
2. Restart backend: `make restart`

### Need to see full prompts temporarily?
```bash
# Enable debug for one request
COMPASS_DEBUG=true make restart
# Make your request
# Then disable
unset COMPASS_DEBUG && make restart
```

## Privacy and Security Considerations

⚠️ **Important**: DEBUG mode logs contain:
- Full user messages and conversation history
- Complete LLM prompts (which may include user data)
- Full LLM responses

**Best Practices:**
- Only enable DEBUG mode in development/testing environments
- Never commit log files to version control (already in .gitignore)
- Rotate and purge logs regularly in production
- Consider implementing PII scrubbing for production logging
- Review logs before sharing externally

## Performance Impact

- **INFO level**: Minimal impact (~1-2% overhead)
- **DEBUG level**: Moderate impact (~5-10% overhead due to string formatting and I/O)

For production deployments, use INFO level unless troubleshooting.

## Example: Full Request Flow

Here's what you'll see for a complete request with DEBUG enabled:

```
================================================================================
[USER REQUEST] New recommendation request
[USER MESSAGE] I need a chatbot for customer support, 2000 users
================================================================================
[INTENT EXTRACTION] Sending prompt to LLM for intent extraction
[LLM PROMPT] You are an expert assistant helping users deploy...
[LLM REQUEST] Role: user, Content length: 512 chars
[LLM RESPONSE] Model: llama3.1:8b, Response length: 185 chars
[LLM RESPONSE CONTENT] {"use_case": "customer_service", "user_count": 2000...}
[EXTRACTED INTENT] {'use_case': 'customer_service', 'user_count': 2000, ...}
Extracted intent: use_case=customer_service, users=2000
Step 2: Generating traffic profile and SLO targets
Traffic profile: 75 QPS
Step 3: Recommending models
Found 3 model candidates
Step 4: Planning GPU capacity
Planning capacity for Mistral-7B-Instruct-v0.2 (score: 85.0)
✓ Recommendation generated successfully
```

## Related Documentation

- [Developer Guide](DEVELOPER_GUIDE.md) - General development workflows
- [Architecture](ARCHITECTURE.md) - System architecture overview
- [Testing Guide](../ui/TESTING.md) - Testing workflows
