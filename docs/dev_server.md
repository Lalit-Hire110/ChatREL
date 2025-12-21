# Windows Dev Server Stability

## Issue
On Windows, running the Flask development server with the default Werkzeug auto-reloader enabled can cause `WinError 10038` crashes during heavy concurrent processing (e.g., large batch inference with HuggingFace).

This happens because the reloader monitors file changes and sometimes triggers a reload when the `concurrent.futures` or `threading` modules are heavily used, closing the server socket while threads are still active.

## Solution
We have disabled the auto-reloader by default on Windows when `HF_CONCURRENT_REQUESTS > 1`.

## Configuration
You can control this behavior using the `CHATREL_DEV_RELOAD` environment variable in your `.env` file:

```bash
# Force reloader ON (may crash on large chats)
CHATREL_DEV_RELOAD=True

# Force reloader OFF (stable)
CHATREL_DEV_RELOAD=False
```

If not set, the system defaults to:
- **Windows**: `False` (if concurrency > 1)
- **Linux/Mac**: `True`

## Recommendation
For development involving large chat uploads on Windows, keep the default behavior (reloader disabled) or explicitly set `CHATREL_DEV_RELOAD=False`.
