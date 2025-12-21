import os
import sys
from chatrel import config

print(f"OS: {os.name}")
print(f"HF_CONCURRENT_REQUESTS: {config.HF_CONCURRENT_REQUESTS}")
print(f"DEV_USE_RELOADER: {config.DEV_USE_RELOADER}")

if os.name == 'nt' and config.HF_CONCURRENT_REQUESTS > 1:
    if not config.DEV_USE_RELOADER:
        print("SUCCESS: Reloader disabled on Windows with high concurrency")
    else:
        print("FAILURE: Reloader enabled despite Windows + high concurrency")
else:
    print("Skipping Windows-specific check")
