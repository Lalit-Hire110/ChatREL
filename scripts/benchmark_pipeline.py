"""
Benchmark script for ChatREL pipeline.
Measures execution time for full analysis of a chat file.
"""

import time
import sys
import logging
from pathlib import Path
from chatrel.pipeline import run_full_analysis

# Configure logging to show timing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_benchmark(filepath: str):
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Starting benchmark on {path}...")
    start_time = time.time()
    
    try:
        # Run analysis (ensure cache is OFF for benchmark to measure raw speed, 
        # or ON if we want to measure warm cache - usually we want cold or warm?
        # The user wants to optimize "large chat uploads", which implies cold start or partial cache.
        # But for "speed up message processing", the HF calls are the bottleneck.
        # Let's run with use_cache=True but assume it's cold for the first run if we clear it,
        # or just measure the processing part. 
        # Actually, to measure improvement in HF calls, we should probably mock or use real API.
        # Real API is slow and costs money/quota. 
        # The user said "Keep results deterministic for tests (maintain mock mode)".
        # But for the benchmark they showed logs of "HFClient initialized...".
        # Let's use the real pipeline but maybe limit messages if it's too huge for a quick test,
        # or just run it. The user said "3500+ msgs".
        
        result = run_full_analysis(path, use_cache=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nBenchmark completed successfully!")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Messages: {len(result['df_processed'])}")
        print(f"Speed: {len(result['df_processed']) / duration:.1f} msgs/sec")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to the large file if available, else sample
        large_file = Path("sample_data/WhatsApp Chat with shrav.txt")
        if large_file.exists():
            file_path = str(large_file)
        else:
            file_path = "sample_data/sample_chat.txt"
            
    run_benchmark(file_path)
