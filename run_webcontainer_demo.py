"""
Quick Demo Runner for WebContainer

Runs the WebContainer-compatible demo showing EBAIF concepts.
"""

import asyncio
import sys

if __name__ == "__main__":
    try:
        print("🚀 Starting EBAIF WebContainer Demo...")
        print("Using standard library Python only (no external dependencies)")
        print()
        
        sys.path.append('.')
        
        from examples.webcontainer_demo import run_webcontainer_demo
        
        # Run demo using asyncio
        asyncio.run(run_webcontainer_demo())
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
