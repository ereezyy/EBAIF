"""
Quick Demo Runner for WebContainer

Runs the WebContainer-compatible demo showing EBAIF concepts.
"""

if __name__ == "__main__":
    try:
        print("🚀 Starting EBAIF WebContainer Demo...")
        print("Using standard library Python only (no external dependencies)")
        print()
        
        import sys
        sys.path.append('.')
        
        from examples.webcontainer_demo import run_webcontainer_demo
        import asyncio
        
        asyncio.run(run_webcontainer_demo())
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()