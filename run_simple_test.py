"""
Run Simple Test

Quick test script to verify everything works.
"""

if __name__ == "__main__":
    try:
        from examples.simple_test import test_genome_creation, test_behavior_parameters
        
        print("=" * 50)
        print("EBAIF Framework Simple Test")
        print("=" * 50)
        
        test_genome_creation()
        print()
        test_behavior_parameters()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Framework is working.")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()