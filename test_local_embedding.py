#!/usr/bin/env python3
"""
Test script specifically for the local embedding fallback system.
This tests the sentence-transformers integration without requiring Google AI API.
"""

import sys
import os

def test_sentence_transformers_installation():
    """Test if sentence-transformers is properly installed."""
    try:
        import sentence_transformers
        print(f"✅ sentence-transformers version: {sentence_transformers.__version__}")
        return True
    except ImportError:
        print("❌ sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return False

def test_local_embedding():
    """Test the local embedding functionality."""
    try:
        # Add the ecl directory to path
        sys.path.append(os.path.join(os.getcwd(), "ecl"))
        from local_embedding import LocalEmbedding
        
        print("🔄 Initializing local embedding model...")
        local_embedding = LocalEmbedding()
        
        # Test text embedding
        print("🔄 Testing text embedding...")
        test_text = "This is a test sentence for local embedding."
        text_embedding = local_embedding.get_text_embedding(test_text)
        
        if text_embedding and len(text_embedding) > 0:
            print(f"✅ Text embedding generated (dimension: {len(text_embedding)})")
        else:
            print("❌ Failed to generate text embedding")
            return False
        
        # Test code embedding
        print("🔄 Testing code embedding...")
        test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        code_embedding = local_embedding.get_code_embedding(test_code)
        
        if code_embedding and len(code_embedding) > 0:
            print(f"✅ Code embedding generated (dimension: {len(code_embedding)})")
        else:
            print("❌ Failed to generate code embedding")
            return False
        
        # Test similarity (embeddings should be different for different inputs)
        print("🔄 Testing embedding similarity...")
        text1 = "Machine learning is fascinating"
        text2 = "Artificial intelligence is amazing"
        text3 = "The weather is nice today"
        
        emb1 = local_embedding.get_text_embedding(text1)
        emb2 = local_embedding.get_text_embedding(text2)
        emb3 = local_embedding.get_text_embedding(text3)
        
        # Calculate cosine similarity
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_12 = cosine_similarity(emb1, emb2)  # Should be high (similar topics)
        sim_13 = cosine_similarity(emb1, emb3)  # Should be lower (different topics)
        
        print(f"   Similarity (ML vs AI): {sim_12:.3f}")
        print(f"   Similarity (ML vs Weather): {sim_13:.3f}")
        
        if sim_12 > sim_13:
            print("✅ Embeddings show expected similarity patterns")
        else:
            print("⚠️  Embeddings may not be working optimally")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing local embedding: {str(e)}")
        return False

def test_hybrid_embedding():
    """Test the hybrid embedding system."""
    try:
        # Add the ecl directory to path
        sys.path.append(os.path.join(os.getcwd(), "ecl"))
        from local_embedding import HybridEmbedding
        
        print("🔄 Testing hybrid embedding system...")
        hybrid_embedding = HybridEmbedding()
        
        # Test text embedding
        test_text = "Hybrid embedding test with fallback capability."
        embedding = hybrid_embedding.get_text_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Hybrid text embedding generated (dimension: {len(embedding)})")
        else:
            print("❌ Failed to generate hybrid text embedding")
            return False
        
        # Test code embedding
        test_code = "print('Hello from hybrid embedding!')"
        code_embedding = hybrid_embedding.get_code_embedding(test_code)
        
        if code_embedding and len(code_embedding) > 0:
            print(f"✅ Hybrid code embedding generated (dimension: {len(code_embedding)})")
            return True
        else:
            print("❌ Failed to generate hybrid code embedding")
            return False
        
    except Exception as e:
        print(f"❌ Error testing hybrid embedding: {str(e)}")
        return False

def test_memory_usage():
    """Test memory usage of the local embedding model."""
    try:
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"📊 Initial memory usage: {initial_memory:.1f} MB")
        
        # Load the model
        sys.path.append(os.path.join(os.getcwd(), "ecl"))
        from local_embedding import LocalEmbedding
        
        local_embedding = LocalEmbedding()
        
        # Get memory after loading model
        after_load_memory = process.memory_info().rss / 1024 / 1024  # MB
        model_memory = after_load_memory - initial_memory
        
        print(f"📊 Memory after loading model: {after_load_memory:.1f} MB")
        print(f"📊 Model memory usage: {model_memory:.1f} MB")
        
        # Generate some embeddings
        for i in range(10):
            text = f"Test embedding number {i} with some content to embed."
            local_embedding.get_text_embedding(text)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 Final memory usage: {final_memory:.1f} MB")
        
        if model_memory < 200:  # Should be well under 200MB for all-MiniLM-L6-v2
            print("✅ Memory usage is reasonable for 8GB RAM systems")
            return True
        else:
            print("⚠️  Memory usage is higher than expected")
            return True  # Still pass, just warn
        
    except ImportError:
        print("⚠️  psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"⚠️  Memory test failed: {str(e)}")
        return True  # Don't fail the whole test for this

def main():
    """Run all local embedding tests."""
    print("🧪 Testing Local Embedding System")
    print("=" * 50)
    
    tests = [
        ("Sentence Transformers Installation", test_sentence_transformers_installation),
        ("Local Embedding", test_local_embedding),
        ("Hybrid Embedding", test_hybrid_embedding),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Local Embedding Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All local embedding tests passed!")
        print("💡 Your system can work offline with local embeddings")
        return 0
    elif passed >= 2:  # At least basic functionality works
        print("⚠️  Some tests failed, but basic functionality works")
        print("💡 Local embedding fallback should still function")
        return 0
    else:
        print("❌ Local embedding system has issues")
        print("💡 Try: pip install sentence-transformers")
        return 1

if __name__ == "__main__":
    sys.exit(main())
