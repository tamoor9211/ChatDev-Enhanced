#!/usr/bin/env python3
"""
Test script to verify Google AI Studio integration works correctly.
"""

import os
import sys
from camel.typing import ModelType
from camel.model_backend import ModelFactory
from camel.configs import ChatGPTConfig

def test_google_ai_integration():
    """Test basic Google AI Studio integration."""
    
    # Check if API key is set
    api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: No API key found. Please set GOOGLE_API_KEY environment variable.")
        return False
    
    print("✅ API key found")
    
    try:
        # Test model creation
        model_type = ModelType.GEMINI_1_5_FLASH
        config = ChatGPTConfig()
        model_backend = ModelFactory.create(model_type, config.__dict__)
        print("✅ Model backend created successfully")
        
        # Test simple message
        test_messages = [
            {"role": "user", "content": "Hello! Please respond with 'Hello from Google AI!'"}
        ]
        
        print("🔄 Testing message generation...")
        response = model_backend.run(messages=test_messages)
        
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            print(f"✅ Response received: {content}")
            return True
        else:
            print("❌ Invalid response format")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def test_embedding():
    """Test embedding functionality with hybrid approach."""
    try:
        from ecl.embedding import OpenAIEmbedding

        embedding_model = OpenAIEmbedding()
        test_text = "This is a test sentence for embedding."

        print("🔄 Testing text embedding (hybrid: Google AI + local fallback)...")
        embedding = embedding_model.get_text_embedding(test_text)

        if embedding and len(embedding) > 0:
            print(f"✅ Text embedding generated (dimension: {len(embedding)})")
        else:
            print("❌ Failed to generate text embedding")
            return False

        print("🔄 Testing code embedding (hybrid: Google AI + local fallback)...")
        test_code = "def hello_world():\n    print('Hello, World!')"
        code_embedding = embedding_model.get_code_embedding(test_code)

        if code_embedding and len(code_embedding) > 0:
            print(f"✅ Code embedding generated (dimension: {len(code_embedding)})")
        else:
            print("❌ Failed to generate code embedding")
            return False

        # Test local embedding fallback specifically
        print("🔄 Testing local embedding fallback...")
        try:
            from ecl.local_embedding import LocalEmbedding
            local_model = LocalEmbedding()
            local_embedding = local_model.get_text_embedding("Test local embedding")

            if local_embedding and len(local_embedding) > 0:
                print(f"✅ Local embedding fallback works (dimension: {len(local_embedding)})")
            else:
                print("⚠️  Local embedding returned empty result")

        except Exception as e:
            print(f"⚠️  Local embedding test failed: {str(e)}")
            print("   This is expected if sentence-transformers is not installed")

        return True

    except Exception as e:
        print(f"❌ Error during embedding test: {str(e)}")
        return False

def test_token_counting():
    """Test token counting functionality."""
    try:
        from camel.utils import num_tokens_from_messages, get_model_token_limit
        
        test_messages = [
            {"role": "user", "content": "This is a test message for token counting."},
            {"role": "assistant", "content": "This is a response message."}
        ]
        
        model_type = ModelType.GEMINI_1_5_FLASH
        token_count = num_tokens_from_messages(test_messages, model_type)
        token_limit = get_model_token_limit(model_type)
        
        print(f"✅ Token counting works: {token_count} tokens")
        print(f"✅ Token limit: {token_limit} tokens")
        return True
        
    except Exception as e:
        print(f"❌ Error during token counting test: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Google AI Studio Integration")
    print("=" * 50)
    
    tests = [
        ("Basic Integration", test_google_ai_integration),
        ("Embedding System", test_embedding),
        ("Token Counting", test_token_counting),
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
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Google AI Studio integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the configuration and API key.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
