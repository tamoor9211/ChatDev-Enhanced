#!/usr/bin/env python3
"""
Setup verification script for ChatDev with Google AI Studio integration.
Run this script to verify your environment is properly configured.
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_api_key():
    """Check if Google AI Studio API key is set."""
    google_key = os.environ.get('GOOGLE_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')
    
    if google_key:
        print("✅ GOOGLE_API_KEY is set")
        return True
    elif openai_key:
        print("✅ OPENAI_API_KEY is set (will be used as GOOGLE_API_KEY)")
        return True
    else:
        print("❌ No API key found. Please set GOOGLE_API_KEY environment variable.")
        print("   Get your API key from: https://aistudio.google.com/")
        return False

def check_required_packages():
    """Check all required packages."""
    packages = [
        ("google-generativeai", "google.generativeai"),
        ("sentence-transformers", "sentence_transformers"),
        ("numpy", "numpy"),
        ("requests", "requests"),
        ("tenacity", "tenacity"),
        ("Flask", "flask"),
        ("beautifulsoup4", "bs4"),
        ("faiss-cpu", "faiss"),
        ("pyyaml", "yaml"),
        ("easydict", "easydict"),
    ]

    results = []
    for package_name, import_name in packages:
        result = check_package(package_name, import_name)
        results.append(result)

        # Special note for sentence-transformers
        if package_name == "sentence-transformers" and not result:
            print("   📝 Note: sentence-transformers provides local embedding fallback")
            print("   📝 Install with: pip install sentence-transformers")

    return all(results)

def check_chatdev_structure():
    """Check if ChatDev directory structure is correct."""
    required_dirs = [
        "camel",
        "chatdev", 
        "ecl",
        "CompanyConfig",
        "WareHouse"
    ]
    
    required_files = [
        "run.py",
        "requirements.txt",
        "camel/model_backend.py",
        "camel/typing.py",
        "ecl/embedding.py"
    ]
    
    print("\n📁 Checking directory structure...")
    
    all_good = True
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"✅ Directory: {directory}")
        else:
            print(f"❌ Missing directory: {directory}")
            all_good = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            all_good = False
    
    return all_good

def run_quick_test():
    """Run a quick test of the Google AI integration."""
    print("\n🧪 Running quick integration test...")
    try:
        # Import and test basic functionality
        from camel.typing import ModelType
        from camel.model_backend import ModelFactory
        from camel.configs import ChatGPTConfig
        
        # Test model creation
        model_type = ModelType.GEMINI_1_5_FLASH
        config = ChatGPTConfig()
        model_backend = ModelFactory.create(model_type, config.__dict__)
        
        print("✅ Model backend creation successful")
        print("✅ Google AI Studio integration appears to be working")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False

def main():
    """Main setup check function."""
    print("🔍 ChatDev Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("API Key", check_api_key),
        ("Required Packages", check_required_packages),
        ("Directory Structure", check_chatdev_structure),
        ("Integration Test", run_quick_test),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {check_name} check failed: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Setup Verification Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Your ChatDev setup is ready to use.")
        print("\n🚀 You can now run ChatDev with:")
        print('   python run.py --task "Create a simple calculator" --name "MyCalculator" --model GEMINI_1_5_FLASH')
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before using ChatDev.")
        print("\n💡 Common solutions:")
        print("   • Install missing packages: pip install -r requirements.txt")
        print("   • Set API key: export GOOGLE_API_KEY='your_api_key_here'")
        print("   • Make sure you're in the ChatDev root directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
