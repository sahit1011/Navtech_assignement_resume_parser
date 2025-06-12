#!/usr/bin/env python3
"""
Test Resume Parser with Real Resumes and DeepSeek OpenRouter
Test the complete setup with actual resume files and your preferred LLM
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available")

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80)

def print_status(component, status, details=""):
    """Print component status"""
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {component}: {'SUCCESS' if status else 'FAILED'}")
    if details:
        print(f"   {details}")

def check_api_keys():
    """Check if API keys are loaded"""
    print_header("Checking API Keys")
    
    api_keys = {
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY')
    }
    
    for key_name, key_value in api_keys.items():
        if key_value and key_value != f"your_{key_name.lower()}_here":
            print_status(key_name, True, f"Found (ends with: ...{key_value[-10:]})")
        else:
            print_status(key_name, False, "Not found or placeholder")
    
    return api_keys

def find_test_resumes():
    """Find test resumes in docs folder"""
    print_header("Finding Test Resumes")
    
    test_dirs = [
        "docs/resumes_for_testing",
        "docs/testing",
        "docs",
        "sample_resumes"
    ]
    
    resume_files = []
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            for ext in ['*.pdf', '*.docx', '*.doc']:
                files = list(Path(test_dir).glob(ext))
                resume_files.extend(files)
    
    print(f"📄 Found {len(resume_files)} resume files:")
    for file in resume_files:
        print(f"   • {file}")
    
    return resume_files

def test_resume_parser_initialization():
    """Test ResumeParser initialization with API keys"""
    print_header("Testing ResumeParser with API Keys")
    
    try:
        from src.resume_parser import ResumeParser
        parser = ResumeParser()
        
        available_providers = parser.get_available_providers()
        print_status("ResumeParser Init", True, f"Available providers: {available_providers}")
        
        return parser, available_providers
    except Exception as e:
        print_status("ResumeParser Init", False, str(e))
        return None, []

def test_openrouter_specifically():
    """Test OpenRouter provider specifically"""
    print_header("Testing OpenRouter DeepSeek Model")
    
    try:
        from src.llm_providers.openrouter_llm import OpenRouterLLMProvider
        
        provider = OpenRouterLLMProvider()
        is_available = provider.is_available()
        
        print_status("OpenRouter Provider", is_available)
        
        if is_available:
            print(f"✅ OpenRouter API Key: Found")
            print(f"🤖 Model: {provider.config['model']}")
            print(f"🌡️ Temperature: {provider.config['temperature']}")
        
        return provider if is_available else None
        
    except Exception as e:
        print_status("OpenRouter Provider", False, str(e))
        return None

def test_resume_parsing(parser, resume_files, provider_name="openrouter"):
    """Test resume parsing with real files"""
    print_header(f"Testing Resume Parsing with {provider_name.upper()}")
    
    if not resume_files:
        print("❌ No resume files found to test")
        return
    
    # Test with first 2 resume files
    test_files = resume_files[:2]
    
    for i, resume_file in enumerate(test_files, 1):
        print(f"\n📄 Test {i}: {resume_file.name}")
        print("-" * 60)
        
        try:
            start_time = time.time()
            
            # Parse the resume
            result = parser.parse_resume(str(resume_file), provider_name)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Convert to dict for display
            result_dict = result.to_dict()
            
            # Display results
            print_status("Parsing", True, f"Completed in {processing_time:.2f} seconds")
            
            print(f"\n📋 Extracted Data:")
            print(f"   👤 Name: {result_dict.get('first_name', '')} {result_dict.get('last_name', '')}")
            print(f"   📧 Email: {result_dict.get('email', 'N/A')}")
            print(f"   📞 Phone: {result_dict.get('phone', 'N/A')}")
            print(f"   🏠 Location: {result_dict.get('address', {}).get('city', 'N/A')}, {result_dict.get('address', {}).get('state', 'N/A')}")
            print(f"   🎯 Skills: {len(result_dict.get('skills', []))} found")
            print(f"   🎓 Education: {len(result_dict.get('education_history', []))} entries")
            print(f"   💼 Work History: {len(result_dict.get('work_history', []))} entries")
            
            # Show first few skills
            skills = result_dict.get('skills', [])
            if skills:
                skill_names = [skill.get('name', '') for skill in skills[:5]]
                print(f"   🔧 Top Skills: {', '.join(skill_names)}")
            
            # Show summary preview
            summary = result_dict.get('summary', '')
            if summary:
                summary_preview = summary[:100] + "..." if len(summary) > 100 else summary
                print(f"   📝 Summary: {summary_preview}")
            
            print(f"\n✅ Successfully parsed {resume_file.name}")
            
        except Exception as e:
            print_status("Parsing", False, str(e))
            continue

def test_enhanced_models(parser, resume_files):
    """Test enhanced transformer models"""
    print_header("Testing Enhanced Transformer Models")
    
    if not resume_files:
        print("❌ No resume files found to test")
        return
    
    # Test enhanced models
    enhanced_providers = ['smart_transformer', 'layoutlm_transformer']
    test_file = resume_files[0]  # Use first resume
    
    for provider in enhanced_providers:
        print(f"\n🤖 Testing {provider}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            result = parser.parse_resume(str(test_file), provider)
            end_time = time.time()
            
            result_dict = result.to_dict()
            processing_time = end_time - start_time
            
            print_status(f"{provider}", True, f"Completed in {processing_time:.2f} seconds")
            print(f"   👤 Name: {result_dict.get('first_name', '')} {result_dict.get('last_name', '')}")
            print(f"   📧 Email: {result_dict.get('email', 'N/A')}")
            print(f"   🎯 Skills: {len(result_dict.get('skills', []))} found")
            
        except Exception as e:
            print_status(f"{provider}", False, str(e))

def main():
    """Main test function"""
    print("🧪 Testing NavTech Resume Parser with Real Data")
    print("=" * 80)
    print("🎯 Focus: DeepSeek OpenRouter + Real Resume Files")
    
    # Check API keys
    api_keys = check_api_keys()
    
    # Find test resumes
    resume_files = find_test_resumes()
    
    # Initialize parser
    parser, available_providers = test_resume_parser_initialization()
    
    if not parser:
        print("❌ Cannot proceed without working parser")
        return
    
    # Test OpenRouter specifically
    openrouter_provider = test_openrouter_specifically()
    
    # Test resume parsing with OpenRouter (DeepSeek)
    if 'openrouter' in available_providers and resume_files:
        test_resume_parsing(parser, resume_files, 'openrouter')
    else:
        print("⚠️ OpenRouter not available or no resume files found")
    
    # Test enhanced transformer models
    if resume_files:
        test_enhanced_models(parser, resume_files)
    
    # Summary
    print_header("Test Summary")
    print(f"📊 API Keys: {len([k for k in api_keys.values() if k and 'your_' not in k])}/3 configured")
    print(f"📄 Resume Files: {len(resume_files)} found")
    print(f"🤖 Available Providers: {len(available_providers)}")
    print(f"🎯 OpenRouter (DeepSeek): {'✅ Available' if 'openrouter' in available_providers else '❌ Not Available'}")
    
    if 'openrouter' in available_providers and resume_files:
        print("\n🎉 SUCCESS: Ready to parse resumes with DeepSeek OpenRouter!")
    else:
        print("\n⚠️ ISSUES: Check API keys or resume files")

if __name__ == "__main__":
    main()
