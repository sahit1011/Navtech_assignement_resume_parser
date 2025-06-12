#!/usr/bin/env python3
"""
Test script to simulate frontend API calls to the Flask application
"""

import requests
import json
import os
from pathlib import Path

def test_openrouter_api():
    """Test OpenRouter API through the Flask application"""
    
    # Flask app URL
    base_url = "http://localhost:8080"
    
    # Test file
    sample_file = Path("sample_resumes/sample_resume.txt")
    
    if not sample_file.exists():
        print("❌ Sample resume file not found")
        return False
    
    print("🧪 Testing OpenRouter DeepSeek API through Flask frontend...")
    print(f"📁 Using file: {sample_file}")
    
    try:
        # Prepare the request data (simulating frontend form submission)
        files = {
            'resume_file': ('sample_resume.txt', open(sample_file, 'rb'), 'text/plain')
        }
        
        data = {
            'llm_provider': 'openrouter',
            'custom_api_key': ''  # Use the API key from .env
        }
        
        print("🚀 Sending request to Flask API...")
        
        # Make the API request
        response = requests.post(
            f"{base_url}/api/parse",
            files=files,
            data=data,
            timeout=120
        )
        
        print(f"📊 Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! OpenRouter API working through frontend")
            print("\n📋 Parsed Data Summary:")
            
            if result.get('success'):
                data = result.get('data', {})
                print(f"   • Name: {data.get('first_name', '')} {data.get('last_name', '')}")
                print(f"   • Email: {data.get('email', '')}")
                print(f"   • Phone: {data.get('phone', '')}")
                print(f"   • Skills: {len(data.get('skills', []))} found")
                print(f"   • Work History: {len(data.get('work_history', []))} entries")
                print(f"   • Education: {len(data.get('education_history', []))} entries")
                
                return True
            else:
                print("❌ API returned success=False")
                print(f"Error: {result}")
                return False
        else:
            print(f"❌ FAILED! HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask app is not running on localhost:8080")
        print("💡 Start the Flask app with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Close the file
        if 'files' in locals():
            files['resume_file'][1].close()

def test_app_status():
    """Test if the Flask app is running"""
    try:
        response = requests.get("http://localhost:8080/status", timeout=10)
        if response.status_code == 200:
            print("✅ Flask app is running")
            return True
        else:
            print(f"⚠️ Flask app responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Flask app is not running")
        return False
    except Exception as e:
        print(f"❌ Error checking app status: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Frontend API Integration...")
    print("=" * 50)
    
    # First check if Flask app is running
    if test_app_status():
        # Test the OpenRouter API
        success = test_openrouter_api()
        
        if success:
            print("\n🎉 Frontend API test PASSED!")
        else:
            print("\n💥 Frontend API test FAILED!")
    else:
        print("\n💡 Please start the Flask app first: python app.py")
