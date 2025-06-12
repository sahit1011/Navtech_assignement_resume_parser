#!/usr/bin/env python3
"""
Flask Web Application for Resume Parser
Provides a user-friendly interface for recruiters to test resume parsing
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.resume_parser import ResumeParser
from config.llm_config import AVAILABLE_PROVIDERS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'navtech-resume-parser-secret-key'

# Configuration
UPLOAD_FOLDER = project_root / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html', providers=AVAILABLE_PROVIDERS)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and parsing"""
    try:
        # Check if file was uploaded
        if 'resume_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['resume_file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        # Get form data
        llm_provider = request.form.get('llm_provider', 'smart_transformer')
        custom_api_key = request.form.get('custom_api_key', '').strip()
        
        # Validate file
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PDF, DOC, DOCX, or TXT files.', 'error')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(file_path))
        
        # Set custom API key if provided
        if custom_api_key:
            if llm_provider == 'gemini':
                os.environ['GEMINI_API_KEY'] = custom_api_key
            elif llm_provider == 'openai':
                os.environ['OPENAI_API_KEY'] = custom_api_key
            elif llm_provider == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = custom_api_key
        
        # Parse resume
        logger.info(f"Parsing resume: {filename} with provider: {llm_provider}")

        parser = ResumeParser()

        try:
            result = parser.parse_resume(str(file_path), llm_provider)

            # Convert to dict for JSON response
            result_dict = result.to_dict()

            # Clean up uploaded file
            file_path.unlink(missing_ok=True)

            return render_template('result.html',
                                 result=result_dict,
                                 filename=filename,
                                 provider=llm_provider,
                                 json_output=json.dumps(result_dict, indent=2))

        except ValueError as e:
            # Clean up uploaded file
            file_path.unlink(missing_ok=True)

            # Handle API key and provider-specific errors with better messages
            error_msg = str(e)
            error_lower = error_msg.lower()

            if "quota" in error_lower or "rate limit" in error_lower:
                flash(f'⚠️ API Quota Exceeded: {error_msg}', 'warning')
                flash('💡 Try: 1) Wait 24 hours for reset, 2) Get OpenRouter free key, 3) Use transformer provider', 'info')
            elif "api key" in error_lower and "invalid" in error_lower:
                if llm_provider == 'gemini':
                    flash(f'🔑 Gemini API Issue: {error_msg}', 'error')
                    flash('💡 This might be a quota issue. Try OpenRouter free tier instead.', 'info')
                else:
                    flash(f'🔑 API Key Error: {error_msg}', 'error')
            elif "not available" in error_lower:
                flash(f'🚫 Provider Error: {error_msg}', 'error')
                flash('💡 Try using the "transformer" provider (works offline) or get an API key.', 'info')
            else:
                flash(f'❌ Parsing Error: {error_msg}', 'error')

            return redirect(url_for('index'))

    except Exception as e:
        # Clean up uploaded file
        file_path.unlink(missing_ok=True)

        logger.error(f"Error processing file: {e}")
        flash(f'Unexpected error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/parse', methods=['POST'])
def api_parse():
    """API endpoint for programmatic access"""
    try:
        # Check if file was uploaded
        if 'resume_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get parameters
        llm_provider = request.form.get('llm_provider', 'smart_transformer')
        custom_api_key = request.form.get('custom_api_key', '').strip()
        
        # Validate file
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Set custom API key if provided
            if custom_api_key:
                if llm_provider == 'gemini':
                    os.environ['GEMINI_API_KEY'] = custom_api_key
                elif llm_provider == 'openai':
                    os.environ['OPENAI_API_KEY'] = custom_api_key
                elif llm_provider == 'openrouter':
                    os.environ['OPENROUTER_API_KEY'] = custom_api_key
            
            # Parse resume
            parser = ResumeParser()
            result = parser.parse_resume(tmp_path, llm_provider)
            
            return jsonify({
                'success': True,
                'data': result.to_dict(),
                'provider': llm_provider,
                'filename': file.filename
            })
        
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo():
    """Demo page with sample resume"""
    return render_template('demo.html', providers=AVAILABLE_PROVIDERS)

@app.route('/demo/parse', methods=['POST'])
def demo_parse():
    """Parse the sample resume"""
    try:
        llm_provider = request.form.get('llm_provider', 'smart_transformer')
        custom_api_key = request.form.get('custom_api_key', '').strip()
        
        # Use sample resume
        sample_file = project_root / 'sample_resumes' / 'sample_resume.txt'
        
        if not sample_file.exists():
            flash('Sample resume not found', 'error')
            return redirect(url_for('demo'))
        
        # Set custom API key if provided
        if custom_api_key:
            if llm_provider == 'gemini':
                os.environ['GEMINI_API_KEY'] = custom_api_key
            elif llm_provider == 'openai':
                os.environ['OPENAI_API_KEY'] = custom_api_key
            elif llm_provider == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = custom_api_key
        
        # Parse resume
        logger.info(f"Parsing sample resume with provider: {llm_provider}")
        
        parser = ResumeParser()
        result = parser.parse_resume(str(sample_file), llm_provider)
        
        # Convert to dict for JSON response
        result_dict = result.to_dict()
        
        return render_template('result.html', 
                             result=result_dict, 
                             filename='sample_resume.txt',
                             provider=llm_provider,
                             json_output=json.dumps(result_dict, indent=2))
    
    except Exception as e:
        logger.error(f"Error processing demo: {e}")
        flash(f'Error processing demo: {str(e)}', 'error')
        return redirect(url_for('demo'))

@app.route('/providers')
def providers():
    """Show available providers and their status"""
    provider_status = []
    
    for provider_info in AVAILABLE_PROVIDERS:
        provider_name = provider_info['name']
        
        try:
            if provider_name == 'gemini':
                from src.llm_providers.gemini_llm import GeminiLLMProvider
                provider = GeminiLLMProvider()
            elif provider_name == 'openai':
                from src.llm_providers.openai_llm import OpenAILLMProvider
                provider = OpenAILLMProvider()
            elif provider_name == 'openrouter':
                from src.llm_providers.openrouter_llm import OpenRouterLLMProvider
                provider = OpenRouterLLMProvider()

            elif provider_name == 'smart_transformer':
                from src.llm_providers.enhanced_transformer_llm import EnhancedTransformerProvider
                provider = EnhancedTransformerProvider()
            elif provider_name == 'layoutlm_transformer':
                from src.llm_providers.improved_layoutlm_transformer import ImprovedLayoutLMProvider
                provider = ImprovedLayoutLMProvider()
            else:
                continue
            
            is_available = provider.is_available()
            model_info = provider.get_model_info()
            
            provider_status.append({
                'name': provider_name,
                'display_name': provider_info['display_name'],
                'description': provider_info['description'],
                'requires_api_key': provider_info['requires_api_key'],
                'available': is_available,
                'model_info': model_info
            })
        
        except Exception as e:
            provider_status.append({
                'name': provider_name,
                'display_name': provider_info['display_name'],
                'description': provider_info['description'],
                'requires_api_key': provider_info['requires_api_key'],
                'available': False,
                'error': str(e)
            })
    
    return render_template('providers.html', providers=provider_status)

@app.route('/status')
def status():
    """Show detailed API status and troubleshooting"""
    from dotenv import load_dotenv
    load_dotenv()

    status_info = {
        'gemini': {
            'api_key_present': bool(os.getenv('GEMINI_API_KEY')) and os.getenv('GEMINI_API_KEY') != 'your_gemini_api_key_here',
            'api_key_preview': os.getenv('GEMINI_API_KEY', '')[:10] + '...' if os.getenv('GEMINI_API_KEY') else 'Not set',
            'status': 'Quota Exceeded (appears as invalid key)'
        },
        'openai': {
            'api_key_present': bool(os.getenv('OPENAI_API_KEY')) and os.getenv('OPENAI_API_KEY') != 'your_openai_api_key_here',
            'api_key_preview': os.getenv('OPENAI_API_KEY', '')[:10] + '...' if os.getenv('OPENAI_API_KEY') else 'Not set',
            'status': 'Needs valid API key'
        },
        'openrouter': {
            'api_key_present': bool(os.getenv('OPENROUTER_API_KEY')) and os.getenv('OPENROUTER_API_KEY') != 'your_openrouter_api_key_here',
            'api_key_preview': os.getenv('OPENROUTER_API_KEY', '')[:10] + '...' if os.getenv('OPENROUTER_API_KEY') else 'Not set',
            'status': 'Get free API key'
        }
    }

    return jsonify({
        'status': status_info,
        'recommendations': {
            'immediate': 'Get OpenRouter free API key from https://openrouter.ai/',
            'alternative': 'Use transformer provider (offline, lower accuracy)',
            'long_term': 'Wait 24 hours for Gemini quota reset or upgrade to paid plan'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting NavTech Resume Parser Web Application")
    print("=" * 50)
    print("📋 Available endpoints:")
    print("   • http://localhost:8080/ - Main upload interface")
    print("   • http://localhost:8080/demo - Demo with sample resume")
    print("   • http://localhost:8080/providers - Provider status")
    print("   • http://localhost:8080/api/parse - API endpoint")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=8080)
