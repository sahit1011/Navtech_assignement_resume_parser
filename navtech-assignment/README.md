# 🚀 Resume Parser with AI/LLM Integration

## NavTech Assignment - AI/ML Engineer Role

A production-ready resume parser that extracts structured information from PDF, DOC, DOCX, and TXT files using state-of-the-art LLM providers and transformer models.

## 🎯 For Recruiters - Instant Testing

**🚀 Try it now in Google Colab (No setup required!):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahit1011/Navtech_assignement_resume_parser/blob/final/navtech-assignment/notebooks/NavTech_Resume_Parser_Updated.ipynb)

1. Click the badge above or open `notebooks/NavTech_Resume_Parser_Updated.ipynb`
2. Get a free API key from [OpenRouter.ai](https://openrouter.ai/keys) (2 minutes)
3. Add your API key and run all cells
4. Upload any resume file and get structured JSON output!

**✅ No installation, no environment setup, works in any browser!**

## ✨ Features

- **🎯 Easy Testing**: Google Colab notebook for instant testing
- **📄 Multiple File Formats**: PDF, DOC, DOCX, TXT support
- **🤖 Multiple LLM Providers**:
  - **OpenRouter** (DeepSeek R1) - Free & Recommended
  - **Google Gemini** - Free with quota limits
  - **OpenAI GPT** - Paid service
  - **Local Transformers** - Offline processing
- **📊 Structured JSON Output**: Production-ready format
- **🚫 No Fallback Data**: Real errors instead of empty responses
- **🔒 Secure**: API keys never committed to repository
- **🌐 Web Interface**: User-friendly Flask application

## Output Format

```json
{
    "first_name": "string",
    "last_name": "string", 
    "email": "string",
    "phone": "string",
    "address": {
        "city": "string",
        "state": "string", 
        "country": "string"
    },
    "summary": "string",
    "skills": [{"skill": "string"}],
    "education_history": [{
        "name": "string",
        "degree": "string",
        "from_date": "string",
        "to_date": "string"
    }],
    "work_history": [{
        "company": "string",
        "title": "string", 
        "description": "string",
        "from_date": "string",
        "to_date": "string"
    }]
}
```

## 🚀 Quick Start for Recruiters

### Option A: Google Colab Testing (Recommended - No Setup Required!)

**🎯 Instant Testing in Browser:**

1. **Open Notebook**: [NavTech_Resume_Parser_Updated.ipynb](./notebooks/NavTech_Resume_Parser_Updated.ipynb)
2. **Upload to Google Colab**: Click "Open in Colab" button
3. **Get Free API Key**: Visit [OpenRouter.ai](https://openrouter.ai/keys) (2 minutes)
4. **Add API Key**: Paste in the configuration cell
5. **Run All Cells**: Execute all cells in order
6. **Upload Resume**: Use the file upload widget
7. **Get Results**: Structured JSON output instantly!

**✅ No installation, no environment setup, works in any browser!**

### Option B: Local Setup

**⚠️ Important: Use the `final` branch (contains all the code):**

```bash
# 1. Clone and checkout the correct branch
git clone https://github.com/sahit1011/Navtech_assignement_resume_parser.git
cd Navtech_assignement_resume_parser
git checkout final
cd navtech-assignment

# 2. Setup environment
pip install -r requirements.txt
cp .env.example .env

# 3. Add your API key to .env file
# Get free key from: https://openrouter.ai/keys
# Edit .env and add: OPENROUTER_API_KEY=your_key_here

# 4. Run web application
python app.py

# 5. Open browser: http://localhost:8080
# 6. Upload resume and select "OpenRouter (DeepSeek R1)"
# 7. Get structured JSON output!
```

### Option C: Command Line Testing

```bash
# Test with sample resume
python src/main.py sample_resumes/sample_resume.txt --provider openrouter

# Test with your own resume
python src/main.py path/to/your/resume.pdf --provider openrouter
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### 🔑 API Key Setup

**⚠️ Important: API keys are NOT included in the repository for security**

1. **Copy the example file**: `cp .env.example .env`
2. **Get your own API keys**:
   - **OpenRouter** (Recommended - Free): https://openrouter.ai/keys
   - **Google Gemini** (Free with quota): https://makersuite.google.com/app/apikey
   - **OpenAI** (Paid): https://platform.openai.com/api-keys

3. **Add keys to .env file**:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```

**🎯 For Quick Testing**: Use Google Colab notebook - just add your API key and run!

### 🤖 LLM Provider Options (in order of recommendation)

1. **`openrouter`** (Recommended) - DeepSeek R1 model, free tier, highest accuracy
2. **`gemini`** - Google's model, free with quota, good accuracy
3. **`openai`** - GPT models, paid service, excellent accuracy
4. **`smart_transformer`** - Local BERT-based, works offline, fast
5. **`layoutlm_transformer`** - Local LayoutLM, works offline, fastest

## Usage

### Web Interface (Recommended)
```bash
python app.py
```
Visit http://localhost:8080 for the user-friendly interface.

### Command Line
```bash
# Using local transformer (no API key needed)
python src/main.py resume.pdf --provider smart_transformer

# Using OpenRouter API
python src/main.py resume.pdf --provider openrouter

# Using Gemini API
python src/main.py resume.pdf --provider gemini
```

### 📓 Jupyter Notebooks

**For Interactive Testing:**

1. **Google Colab** (Recommended):
   - Open `notebooks/NavTech_Resume_Parser_Updated.ipynb`
   - Upload to Google Colab
   - Add API key and run all cells
   - Upload resume files and get instant results

2. **Local Jupyter**:
   ```bash
   jupyter notebook notebooks/NavTech_Resume_Parser_Updated.ipynb
   ```

**Features:**
- ✅ Real API integration (no hardcoded responses)
- ✅ File upload widget for easy testing
- ✅ Sample resume included for quick demo
- ✅ Step-by-step instructions for recruiters
- ✅ Works in Google Colab without any setup

## 🚨 Troubleshooting

### Common Issues

**❌ "No code found after cloning"**
- ✅ **Solution**: Use `git checkout final` - the main branch is empty!

**❌ "API key not available" or "401 Unauthorized"**
- ✅ **Solution**: Get your own API key from [OpenRouter.ai](https://openrouter.ai/keys) (free)
- ✅ **Alternative**: Use Google Colab notebook for easier setup

**❌ "Failed to parse JSON response"**
- ✅ **Solution**: API quota exceeded - get a fresh API key
- ✅ **Alternative**: Try a different provider (gemini, openai)

**❌ "No LLM providers available"**
- ✅ **Solution**: Add API keys to `.env` file or use Colab notebook

### 🎯 Quick Solutions

1. **🚀 Easiest**: Use Google Colab notebook (no setup required)
2. **🔑 Local Setup**: Get free OpenRouter API key (2 minutes)
3. **🆓 Offline**: Use local transformers (no API needed)

### ⚡ Performance Comparison

- **Fastest**: `layoutlm_transformer` (~3 seconds, offline)
- **Most Accurate**: `openrouter` with DeepSeek R1 (~15 seconds, free)
- **Best Balance**: `smart_transformer` (~5 seconds, offline)
- **Highest Quality**: `openai` GPT-4 (~10 seconds, paid)

## Web Interface

Start the web application:
```bash
python app.py
```

Visit:
- http://localhost:8080/ - Main interface
- http://localhost:8080/status - API status
- http://localhost:8080/providers - Provider status
- http://localhost:8080/demo - Demo with sample resume

## Project Structure

```
navtech-assignment/
├── config/           # Configuration files
├── src/             # Source code
│   ├── file_processors/  # File format handlers
│   ├── llm_providers/    # LLM integrations
│   ├── extractors/       # Information extractors
│   └── utils/           # Utility functions
├── notebooks/       # Jupyter notebooks
├── sample_resumes/  # Test files
└── tests/          # Unit tests
```

## License

This project is created for the NavTech assignment and is for educational purposes.
