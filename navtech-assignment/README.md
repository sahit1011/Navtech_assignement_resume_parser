# Resume Parser with Transformer Models

## NavTech Assignment - AI/ML Engineer Role

A comprehensive resume parser that extracts structured information from PDF, DOC, and DOCX files using various LLM providers and transformer models.

## Features

- **Multiple File Format Support**: PDF, DOC, DOCX
- **Multiple LLM Options**:
  - Google Gemini
  - OpenAI GPT-3.5/4
  - OpenRouter (Free APIs)
  - Local Transformer Models (BERT, RoBERTa)
- **Structured JSON Output**: Standardized format for easy integration
- **Hybrid Extraction**: Combines LLM intelligence with rule-based patterns
- **Validation & Error Handling**: Robust output validation

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

## Quick Start (For Recruiters)

### 1. Clone and Setup
```bash
git clone https://github.com/sahit1011/AI_Resume_Parser.git
cd AI_Resume_Parser/navtech-assignment
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys (or use the provided working keys in .env.example)
```

### 3. Run the Application
```bash
# Web Interface (Recommended)
python app.py
# Then visit: http://localhost:5000

# Or Command Line
python src/main.py sample_resumes/sample_resume.txt --provider smart_transformer
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Setup
1. **Copy the example file**: `cp .env.example .env`
2. **Get API keys** (optional - local models work without keys):
   - **OpenRouter** (Free): https://openrouter.ai/keys
   - **Google Gemini** (Free with quota): https://makersuite.google.com/app/apikey
   - **OpenAI** (Paid): https://platform.openai.com/api-keys

3. **Working API keys provided** in `.env.example` for immediate testing

### Model Options (in order of recommendation)
1. **`smart_transformer`** (Default) - Fast, accurate, works offline
2. **`openrouter`** - Highest accuracy using DeepSeek R1 (free)
3. **`layoutlm_transformer`** - Fastest, good accuracy, works offline
4. **`gemini`** - Good accuracy, requires API key

## Usage

### Web Interface (Recommended)
```bash
python app.py
```
Visit http://localhost:5000 for the user-friendly interface.

### Command Line
```bash
# Using local transformer (no API key needed)
python src/main.py resume.pdf --provider smart_transformer

# Using OpenRouter API
python src/main.py resume.pdf --provider openrouter

# Using Gemini API
python src/main.py resume.pdf --provider gemini
```

### Google Colab
Open `notebooks/Resume_Parser_Colab.ipynb` in Google Colab for interactive usage.

## 🚨 Troubleshooting

### API Issues
- **"API key invalid"**: Usually a quota issue. Try the working keys in `.env.example`
- **"No API key found"**: Make sure you copied `.env.example` to `.env`
- **OpenRouter 401 error**: Check that you're using the correct model name in config

### Quick Solutions
1. 🆓 **Use local models** (no API needed): `smart_transformer` or `layoutlm_transformer`
2. 🔑 **Use provided keys**: Working API keys are in `.env.example`
3. 🆓 **Get OpenRouter free key**: https://openrouter.ai/keys (5 minutes)

### Performance Tips
- **Fastest**: `layoutlm_transformer` (~5 seconds)
- **Most accurate**: `openrouter` with DeepSeek R1 (~30 seconds)
- **Best balance**: `smart_transformer` (~6 seconds)

## Web Interface

Start the web application:
```bash
python app.py
```

Visit:
- http://localhost:5000/ - Main interface
- http://localhost:5000/status - API status
- http://localhost:5000/providers - Provider status

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
