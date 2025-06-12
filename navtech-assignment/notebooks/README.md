# 📓 NavTech Resume Parser - Jupyter Notebooks

This folder contains Jupyter notebooks for easy testing and demonstration of the resume parser functionality.

## 📋 Available Notebooks

### 1. `NavTech_Resume_Parser_Updated.ipynb` ⭐ **RECOMMENDED**
**Updated notebook with current codebase integration**

- ✅ **Real API Integration**: Uses actual OpenRouter DeepSeek R1 API
- ✅ **No Fallback Data**: Shows actual errors instead of empty responses
- ✅ **Current Schema**: Matches the latest data structure
- ✅ **Easy Setup**: Simple API key configuration
- ✅ **File Upload**: Works in Google Colab with upload widget
- ✅ **Sample Testing**: Built-in sample resume for quick testing

### 2. `Resume_Parser_Colab.ipynb`
**Legacy notebook** (may have outdated dependencies)

## 🚀 Quick Start for Recruiters

### Option A: Google Colab (Recommended)
1. **Open in Colab**: Click the "Open in Colab" button or upload the notebook
2. **Get API Key**: Visit [OpenRouter](https://openrouter.ai/keys) and get a free API key
3. **Configure**: Add your API key in the configuration cell
4. **Run All Cells**: Execute all cells in order
5. **Upload Resume**: Use the file upload widget to test with your resume
6. **Get Results**: View structured JSON output

### Option B: Local Jupyter
1. **Install Dependencies**: `pip install -r ../requirements.txt`
2. **Start Jupyter**: `jupyter notebook`
3. **Open Notebook**: Open `NavTech_Resume_Parser_Updated.ipynb`
4. **Configure API Keys**: Add your keys in the configuration section
5. **Run and Test**: Execute cells and upload resume files

## 🔑 API Key Setup

### OpenRouter (Recommended - Free)
- **URL**: https://openrouter.ai/keys
- **Model**: DeepSeek R1 (Free tier available)
- **Setup**: Add to Google Colab secrets as `OPENROUTER_API_KEY`

### Google Gemini (Alternative - Free with quota)
- **URL**: https://makersuite.google.com/app/apikey
- **Setup**: Add to Google Colab secrets as `GEMINI_API_KEY`

### OpenAI (Alternative - Paid)
- **URL**: https://platform.openai.com/api-keys
- **Setup**: Add to Google Colab secrets as `OPENAI_API_KEY`

## 📊 Expected Output

The notebook will produce structured JSON output like:

```json
{
  "first_name": "John",
  "last_name": "Smith",
  "email": "john.smith@email.com",
  "phone": "+1-555-123-4567",
  "address": {
    "street": "",
    "city": "San Francisco",
    "state": "CA",
    "zip_code": "",
    "country": "USA"
  },
  "summary": "Experienced software engineer with 5+ years...",
  "skills": [
    {"name": "Python", "proficiency": ""},
    {"name": "JavaScript", "proficiency": ""},
    {"name": "React", "proficiency": ""}
  ],
  "education_history": [
    {
      "institution": "Stanford University",
      "degree": "Bachelor of Science in Computer Science",
      "field_of_study": "Computer Science",
      "graduation_year": "2019",
      "gpa": ""
    }
  ],
  "work_history": [
    {
      "company": "Tech Corp Inc.",
      "position": "Senior Software Engineer",
      "start_date": "January 2021",
      "end_date": "Present",
      "description": "Led development of microservices..."
    }
  ]
}
```

## 🛠️ Troubleshooting

### Common Issues:

1. **"No LLM providers available"**
   - ❌ API keys not configured
   - ✅ Add your API key in the configuration section

2. **"Invalid JSON response"**
   - ❌ API quota exceeded or invalid key
   - ✅ Check your API key and quota limits

3. **"Failed to extract text"**
   - ❌ Unsupported file format or corrupted file
   - ✅ Use PDF, DOC, DOCX, or TXT files

4. **Import errors**
   - ❌ Missing dependencies
   - ✅ Run the installation cell first

## 🎯 Features Demonstrated

- **Real AI Processing**: Actual LLM API calls (no hardcoded responses)
- **Multiple File Formats**: PDF, DOC, DOCX, TXT support
- **Error Handling**: Clear error messages for debugging
- **Structured Output**: JSON format ready for integration
- **Easy Testing**: Upload widget and sample data
- **Production Ready**: Code matches the main application

## 📝 Notes for Developers

- The notebook uses the same core logic as the main application
- API providers are simplified but functionally equivalent
- Error handling matches the updated codebase (no fallback mechanism)
- Schema definitions are current and up-to-date

## 🔗 Related Files

- **Main Application**: `../app.py` - Flask web interface
- **Core Logic**: `../src/` - Main parsing logic
- **Requirements**: `../requirements.txt` - Dependencies
- **Configuration**: `../.env.example` - Environment setup

---

**Ready to test? Open `NavTech_Resume_Parser_Updated.ipynb` and start parsing resumes! 🚀**
