#!/usr/bin/env python3
"""
NavTech Resume Parser - Main CLI Interface
Command-line interface for the resume parser assignment
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.resume_parser import ResumeParser
from config.llm_config import AVAILABLE_PROVIDERS

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_available_providers():
    """Print available LLM providers"""
    print("\n📋 Available LLM Providers:")
    print("=" * 50)
    
    for provider in AVAILABLE_PROVIDERS:
        print(f"• {provider['name']}: {provider['display_name']}")
        print(f"  Description: {provider['description']}")
        print(f"  Requires API Key: {'Yes' if provider['requires_api_key'] else 'No'}")
        print()

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="NavTech Resume Parser - Extract structured data from resumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py resume.pdf --provider gemini
  python src/main.py resume.pdf --provider smart_transformer --output result.json
  python src/main.py resume.pdf --provider openrouter --api-key YOUR_KEY
        """
    )
    
    parser.add_argument(
        'resume_file',
        help='Path to the resume file (PDF, DOC, DOCX, or TXT)'
    )
    
    parser.add_argument(
        '--provider', '-p',
        choices=[p['name'] for p in AVAILABLE_PROVIDERS],
        default='smart_transformer',
        help='LLM provider to use for parsing (default: smart_transformer)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for JSON result (default: print to stdout)'
    )
    
    parser.add_argument(
        '--api-key', '-k',
        help='API key for the selected provider (overrides environment variables)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--list-providers',
        action='store_true',
        help='List available providers and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # List providers if requested
    if args.list_providers:
        print_available_providers()
        return
    
    # Validate resume file
    resume_path = Path(args.resume_file)
    if not resume_path.exists():
        print(f"❌ Error: Resume file not found: {args.resume_file}")
        sys.exit(1)
    
    # Set API key if provided
    if args.api_key:
        if args.provider == 'gemini':
            os.environ['GEMINI_API_KEY'] = args.api_key
        elif args.provider == 'openai':
            os.environ['OPENAI_API_KEY'] = args.api_key
        elif args.provider == 'openrouter':
            os.environ['OPENROUTER_API_KEY'] = args.api_key
    
    try:
        # Initialize parser
        print(f"🚀 Initializing NavTech Resume Parser...")
        parser_instance = ResumeParser()
        
        # Check provider availability
        available_providers = parser_instance.get_available_providers()
        if args.provider not in available_providers:
            print(f"❌ Error: Provider '{args.provider}' is not available.")
            print(f"Available providers: {', '.join(available_providers)}")
            print("\n💡 Tip: Use --list-providers to see all options")
            sys.exit(1)
        
        # Parse resume
        print(f"📄 Parsing resume: {resume_path.name}")
        print(f"🔧 Using provider: {args.provider}")
        
        result = parser_instance.parse_resume(str(resume_path), args.provider)
        result_dict = result.to_dict()
        
        # Output result
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {output_path}")
        else:
            print("\n📋 Parsing Results:")
            print("=" * 50)
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   • Name: {result_dict.get('first_name', '')} {result_dict.get('last_name', '')}")
        print(f"   • Email: {result_dict.get('email', 'N/A')}")
        print(f"   • Phone: {result_dict.get('phone', 'N/A')}")
        print(f"   • Skills: {len(result_dict.get('skills', []))} found")
        print(f"   • Work History: {len(result_dict.get('work_history', []))} entries")
        print(f"   • Education: {len(result_dict.get('education_history', []))} entries")
        
        print("\n✅ Resume parsing completed successfully!")
        
    except Exception as e:
        logger.error(f"Resume parsing failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
