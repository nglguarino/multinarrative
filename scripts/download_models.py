"""
Download and cache all required NLP models.

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("DOWNLOADING NLP MODELS")
print("="*60)

# Download spaCy model
print("\n1. Downloading spaCy model (en_core_web_lg)...")
try:
    import spacy
    spacy.cli.download("en_core_web_lg")
    print("   ✓ Downloaded successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Download NLTK data
print("\n2. Downloading NLTK data...")
try:
    import nltk
    nltk.download('punkt', quiet=True)
    print("   ✓ Downloaded successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Cache transformers model
print("\n3. Caching zero-shot classification model...")
try:
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    print("   ✓ Cached successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("   Note: This is optional. Fallback methods will be used if unavailable.")

print("\n" + "="*60)
print("MODEL DOWNLOAD COMPLETE")
print("="*60)
print("\nYou can now run:")
print("  python scripts/preprocess_articles.py --input <dir> --output <file>")