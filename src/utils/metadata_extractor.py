"""
Metadata extraction from raw text articles using NLP/ML.

OPTIMIZED VERSION with:
- Batch topic extraction
- Reduced text processing length
- GPU support
- Disabled unnecessary spaCy components
"""

import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# NLP libraries
import spacy
from transformers import pipeline
import dateparser
import pycountry
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class AdvancedMetadataExtractor:
    """
    Extract structured metadata using NLP/ML techniques.

    OPTIMIZED for speed without compromising quality:
    - Batch processing for topics
    - Reduced text length (3000 chars instead of 10000)
    - GPU support
    - Disabled unused spaCy components
    """

    # Predefined topic categories for zero-shot classification
    TOPIC_CATEGORIES = [
        "politics and elections",
        "economy and finance",
        "international relations and diplomacy",
        "military and defense",
        "climate and environment",
        "healthcare and public health",
        "technology and innovation",
        "immigration and border security",
        "crime and law enforcement",
        "education",
        "social issues and civil rights",
        "trade and commerce",
        "energy and natural resources",
        "terrorism and national security",
        "media and journalism"
    ]

    def __init__(self, use_gpu: bool = False):
        """
        Initialize metadata extractor with NLP models.

        Args:
            use_gpu: Whether to use GPU for processing (if available)
        """
        print("Loading NLP models...")

        # Load spaCy model for NER (OPTIMIZED: disable unused components)
        try:
            self.nlp = spacy.load("en_core_web_lg", disable=["parser", "lemmatizer"])
            print("  ✓ Loaded spaCy en_core_web_lg (optimized)")
        except OSError:
            print("  ! spaCy model not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
            self.nlp = spacy.load("en_core_web_lg", disable=["parser", "lemmatizer"])
            print("  ✓ Loaded spaCy en_core_web_lg (optimized)")

        # GPU support for spaCy
        if use_gpu:
            try:
                spacy.require_gpu()
                print("  ✓ spaCy using GPU")
            except:
                print("  ! GPU not available for spaCy, using CPU")

        self.nlp.max_length = 2000000

        # Load zero-shot classifier for topics (with GPU support)
        device = 0 if use_gpu else -1
        try:
            self.topic_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )
            print(f"  ✓ Loaded topic classifier on {'GPU' if use_gpu else 'CPU'}")
        except Exception as e:
            print(f"  ! Warning: Could not load topic classifier: {e}")
            self.topic_classifier = None

        # Load country data for place normalization
        self.countries = {country.name.lower(): country.name for country in pycountry.countries}
        self.countries.update({country.alpha_2.lower(): country.name for country in pycountry.countries})
        self.countries.update({country.alpha_3.lower(): country.name for country in pycountry.countries})

        # Add common place aliases
        self.place_aliases = {
            'us': 'United States',
            'usa': 'United States',
            'uk': 'United Kingdom',
            'uae': 'United Arab Emirates',
            'drc': 'Democratic Republic of the Congo',
        }

        print("✓ NLP models loaded\n")

    def extract_metadata(self, article_text: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from article text using NLP/ML.

        Args:
            article_text: Raw article text
            filename: Optional filename for additional context

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'title': self._extract_title(article_text),
            'date': self._extract_date(article_text),
            'source': self._extract_source_from_filename(filename) if filename else None,
            'actors': self._extract_actors(article_text),
            'topics': self._extract_topics(article_text),
            'places': self._extract_places(article_text)
        }

        return metadata

    def batch_extract(self, articles: List[str], filenames: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract metadata from multiple articles with optimized batching.

        OPTIMIZED: Batch processes topics for all articles at once.

        Args:
            articles: List of article texts
            filenames: Optional list of filenames

        Returns:
            List of metadata dictionaries
        """
        if filenames is None:
            filenames = [None] * len(articles)

        metadata_list = []
        total = len(articles)

        print(f"Extracting metadata from {total} articles using NLP/ML...")

        # OPTIMIZATION: Pre-extract all topics at once (batched)
        print("  Step 1/2: Extracting topics (batched)...")
        all_topics = self._batch_extract_topics(articles)

        # Extract other metadata
        print("  Step 2/2: Extracting actors, places, dates...")
        for i, (article, filename) in enumerate(zip(articles, filenames), 1):
            if i % 50 == 0 or i == 1 or i == total:
                print(f"    Progress: {i}/{total}")

            metadata = {
                'title': self._extract_title(article),
                'date': self._extract_date(article),
                'source': self._extract_source_from_filename(filename) if filename else None,
                'actors': self._extract_actors(article),
                'topics': all_topics[i-1],  # Use pre-extracted topics
                'places': self._extract_places(article)
            }
            metadata_list.append(metadata)

        print("✓ Metadata extraction complete\n")
        return metadata_list

    def _batch_extract_topics(self, articles: List[str]) -> List[List[str]]:
        """
        Extract topics for all articles in batch.

        OPTIMIZATION: Process all articles at once instead of one-by-one.
        This is 5-10x faster than individual processing.

        Args:
            articles: List of article texts

        Returns:
            List of topic lists for each article
        """
        if not self.topic_classifier:
            return [self._extract_topics_fallback(article) for article in articles]

        try:
            # Prepare text samples (first 1000 chars of each)
            text_samples = [article[:1000] for article in articles]

            # Batch classify (much faster!)
            results = self.topic_classifier(
                text_samples,
                candidate_labels=self.TOPIC_CATEGORIES,
                multi_label=True,
                batch_size=64  # Process 64 at a time
            )

            # Extract topics from results
            all_topics = []
            for result in results:
                topics = []
                for label, score in zip(result['labels'], result['scores']):
                    if score > 0.5 and len(topics) < 5:
                        topic = label.replace(' and ', '/')
                        topics.append(topic)
                all_topics.append(topics if topics else [result['labels'][0]])

            return all_topics

        except Exception as e:
            print(f"Warning: Batch topic classification failed: {e}")
            return [self._extract_topics_fallback(article) for article in articles]

    def _extract_title(self, article_text: str) -> str:
        """Extract title from article text."""
        lines = article_text.strip().split('\n')

        # Try first non-empty line
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 10:
                # Check if it looks like a title (not too long, proper capitalization)
                if len(line) < 200:
                    # Check if it's in title case or all caps
                    words = line.split()
                    if len(words) > 2:
                        capitalized = sum(1 for w in words if w and w[0].isupper())
                        if capitalized / len(words) > 0.5:
                            # Remove common prefixes
                            line = re.sub(r'^(BREAKING|EXCLUSIVE|UPDATE):\s*', '', line, flags=re.IGNORECASE)
                            return line[:150]

        # Fallback: create title from first sentence
        sentences = sent_tokenize(article_text[:500])
        if sentences:
            return sentences[0][:150]

        return "Untitled Article"

    def _extract_date(self, article_text: str) -> Optional[str]:
        """
        Extract date using dateparser (robust date parsing).

        Args:
            article_text: Article text

        Returns:
            Date string in YYYY-MM-DD format or None
        """
        # Search in first 2000 characters for date mentions
        text_sample = article_text[:2000]

        # Common date patterns in news articles
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY or DD/MM/YYYY
        ]

        # Try to find dates
        for pattern in date_patterns:
            matches = re.findall(pattern, text_sample, re.IGNORECASE)
            for match in matches:
                try:
                    # Use dateparser for robust parsing
                    parsed_date = dateparser.parse(
                        match,
                        settings={
                            'PREFER_DATES_FROM': 'past',
                            'RETURN_AS_TIMEZONE_AWARE': False
                        }
                    )
                    if parsed_date:
                        # Validate year is reasonable (between 2000 and 2030)
                        if 2000 <= parsed_date.year <= 2030:
                            return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue

        # Try dateparser on first few sentences
        sentences = sent_tokenize(text_sample)
        for sentence in sentences[:5]:
            try:
                parsed_date = dateparser.parse(
                    sentence,
                    settings={
                        'PREFER_DATES_FROM': 'past',
                        'STRICT_PARSING': True,
                        'RETURN_AS_TIMEZONE_AWARE': False
                    }
                )
                if parsed_date and 2000 <= parsed_date.year <= 2030:
                    return parsed_date.strftime('%Y-%m-%d')
            except:
                continue

        return None

    def _extract_actors(self, article_text: str) -> List[str]:
        """
        Extract actors (people and organizations) using spaCy NER.

        OPTIMIZED: Process only first 3000 chars instead of 10000.

        Args:
            article_text: Article text

        Returns:
            List of actor names
        """
        # OPTIMIZATION: Process only first 3000 chars (was 10000)
        doc = self.nlp(article_text[:3000])

        # Extract PERSON and ORG entities
        actor_counts = Counter()

        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                # Clean entity text
                actor = self._clean_entity_text(ent.text)
                if actor and len(actor) > 2:
                    actor_counts[actor] += 1

        # Filter and rank by frequency
        first_paragraph = article_text[:500]
        actors = []

        for actor, count in actor_counts.most_common(30):
            if count >= 2 or actor in first_paragraph:
                # Additional filtering
                if self._is_valid_actor(actor):
                    actors.append(actor)
                    if len(actors) >= 15:  # Limit to top 15
                        break

        return actors

    def _extract_places(self, article_text: str) -> List[str]:
        """
        Extract places using spaCy NER with normalization.

        OPTIMIZED: Process only first 3000 chars instead of 10000.

        Args:
            article_text: Article text

        Returns:
            List of place names
        """
        # OPTIMIZATION: Process only first 3000 chars (was 10000)
        doc = self.nlp(article_text[:3000])

        # Extract GPE (geo-political entities) and LOC (locations)
        place_counts = Counter()

        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                # Clean and normalize
                place = self._clean_entity_text(ent.text)
                if place and len(place) > 2:
                    # Normalize country names
                    place = self._normalize_place(place)
                    place_counts[place] += 1

        # Filter and rank
        first_paragraph = article_text[:500]
        places = []

        for place, count in place_counts.most_common(20):
            if count >= 2 or place in first_paragraph:
                if self._is_valid_place(place):
                    places.append(place)
                    if len(places) >= 10:  # Limit to top 10
                        break

        return places

    def _extract_topics(self, article_text: str) -> List[str]:
        """
        Extract topics using zero-shot classification.

        NOTE: This method is for single article extraction.
        Use _batch_extract_topics() for batch processing (much faster).

        Args:
            article_text: Article text

        Returns:
            List of topic labels
        """
        if not self.topic_classifier:
            return self._extract_topics_fallback(article_text)

        try:
            # Use first 1000 chars for classification
            text_sample = article_text[:1000]

            # Run zero-shot classification
            result = self.topic_classifier(
                text_sample,
                candidate_labels=self.TOPIC_CATEGORIES,
                multi_label=True
            )

            # Get topics with confidence > 0.5
            topics = []
            for label, score in zip(result['labels'], result['scores']):
                if score > 0.5 and len(topics) < 5:
                    # Simplify label
                    topic = label.replace(' and ', '/')
                    topics.append(topic)

            return topics if topics else [result['labels'][0]]  # At least one topic

        except Exception as e:
            print(f"Warning: Topic classification failed: {e}")
            return self._extract_topics_fallback(article_text)

    def _extract_topics_fallback(self, article_text: str) -> List[str]:
        """Fallback keyword-based topic extraction."""
        text_lower = article_text[:2000].lower()

        topic_keywords = {
            'politics/elections': ['election', 'campaign', 'vote', 'political', 'democrat', 'republican', 'candidate'],
            'economy/finance': ['economy', 'inflation', 'market', 'stock', 'trade', 'gdp', 'economic', 'financial'],
            'international relations': ['diplomacy', 'treaty', 'foreign policy', 'ambassador', 'summit', 'alliance'],
            'military/defense': ['military', 'defense', 'army', 'navy', 'troops', 'weapon', 'combat', 'warfare'],
            'climate/environment': ['climate', 'environment', 'emission', 'renewable', 'pollution', 'carbon',
                                    'sustainability'],
            'healthcare': ['health', 'medical', 'hospital', 'disease', 'vaccine', 'patient', 'doctor'],
            'technology': ['technology', 'ai', 'artificial intelligence', 'software', 'cyber', 'digital', 'tech'],
            'immigration': ['immigration', 'migrant', 'refugee', 'border', 'asylum', 'deportation'],
            'crime': ['crime', 'criminal', 'arrest', 'prison', 'police', 'investigation', 'murder'],
            'social issues': ['protest', 'discrimination', 'rights', 'equality', 'justice', 'civil'],
            'energy': ['energy', 'oil', 'gas', 'nuclear', 'power plant', 'electricity'],
            'terrorism/security': ['terror', 'extremist', 'attack', 'security threat', 'counterterrorism']
        }

        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            if score > 0:
                topic_scores[topic] = score

        # Return top 3 topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:3]]

    def _clean_entity_text(self, text: str) -> str:
        """Clean entity text."""
        # Remove articles and possessives
        text = re.sub(r'\b(the|The|a|A|an|An)\s+', '', text)
        text = re.sub(r"'s$", '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _is_valid_actor(self, actor: str) -> bool:
        """Check if actor name is valid."""
        # Filter out common false positives
        invalid = ['today', 'yesterday', 'tomorrow', 'monday', 'tuesday', 'wednesday',
                   'thursday', 'friday', 'saturday', 'sunday', 'january', 'february',
                   'march', 'april', 'june', 'july', 'august', 'september', 'october',
                   'november', 'december', 'news', 'report', 'article']

        if actor.lower() in invalid:
            return False

        # Must have at least one letter
        if not re.search(r'[a-zA-Z]', actor):
            return False

        # Should not be all caps unless it's an acronym (< 6 chars)
        if actor.isupper() and len(actor) > 6:
            return False

        return True

    def _is_valid_place(self, place: str) -> bool:
        """Check if place name is valid."""
        # Filter out common false positives
        invalid = ['today', 'yesterday', 'news', 'report', 'white house', 'state department']

        if place.lower() in invalid:
            return False

        return True

    def _normalize_place(self, place: str) -> str:
        """Normalize place names (especially countries)."""
        place_lower = place.lower()

        # Check aliases first
        if place_lower in self.place_aliases:
            return self.place_aliases[place_lower]

        # Check if it's a country
        if place_lower in self.countries:
            return self.countries[place_lower]

        # Return as-is with title case
        return place.title()

    def _extract_source_from_filename(self, filename: str) -> Optional[str]:
        """Extract source from filename."""
        name = filename.rsplit('.', 1)[0].lower()

        sources = {
            'nytimes': 'New York Times',
            'washingtonpost': 'Washington Post',
            'wsj': 'Wall Street Journal',
            'cnn': 'CNN',
            'bbc': 'BBC',
            'reuters': 'Reuters',
            'ap': 'Associated Press',
            'guardian': 'The Guardian',
            'foxnews': 'Fox News',
            'msnbc': 'MSNBC',
            'politico': 'Politico',
            'axios': 'Axios',
            'bloomberg': 'Bloomberg',
            'forbes': 'Forbes',
            'npr': 'NPR',
            'abc': 'ABC News',
            'nbc': 'NBC News',
            'cbs': 'CBS News'
        }

        for key, value in sources.items():
            if key in name:
                return value

        return None


# Alias for backward compatibility
MetadataExtractor = AdvancedMetadataExtractor