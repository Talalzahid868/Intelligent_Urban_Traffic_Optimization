"""
NLP Module Wrapper

Provides a standardized interface to the NLP-based social media analysis
module (module6_NLP) for traffic event detection and sentiment analysis.
"""

import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float
    
    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'score': self.score
        }


@dataclass
class EventResult:
    """Result from traffic event detection."""
    events: List[str]
    severity: str  # high, medium, low
    has_traffic_impact: bool
    
    def to_dict(self) -> dict:
        return {
            'events': self.events,
            'severity': self.severity,
            'has_traffic_impact': self.has_traffic_impact
        }


@dataclass
class NLPAnalysisResult:
    """Complete result from NLP analysis."""
    original_text: str
    cleaned_text: str
    events: EventResult
    sentiment: SentimentResult
    locations: List[str]
    impact_score: float
    
    def to_dict(self) -> dict:
        return {
            'original_text': self.original_text,
            'cleaned_text': self.cleaned_text,
            'events': self.events.to_dict(),
            'sentiment': self.sentiment.to_dict(),
            'locations': self.locations,
            'impact_score': self.impact_score
        }


# Traffic event keywords for rule-based classification
TRAFFIC_EVENTS = {
    'accident': ['accident', 'crash', 'collision', 'overturned', 'wreck', 'fender bender'],
    'construction': ['construction', 'roadwork', 'repair', 'maintenance', 'closure', 'lane closure'],
    'event': ['concert', 'game', 'marathon', 'parade', 'festival', 'stadium', 'arena'],
    'weather': ['rain', 'snow', 'ice', 'fog', 'storm', 'flood', 'visibility'],
    'congestion': ['traffic', 'gridlock', 'jam', 'slow', 'backed up', 'delay', 'stuck']
}


class NLPWrapper:
    """
    Wrapper for NLP-based social media traffic analysis.
    
    This wrapper provides a clean interface to NLP models for analyzing
    traffic-related social media content.
    
    Example:
        wrapper = NLPWrapper()
        result = wrapper.analyze_text("Major accident on Highway 101!")
        print(f"Impact score: {result.impact_score}")
    """
    
    def __init__(self, device: int = -1):
        """
        Initialize the NLP wrapper.
        
        Args:
            device: Compute device. -1 for CPU, 0+ for GPU.
        """
        self._sentiment_model = None
        self._ner_model = None
        self._device = device
        self._models_loaded = False
    
    def _load_models(self):
        """Lazy load NLP models."""
        if self._models_loaded:
            return
        
        try:
            from transformers import pipeline
            
            # Load sentiment analysis model
            self._sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self._device
            )
            
            # Load NER model for location extraction
            self._ner_model = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=self._device
            )
            
            self._models_loaded = True
            
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers torch"
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Keep hashtag content
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emojis but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _detect_events(self, text: str) -> EventResult:
        """Detect traffic events from text."""
        text_lower = text.lower()
        detected_events = []
        
        for event_type, keywords in TRAFFIC_EVENTS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_events.append(event_type)
                    break
        
        # Determine severity
        severity = 'low'
        if 'accident' in detected_events:
            severity = 'high'
        elif 'construction' in detected_events or 'event' in detected_events:
            severity = 'medium'
        elif 'weather' in detected_events:
            severity = 'medium'
        
        return EventResult(
            events=list(set(detected_events)) if detected_events else ['general'],
            severity=severity,
            has_traffic_impact=len(detected_events) > 0
        )
    
    def _analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        self._load_models()
        
        try:
            result = self._sentiment_model(text[:512])[0]
            return SentimentResult(
                label=result['label'],
                score=round(result['score'], 4)
            )
        except Exception:
            return SentimentResult(label='NEUTRAL', score=0.5)
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract location entities from text."""
        self._load_models()
        
        try:
            entities = self._ner_model(text)
            return [
                ent['word'] for ent in entities
                if ent['entity_group'] == 'LOC'
            ]
        except Exception:
            return []
    
    def _calculate_impact_score(
        self, 
        events: EventResult, 
        sentiment: SentimentResult,
        locations: List[str]
    ) -> float:
        """Calculate traffic impact score (0-100)."""
        score = 0
        
        # Severity contribution
        severity_scores = {'high': 40, 'medium': 25, 'low': 10}
        score += severity_scores.get(events.severity, 0)
        
        # Negative sentiment indicates frustration
        if sentiment.label == 'NEGATIVE':
            score += 20 * sentiment.score
        
        # More event types = higher complexity
        score += len(events.events) * 5
        
        # Location specificity bonus
        if len(locations) > 0:
            score += 10
        
        return min(100, round(score, 2))
    
    def analyze_text(self, text: str) -> NLPAnalysisResult:
        """
        Analyze a single social media post for traffic relevance.
        
        Args:
            text: Social media post text.
            
        Returns:
            NLPAnalysisResult with event detection, sentiment, and impact score.
        """
        cleaned = self._clean_text(text)
        events = self._detect_events(cleaned)
        sentiment = self._analyze_sentiment(cleaned)
        locations = self._extract_locations(text)  # Use original for NER
        impact = self._calculate_impact_score(events, sentiment, locations)
        
        return NLPAnalysisResult(
            original_text=text,
            cleaned_text=cleaned,
            events=events,
            sentiment=sentiment,
            locations=locations,
            impact_score=impact
        )
    
    def analyze_batch(self, texts: List[str]) -> List[NLPAnalysisResult]:
        """
        Analyze multiple social media posts.
        
        Args:
            texts: List of social media post texts.
            
        Returns:
            List of NLPAnalysisResults.
        """
        return [self.analyze_text(text) for text in texts]
    
    def detect_events(self, text: str) -> EventResult:
        """
        Quick event detection without full NLP analysis.
        
        Args:
            text: Text to analyze.
            
        Returns:
            EventResult with detected events.
        """
        cleaned = self._clean_text(text)
        return self._detect_events(cleaned)
    
    def get_sentiment(self, text: str) -> SentimentResult:
        """
        Get sentiment for text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            SentimentResult.
        """
        cleaned = self._clean_text(text)
        return self._analyze_sentiment(cleaned)
    
    def extract_locations(self, text: str) -> List[str]:
        """
        Extract location mentions from text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            List of location strings.
        """
        return self._extract_locations(text)
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded models."""
        return {
            'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english',
            'ner_model': 'dslim/bert-base-NER',
            'device': 'CPU' if self._device == -1 else f'GPU:{self._device}'
        }
