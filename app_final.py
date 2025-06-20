import streamlit as st
import streamlit_ace as st_ace
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util
import re
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import base64
import io
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom CSS for enhanced UI
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #667eea;
        color: #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-title {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    .success-box {
        background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2d3436;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2d3436;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar .stSelectbox label {
        color: #667eea;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: #667eea;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #667eea;
    }
    
    .analysis-result {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .misconception-item {
        background: #fff3cd;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    
    .feedback-positive {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .feedback-negative {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .code-editor-container {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        background: #f8f9fa;
    }
    
    .analysis-progress {
        background: #e9ecef;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Enhanced Text Analyzer ---
class EnhancedTextAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.max_chunk_tokens = 256
        
        # Initialize text classification pipeline
        try:
            self.classifier = pipeline("text-classification", 
                                     model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except:
            self.classifier = None
            
        # Grammar and style patterns
        self.grammar_patterns = {
            'run_on_sentences': r'[.!?]\s*[a-z]',
            'passive_voice': r'\b(was|were|is|are|been|being)\s+\w+ed\b',
            'weak_words': r'\b(very|really|quite|rather|somewhat|pretty|fairly)\b',
            'repeated_words': r'\b(\w+)\s+\1\b',
            'long_sentences': lambda text: [s for s in sent_tokenize(text) if len(s.split()) > 25]
        }

    def analyze_text_comprehensive(self, text: str, reference_text: str = None, 
                                 subject: str = "general", topic: str = "general", 
                                 analysis_type: str = "complete") -> Dict[str, Any]:
        """Comprehensive text analysis with multiple metrics"""
        logger.info(f"Analyzing text: subject={subject}, topic={topic}, analysis_type={analysis_type}")
        
        try:
            cleaned_text = self._preprocess_text(text)
            chunks = self._chunk_text(cleaned_text)

            # Core analysis components
            analysis_result = {
                "word_stats": self._get_word_statistics(cleaned_text),
                "sentence_stats": self._get_sentence_statistics(cleaned_text),
                "sentiment": self._get_enhanced_sentiment(cleaned_text),
                "readability": self._get_readability_metrics(cleaned_text),
                "coherence": self._get_coherence_score(chunks),
                "grammar_issues": self._detect_grammar_issues(cleaned_text),
                "style_analysis": self._analyze_writing_style(cleaned_text),
                "vocabulary_analysis": self._analyze_vocabulary(cleaned_text),
                "structure_analysis": self._analyze_text_structure(cleaned_text),
                "subject": subject.lower(),
                "topic": topic.lower(),
                "misconceptions": self._detect_subject_misconceptions(cleaned_text, subject, topic)
            }
            
            # Reference comparison if provided
            if reference_text:
                cleaned_reference = self._preprocess_text(reference_text)
                ref_chunks = self._chunk_text(cleaned_reference)
                analysis_result["accuracy"] = self._calculate_semantic_similarity(chunks, ref_chunks)
                analysis_result["content_overlap"] = self._calculate_content_overlap(cleaned_text, cleaned_reference)
            
            # Overall scoring
            analysis_result["overall_score"] = self._calculate_overall_score(analysis_result)
            
            logger.info(f"Text analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return {"error": str(e), "overall_score": 0}

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        return text

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            tokens = len(word_tokenize(sentence))
            if current_tokens + tokens <= self.max_chunk_tokens:
                current_chunk += " " + sentence
                current_tokens += tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = tokens
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _get_word_statistics(self, text: str) -> Dict[str, Any]:
        """Analyze word-level statistics"""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        return {
            "total_words": len(words),
            "unique_words": len(set(words)),
            "avg_word_length": np.mean([len(word) for word in words if word.isalpha()]) if any(word.isalpha() for word in words) else 0,
            "lexical_diversity": len(set(words)) / len(words) if words else 0,
            "words_per_sentence": len(words) / len(sentences) if sentences else 0
        }

    def _get_sentence_statistics(self, text: str) -> Dict[str, Any]:
        """Analyze sentence-level statistics"""
        sentences = sent_tokenize(text)
        if not sentences:
            return {"total_sentences": 0, "avg_sentence_length": 0, "sentence_variety": 0}
            
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        
        return {
            "total_sentences": len(sentences),
            "avg_sentence_length": np.mean(sentence_lengths),
            "sentence_length_std": np.std(sentence_lengths),
            "shortest_sentence": min(sentence_lengths) if sentence_lengths else 0,
            "longest_sentence": max(sentence_lengths) if sentence_lengths else 0,
            "sentence_variety": np.std(sentence_lengths) / np.mean(sentence_lengths) if sentence_lengths else 0
        }

    def _get_enhanced_sentiment(self, text: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis with multiple approaches"""
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        result = {
            "vader": {
                "compound": vader_scores['compound'],
                "positive": vader_scores['pos'],
                "negative": vader_scores['neg'],
                "neutral": vader_scores['neu']
            }
        }
        
        # RoBERTa sentiment if available
        if self.classifier:
            try:
                roberta_result = self.classifier(text[:512])  # Limit length
                result["roberta"] = {
                    "label": roberta_result[0]['label'],
                    "confidence": roberta_result[0]['score']
                }
            except:
                result["roberta"] = None
        
        # Overall sentiment classification
        compound = vader_scores['compound']
        if compound >= 0.05:
            result["overall"] = "positive"
        elif compound <= -0.05:
            result["overall"] = "negative"
        else:
            result["overall"] = "neutral"
            
        return result

    def _get_readability_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate multiple readability metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalpha())
        
        sentence_count = len(sentences)
        word_count = len([w for w in words if w.isalpha()])
        
        if sentence_count == 0 or word_count == 0:
            return {"flesch_reading_ease": 0, "flesch_kincaid_grade": 0, "readability_level": "Unknown"}
        
        # Flesch Reading Ease
        flesch_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        flesch_ease = max(0, min(100, flesch_ease))
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        fk_grade = max(0, fk_grade)
        
        # Readability level classification
        if flesch_ease >= 90:
            level = "Very Easy"
        elif flesch_ease >= 80:
            level = "Easy"
        elif flesch_ease >= 70:
            level = "Fairly Easy"
        elif flesch_ease >= 60:
            level = "Standard"
        elif flesch_ease >= 50:
            level = "Fairly Difficult"
        elif flesch_ease >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"
        
        return {
            "flesch_reading_ease": flesch_ease,
            "flesch_kincaid_grade": fk_grade,
            "readability_level": level,
            "avg_words_per_sentence": word_count / sentence_count,
            "avg_syllables_per_word": syllable_count / word_count if word_count > 0 else 0
        }

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_char_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
                
        if word.endswith("e"):
            count = max(1, count - 1)
            
        return max(1, count)

    def _get_coherence_score(self, chunks: List[str]) -> float:
        """Calculate text coherence using semantic similarity"""
        if len(chunks) < 2:
            return 1.0
            
        embeddings = self.sentence_model.encode(chunks, convert_to_tensor=True)
        similarities = util.cos_sim(embeddings, embeddings)
        
        # Get upper triangular similarities (excluding diagonal)
        upper_indices = np.triu_indices(len(chunks), k=1)
        avg_similarity = float(np.mean(similarities.cpu().numpy()[upper_indices]))
        
        return max(0.0, min(1.0, avg_similarity))

    def _detect_grammar_issues(self, text: str) -> Dict[str, Any]:
        """Detect common grammar and style issues"""
        issues = {}
        
        # Run-on sentences
        issues["run_on_sentences"] = len(re.findall(self.grammar_patterns['run_on_sentences'], text))
        
        # Passive voice detection
        issues["passive_voice_count"] = len(re.findall(self.grammar_patterns['passive_voice'], text, re.IGNORECASE))
        
        # Weak words
        issues["weak_words"] = len(re.findall(self.grammar_patterns['weak_words'], text, re.IGNORECASE))
        
        # Repeated words
        issues["repeated_words"] = len(re.findall(self.grammar_patterns['repeated_words'], text, re.IGNORECASE))
        
        # Long sentences
        long_sentences = self.grammar_patterns['long_sentences'](text)
        issues["long_sentences_count"] = len(long_sentences)
        issues["longest_sentence_length"] = max([len(s.split()) for s in long_sentences]) if long_sentences else 0
        
        return issues

    def _analyze_writing_style(self, text: str) -> Dict[str, Any]:
        """Analyze writing style characteristics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Sentence type analysis
        question_count = len([s for s in sentences if s.strip().endswith('?')])
        exclamation_count = len([s for s in sentences if s.strip().endswith('!')])
        
        return {
            "formality_score": self._calculate_formality_score(text),
            "complexity_score": self._calculate_complexity_score(text),
            "question_ratio": question_count / len(sentences) if sentences else 0,
            "exclamation_ratio": exclamation_count / len(sentences) if sentences else 0,
            "avg_word_frequency": self._calculate_avg_word_frequency(words)
        }

    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score based on various indicators"""
        formal_indicators = ['therefore', 'however', 'furthermore', 'consequently', 'nevertheless']
        informal_indicators = ['gonna', 'wanna', 'yeah', 'ok', 'awesome', 'cool']
        
        formal_count = sum(text.lower().count(word) for word in formal_indicators)
        informal_count = sum(text.lower().count(word) for word in informal_indicators)
        
        total_words = len(word_tokenize(text))
        if total_words == 0:
            return 0.5
            
        formal_ratio = formal_count / total_words
        informal_ratio = informal_count / total_words
        
        return min(1.0, max(0.0, 0.5 + (formal_ratio - informal_ratio) * 10))

    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score"""
        words = word_tokenize(text)
        if not words:
            return 0
            
        # Factors contributing to complexity
        avg_word_length = np.mean([len(word) for word in words if word.isalpha()]) if any(word.isalpha() for word in words) else 0
        long_words_ratio = len([word for word in words if len(word) > 6]) / len(words) if words else 0
        
        return min(1.0, (avg_word_length - 3) / 5 + long_words_ratio)

    def _calculate_avg_word_frequency(self, words: List[str]) -> float:
        """Calculate average word frequency (lexical diversity indicator)"""
        if not words:
            return 0
            
        from collections import Counter
        word_counts = Counter(word.lower() for word in words if word.isalpha())
        return np.mean(list(word_counts.values())) if word_counts else 0

    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze vocabulary richness and diversity"""
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        
        if not words:
            return {"vocabulary_richness": 0, "rare_words_count": 0, "academic_words_count": 0}
        
        from collections import Counter
        word_freq = Counter(words)
        
        # Common academic words
        academic_words = {'analyze', 'evaluate', 'synthesize', 'hypothesis', 'methodology', 
                         'empirical', 'theoretical', 'paradigm', 'construct', 'phenomenon'}
        
        return {
            "vocabulary_richness": len(set(words)) / len(words),
            "total_unique_words": len(set(words)),
            "most_common_words": dict(word_freq.most_common(5)),
            "rare_words_count": len([word for word, count in word_freq.items() if count == 1]),
            "academic_words_count": len([word for word in words if word in academic_words])
        }

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure and organization"""
        paragraphs = text.split('\n\n')
        sentences = sent_tokenize(text)
        
        # Transition words
        transition_words = ['first', 'second', 'third', 'finally', 'however', 'therefore', 
                           'furthermore', 'moreover', 'consequently', 'in addition']
        
        transition_count = sum(text.lower().count(word) for word in transition_words)
        
        return {
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "avg_sentences_per_paragraph": len(sentences) / max(1, len([p for p in paragraphs if p.strip()])),
            "transition_words_count": transition_count,
            "structure_score": min(1.0, transition_count / max(1, len(sentences)) * 10)
        }

    def _calculate_semantic_similarity(self, text_chunks: List[str], reference_chunks: List[str]) -> float:
        """Calculate semantic similarity between text and reference"""
        if not text_chunks or not reference_chunks:
            return 0.0
            
        text_embeddings = self.sentence_model.encode(text_chunks, convert_to_tensor=True)
        ref_embeddings = self.sentence_model.encode(reference_chunks, convert_to_tensor=True)
        
        similarities = util.cos_sim(text_embeddings, ref_embeddings)
        avg_similarity = float(np.mean(similarities.cpu().numpy()))
        
        return max(0.0, min(100.0, avg_similarity * 100))

    def _calculate_content_overlap(self, text: str, reference: str) -> Dict[str, Any]:
        """Calculate content overlap metrics"""
        text_words = set(word_tokenize(text.lower()))
        ref_words = set(word_tokenize(reference.lower()))
        
        if not text_words or not ref_words:
            return {"jaccard_similarity": 0, "overlap_ratio": 0, "unique_words": 0}
        
        intersection = text_words.intersection(ref_words)
        union = text_words.union(ref_words)
        
        return {
            "jaccard_similarity": len(intersection) / len(union) if union else 0,
            "overlap_ratio": len(intersection) / len(text_words) if text_words else 0,
            "unique_words_count": len(text_words - ref_words),
            "common_words": list(intersection)[:10]  # First 10 common words
        }

    def _detect_subject_misconceptions(self, text: str, subject: str, topic: str) -> List[Dict[str, Any]]:
        """Detect subject-specific misconceptions"""
        misconceptions = []
        text_lower = text.lower()
        
        if subject.lower() == "math":
            # Mathematical misconceptions
            if "multiply" in text_lower and "add" in text_lower and "same" in text_lower:
                misconceptions.append({
                    "type": "operation_confusion",
                    "description": "Possible confusion between multiplication and addition operations",
                    "severity": "medium"
                })
            
            if "divide by zero" in text_lower and ("undefined" not in text_lower and "impossible" not in text_lower):
                misconceptions.append({
                    "type": "division_by_zero",
                    "description": "Division by zero not recognized as undefined",
                    "severity": "high"
                })
        
        elif subject.lower() == "programming":
            # Programming misconceptions
            if "=" in text and "assign" in text_lower and "equal" in text_lower:
                misconceptions.append({
                    "type": "assignment_vs_equality",
                    "description": "Possible confusion between assignment (=) and equality comparison (==)",
                    "severity": "medium"
                })
        
        elif subject.lower() == "science":
            # Science misconceptions
            if "evolution" in text_lower and "theory" in text_lower and ("just" in text_lower or "only" in text_lower):
                misconceptions.append({
                    "type": "theory_misunderstanding",
                    "description": "Possible misunderstanding of scientific theory vs. hypothesis",
                    "severity": "medium"
                })
        
        return misconceptions

    def _calculate_overall_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall text quality score"""
        try:
            # Weight different components
            weights = {
                "readability": 0.25,
                "coherence": 0.20,
                "grammar": 0.20,
                "vocabulary": 0.15,
                "structure": 0.10,
                "style": 0.10
            }
            
            # Extract scores
            readability_score = min(1.0, analysis_result["readability"]["flesch_reading_ease"] / 100)
            coherence_score = analysis_result["coherence"]
            
            # Grammar score (inverse of issues)
            grammar_issues = analysis_result["grammar_issues"]
            total_issues = sum([v for v in grammar_issues.values() if isinstance(v, (int, float))])
            grammar_score = max(0, 1 - (total_issues / 20))  # Normalize by expected max issues
            
            # Vocabulary score
            vocab_score = analysis_result["vocabulary_analysis"]["vocabulary_richness"]
            
            # Structure score
            structure_score = analysis_result["structure_analysis"]["structure_score"]
            
            # Style score
            style_score = analysis_result["style_analysis"]["complexity_score"]
            
            # Calculate weighted score
            overall_score = (
                weights["readability"] * readability_score +
                weights["coherence"] * coherence_score +
                weights["grammar"] * grammar_score +
                weights["vocabulary"] * vocab_score +
                weights["structure"] * structure_score +
                weights["style"] * style_score
            ) * 100
            
            return max(0, min(100, overall_score))
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 50.0  # Default score

# --- Enhanced Session Manager ---
class EnhancedSessionManager:
    def __init__(self):
        if 'session_data' not in st.session_state:
            st.session_state.session_data = {
                'responses': [],
                'feedback': [],
                'misconceptions': [],
                'problems': [],
                'user_profile': {
                    'feedback_detail': 'Detailed', 
                    'theme': 'light',
                    'preferred_subjects': [],
                    'learning_goals': []
                },
                'analytics': {
                    'session_start': datetime.now().isoformat(),
                    'total_analyses': 0,
                    'time_spent': 0
                }
            }
    
    def add_response(self, response_data: Dict[str, Any]):
        st.session_state.session_data['responses'].append(response_data)
        st.session_state.session_data['analytics']['total_analyses'] += 1
    
    def add_feedback(self, feedback_data: Dict[str, Any]):
        st.session_state.session_data['feedback'].append(feedback_data)
    
    def update_misconceptions(self, misconceptions: List[Dict[str, Any]]):
        st.session_state.session_data['misconceptions'].extend(misconceptions)
    
    def add_problem(self, problem_data: Dict[str, Any]):
        st.session_state.session_data['problems'].append(problem_data)
    
    def get_session_responses(self) -> List[Dict[str, Any]]:
        return st.session_state.session_data['responses']
    
    def get_user_profile(self) -> Dict[str, Any]:
        return st.session_state.session_data['user_profile']
    
    def get_misconceptions(self) -> List[Dict[str, Any]]:
        return st.session_state.session_data['misconceptions']
    
    def get_session_problems(self) -> List[Dict[str, Any]]:
        return st.session_state.session_data['problems']
    
    def get_analytics(self) -> Dict[str, Any]:
        return st.session_state.session_data['analytics']
    
    def clear_session(self):
        st.session_state.session_data = {
            'responses': [],
            'feedback': [],
            'misconceptions': [],
            'problems': [],
            'user_profile': st.session_state.session_data['user_profile'],  # Keep user preferences
            'analytics': {
                'session_start': datetime.now().isoformat(),
                'total_analyses': 0,
                'time_spent': 0
            }
        }

# --- Enhanced Code Analyzer ---
class CodeAnalyzer:
    def __init__(self):
        pass

    def analyze_code(self, code: str, language: str = "python", problem_context: str = "", analysis_type: str = "complete_analysis") -> Dict[str, Any]:
        logger.info(f"Analyzing code: language={language}, analysis_type={analysis_type}")
        try:
            if language.lower() != "python":
                return {"error": "Only Python is supported", "overall_score": 0}

            syntax_score = self._check_syntax(code)
            logic_score = self._check_logic(code, problem_context)
            efficiency_score = self._check_efficiency(code)
            style_score = self._check_style(code)

            return {
                "syntax_score": syntax_score,
                "logic_score": logic_score,
                "efficiency_score": efficiency_score,
                "style_score": style_score,
                "overall_score": (syntax_score + logic_score + efficiency_score + style_score) / 4,
                "detailed_feedback": self._generate_code_feedback(code, syntax_score, logic_score, efficiency_score, style_score)
            }
        except Exception as e:
            logger.error(f"Code analysis failed: {str(e)}")
            return {"error": str(e), "overall_score": 0}

    def _check_syntax(self, code: str) -> float:
        """Check code syntax"""
        try:
            compile(code, '<string>', 'exec')
            return 100.0
        except SyntaxError:
            return 0.0
        except:
            return 50.0

    def _check_logic(self, code: str, context: str) -> float:
        """Basic logic checking"""
        score = 70.0  # Base score
        
        # Check for common logical issues
        if "if" in code and "else" not in code:
            score -= 10
        if code.count("for") > 3:  # Too many nested loops
            score -= 15
        if "while True" in code and "break" not in code:
            score -= 20
            
        return max(0, min(100, score))

    def _check_efficiency(self, code: str) -> float:
        """Check code efficiency"""
        score = 80.0
        
        # Simple efficiency checks
        if code.count("for") > 2:  # Nested loops
            score -= 20
        if "import" in code and len(code.split('\n')) > 50:
            score += 10  # Good for using libraries
            
        return max(0, min(100, score))

    def _check_style(self, code: str) -> float:
        """Check code style"""
        score = 75.0
        lines = code.split('\n')
        
        # Check for comments
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        if len(comment_lines) > 0:
            score += 15
            
        # Check line length
        long_lines = [line for line in lines if len(line) > 80]
        score -= len(long_lines) * 5
        
        return max(0, min(100, score))

    def _generate_code_feedback(self, code: str, syntax_score: float, logic_score: float, 
                              efficiency_score: float, style_score: float) -> List[str]:
        """Generate detailed feedback for code"""
        feedback = []
        
        if syntax_score < 100:
            feedback.append("❌ Syntax Error: Please check your code syntax")
        if logic_score < 70:
            feedback.append("⚠️ Logic Issues: Consider reviewing your control structures")
        if efficiency_score < 60:
            feedback.append("📈 Efficiency: Your code could be optimized further")
        if style_score < 70:
            feedback.append("✨ Style: Consider adding comments and following PEP 8")
            
        if not feedback:
            feedback.append("✅ Great code! Well structured and readable.")
            
        return feedback

# --- Main Streamlit Application ---
def main():
    st.set_page_config(
        page_title="Enhanced Text & Code Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize components
    text_analyzer = EnhancedTextAnalyzer()
    code_analyzer = CodeAnalyzer()
    session_manager = EnhancedSessionManager()
    
    # Main header
    st.markdown('<div class="main-header"> AI Powered Educational Feedback System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Text Analysis", "Code Analysis", "Comparison Mode"],
            index=0,
            key="analysis_mode"
        )
        
        if analysis_mode in ["Text Analysis", "Comparison Mode"]:
            st.markdown("#### Text Analysis Settings")
            subject = st.selectbox("Subject Area", 
                                 ["General", "Math", "Science", "Programming", "Literature", "History"],
                                 key="text_subject")
            topic = st.text_input("Specific Topic", placeholder="e.g., Calculus, Biology", key="text_topic")
            analysis_depth = st.selectbox("Analysis Depth", 
                                        ["Quick", "Standard", "Comprehensive"],
                                        key="text_depth")
        
        if analysis_mode in ["Code Analysis", "Comparison Mode"]:
            st.markdown("#### Code Analysis Settings")
            programming_language = st.selectbox("Programming Language", 
                                              ["Python", "JavaScript", "Java", "C++"],
                                              key="code_language")
            code_context = st.text_area("Problem Context", 
                                      placeholder="Describe what the code should do...",
                                      key="code_context")
        
        st.markdown("---")
        st.markdown("#### User Preferences")
        feedback_style = st.selectbox("Feedback Style", 
                                    ["Encouraging", "Detailed", "Concise"],
                                    key="feedback_style")
        show_visualizations = st.checkbox("Show Visualizations", value=True, key="show_viz")
        
        if st.button("Clear Session", key="clear_session"):
            session_manager.clear_session()
            st.success("Session cleared!")
            st.rerun()

    # Main content area
    if analysis_mode == "Text Analysis":
        render_text_analysis_ui(text_analyzer, session_manager, subject, topic, 
                              analysis_depth, feedback_style, show_visualizations)
    
    elif analysis_mode == "Code Analysis":
        render_code_analysis_ui(code_analyzer, session_manager, programming_language, 
                              code_context, feedback_style, show_visualizations)
    
    elif analysis_mode == "Comparison Mode":
        render_comparison_ui(text_analyzer, code_analyzer, session_manager, 
                           subject, topic, programming_language, code_context,
                           feedback_style, show_visualizations)

def render_text_analysis_ui(text_analyzer, session_manager, subject, topic, 
                           analysis_depth, feedback_style, show_visualizations):
    """Render the text analysis interface"""
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["✍️ Text Input", "📄 File Upload", "🔄 Batch Analysis"])
    
    with tab1:
        st.markdown('<div class="section-header">Text Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Enter your text for analysis:**")
            user_text = st.text_area(
                "Text to Analyze",
                height=400,
                placeholder="Paste or type your text here...",
                label_visibility="collapsed",
                key="user_text"
            )
            
            # Reference text for comparison (optional)
            with st.expander("📚 Add Reference Text (Optional)", expanded=False):
                reference_text = st.text_area(
                    "Reference Text",
                    height=200,
                    placeholder="Add reference text for comparison...",
                    help="This will be used to compare accuracy and similarity",
                    key="reference_text"
                )
        
        with col2:
            st.markdown("**Quick Stats:**")
            if user_text:
                word_count = len(user_text.split())
                char_count = len(user_text)
                sentence_count = len(sent_tokenize(user_text))
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Words</div>
                    <div class="metric-value">{word_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Characters</div>
                    <div class="metric-value">{char_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sentences</div>
                    <div class="metric-value">{sentence_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick readability estimate
                if sentence_count > 0:
                    avg_words_per_sentence = word_count / sentence_count
                    if avg_words_per_sentence < 15:
                        readability = "Easy"
                    elif avg_words_per_sentence < 20:
                        readability = "Medium"
                    else:
                        readability = "Complex"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Readability</div>
                        <div class="metric-value">{readability}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Analysis button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True, key="analyze_text"):
            if user_text.strip():
                with st.spinner("Analyzing your text..."):
                    # Perform analysis
                    analysis_result = text_analyzer.analyze_text_comprehensive(
                        text=user_text,
                        reference_text=reference_text if reference_text.strip() else None,
                        subject=subject,
                        topic=topic,
                        analysis_type=analysis_depth.lower()
                    )
                    
                    # Store in session
                    session_manager.add_response({
                        'type': 'text_analysis',
                        'timestamp': datetime.now().isoformat(),
                        'text': user_text,
                        'analysis': analysis_result,
                        'subject': subject,
                        'topic': topic
                    })
                    
                    if analysis_result.get('misconceptions'):
                        session_manager.update_misconceptions(analysis_result['misconceptions'])
                    
                    # Display results
                    display_text_analysis_results(analysis_result, show_visualizations, feedback_style)
            else:
                st.warning("Please enter some text to analyze!")
    
    with tab2:
        st.markdown('<div class="section-header">File Upload Analysis</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt'],
            help="Upload a text file for analysis",
            key="file_upload"
        )
        
        if uploaded_file is not None:
            # Process uploaded file
            file_content = process_uploaded_file(uploaded_file)
            if file_content:
                st.markdown(f"""
                <div class="success-box">
                    <strong>File uploaded successfully!</strong><br>
                    {len(file_content.split()):,} words detected
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔍 Analyze Uploaded Text", type="primary", key="analyze_uploaded"):
                    with st.spinner("Analyzing uploaded file..."):
                        analysis_result = text_analyzer.analyze_text_comprehensive(
                            text=file_content,
                            subject=subject,
                            topic=topic,
                            analysis_type=analysis_depth.lower()
                        )
                        
                        session_manager.add_response({
                            'type': 'file_analysis',
                            'timestamp': datetime.now().isoformat(),
                            'filename': uploaded_file.name,
                            'analysis': analysis_result,
                            'subject': subject,
                            'topic': topic
                        })
                        
                        if analysis_result.get('misconceptions'):
                            session_manager.update_misconceptions(analysis_result['misconceptions'])
                        
                        display_text_analysis_results(analysis_result, show_visualizations, feedback_style)
    
    with tab3:
        st.markdown('<div class="section-header">Batch Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Upload multiple text files for comparative analysis
        </div>
        """, unsafe_allow_html=True)
        
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Choose multiple text files",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload multiple text files for batch analysis",
            key="batch_upload"
        )
        
        if uploaded_files and len(uploaded_files) > 1:
            if st.button("🔍 Analyze All Files", type="primary", key="analyze_batch"):
                batch_results = []
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    file_content = process_uploaded_file(file)
                    if file_content:
                        analysis_result = text_analyzer.analyze_text_comprehensive(
                            text=file_content,
                            subject=subject,
                            topic=topic,
                            analysis_type="quick"
                        )
                        batch_results.append({
                            'filename': file.name,
                            'analysis': analysis_result
                        })
                        if analysis_result.get('misconceptions'):
                            session_manager.update_misconceptions(analysis_result['misconceptions'])
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display batch results
                display_batch_analysis_results(batch_results)

def render_code_analysis_ui(code_analyzer, session_manager, programming_language, 
                           code_context, feedback_style, show_visualizations):
    """Render the code analysis interface"""
    
    st.markdown('<div class="section-header">Code Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Enter your code for analysis:**")
        code_input = st_ace.st_ace(
            value="",
            language=programming_language.lower(),
            theme="monokai",
            keybinding="vscode",
            font_size=14,
            min_lines=20,
            placeholder="Write or paste your code here...",
            key="code_input"
        )
        
        # Expected output (optional)
        with st.expander("📋 Add Expected Output (Optional)", expanded=False):
            expected_output = st.text_area(
                "Expected Output",
                height=150,
                placeholder="Describe or paste the expected output...",
                help="This will help validate the code's correctness",
                key="expected_output"
            )
    
    with col2:
        st.markdown("**Quick Stats:**")
        if code_input:
            line_count = len(code_input.split('\n'))
            char_count = len(code_input)
            comment_count = len([line for line in code_input.split('\n') if line.strip().startswith('#')])
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Lines</div>
                <div class="metric-value">{line_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Characters</div>
                <div class="metric-value">{char_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Comments</div>
                <div class="metric-value">{comment_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick syntax check
            try:
                compile(code_input, '<string>', 'exec')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Syntax</div>
                    <div class="metric-value">✅ Valid</div>
                </div>
                """, unsafe_allow_html=True)
            except SyntaxError:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Syntax</div>
                    <div class="metric-value">❌ Invalid</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Analysis button
    if st.button("🔍 Analyze Code", type="primary", use_container_width=True, key="analyze_code"):
        if code_input.strip():
            with st.spinner("Analyzing your code..."):
                # Perform analysis
                analysis_result = code_analyzer.analyze_code(
                    code=code_input,
                    language=programming_language,
                    problem_context=code_context,
                    analysis_type="complete"
                )
                
                # Store in session
                session_manager.add_response({
                    'type': 'code_analysis',
                    'timestamp': datetime.now().isoformat(),
                    'code': code_input,
                    'analysis': analysis_result,
                    'language': programming_language,
                    'context': code_context
                })
                
                # Display results
                display_code_analysis_results(analysis_result, show_visualizations, feedback_style)
        else:
            st.warning("Please enter some code to analyze!")

def display_code_analysis_results(analysis_result, show_visualizations, feedback_style):
    """Display code analysis results"""
    
    if "error" in analysis_result:
        st.error(f"Analysis failed: {analysis_result['error']}")
        return
    
    # Overall Score
    st.markdown('<div class="section-header">📊 Code Analysis Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = analysis_result.get("overall_score", 0)
        color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overall Score</div>
            <div class="metric-value">{color} {score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        syntax_score = analysis_result.get("syntax_score", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Syntax Score</div>
            <div class="metric-value">{syntax_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        logic_score = analysis_result.get("logic_score", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Logic Score</div>
            <div class="metric-value">{logic_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        style_score = analysis_result.get("style_score", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Style Score</div>
            <div class="metric-value">{style_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Analysis Tabs
    tab1, tab2 = st.tabs(["📝 Feedback", "📊 Visualizations"])
    
    with tab1:
        st.markdown("#### 📋 Detailed Feedback")
        feedback = analysis_result.get("detailed_feedback", [])
        
        for comment in feedback:
            if "✅" in comment:
                st.markdown(f"""
                <div class="success-box">
                    {comment}
                </div>
                """, unsafe_allow_html=True)
            elif "❌" in comment or "⚠️" in comment:
                st.markdown(f"""
                <div class="warning-box">
                    {comment}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    {comment}
                </div>
                """, unsafe_allow_html=True)
        
        # Personalized recommendations
        if feedback_style == "Detailed":
            st.markdown("#### 💡 Recommendations")
            if analysis_result.get("syntax_score", 0) < 80:
                st.info("**Improve Syntax:** Check for missing colons, incorrect indentation, or unbalanced parentheses.")
            if analysis_result.get("logic_score", 0) < 70:
                st.info("**Refine Logic:** Ensure proper use of control structures and validate loop conditions.")
            if analysis_result.get("efficiency_score", 0) < 60:
                st.info("**Optimize Efficiency:** Consider reducing nested loops or using built-in functions.")
            if analysis_result.get("style_score", 0) < 70:
                st.info("**Enhance Style:** Add meaningful comments and follow PEP 8 guidelines for better readability.")
    
    with tab2:
        if show_visualizations:
            st.markdown("#### 📈 Code Quality Metrics")
            scores = [
                analysis_result.get("syntax_score", 0),
                analysis_result.get("logic_score", 0),
                analysis_result.get("efficiency_score", 0),
                analysis_result.get("style_score", 0)
            ]
            categories = ['Syntax', 'Logic', 'Efficiency', 'Style']
            
            fig = go.Figure(data=[
                go.Bar(name='Score', x=categories, y=scores, marker_color='#667eea')
            ])
            
            fig.update_layout(
                title="Code Quality Metrics",
                yaxis=dict(range=[0, 100], title="Score (%)"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_comparison_ui(text_analyzer, code_analyzer, session_manager, 
                        subject, topic, programming_language, code_context,
                        feedback_style, show_visualizations):
    """Render the comparison mode interface"""
    
    st.markdown('<div class="section-header">Comparison Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Compare text and code side-by-side to evaluate their quality and coherence.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✍️ Text Input")
        text_input = st.text_area(
            "Text to Analyze",
            height=300,
            placeholder="Paste or type your text here...",
            key="comparison_text"
        )
        
        if st.button("🔍 Analyze Text", type="primary", key="compare_text"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    text_result = text_analyzer.analyze_text_comprehensive(
                        text=text_input,
                        subject=subject,
                        topic=topic,
                        analysis_type="standard"
                    )
                    session_manager.add_response({
                        'type': 'text_comparison',
                        'timestamp': datetime.now().isoformat(),
                        'text': text_input,
                        'analysis': text_result,
                        'subject': subject,
                        'topic': topic
                    })
                    if text_result.get('misconceptions'):
                        session_manager.update_misconceptions(text_result['misconceptions'])
                    st.session_state['text_comparison_result'] = text_result
            else:
                st.warning("Please enter text to analyze!")
    
    with col2:
        st.markdown("### 💻 Code Input")
        code_input = st_ace.st_ace(
            value="",
            language=programming_language.lower(),
            theme="monokai",
            keybinding="vscode",
            font_size=14,
            min_lines=20,
            placeholder="Write or paste your code here...",
            key="comparison_code"
        )
        
        if st.button("🔍 Analyze Code", type="primary", key="compare_code"):
            if code_input.strip():
                with st.spinner("Analyzing code..."):
                    code_result = code_analyzer.analyze_code(
                        code=code_input,
                        language=programming_language,
                        problem_context=code_context,
                        analysis_type="complete"
                    )
                    session_manager.add_response({
                        'type': 'code_comparison',
                        'timestamp': datetime.now().isoformat(),
                        'code': code_input,
                        'analysis': code_result,
                        'language': programming_language,
                        'context': code_context
                    })
                    st.session_state['code_comparison_result'] = code_result
            else:
                st.warning("Please enter code to analyze!")
    
    # Display comparison results
    if 'text_comparison_result' in st.session_state and 'code_comparison_result' in st.session_state:
        st.markdown("### 📊 Comparison Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Text Analysis Summary")
            display_text_analysis_results(st.session_state['text_comparison_result'], 
                                       show_visualizations, feedback_style)
        
        with col2:
            st.markdown("#### Code Analysis Summary")
            display_code_analysis_results(st.session_state['code_comparison_result'], 
                                        show_visualizations, feedback_style)
        
        if show_visualizations:
            st.markdown("#### 📈 Score Comparison")
            fig = go.Figure(data=[
                go.Bar(
                    name='Text Score',
                    x=['Overall Score'],
                    y=[st.session_state['text_comparison_result'].get('overall_score', 0)],
                    marker_color='#667eea'
                ),
                go.Bar(
                    name='Code Score',
                    x=['Overall Score'],
                    y=[st.session_state['code_comparison_result'].get('overall_score', 0)],
                    marker_color='#764ba2'
                )
            ])
            
            fig.update_layout(
                barmode='group',
                title="Text vs Code Overall Score",
                yaxis=dict(range=[0, 100], title="Score (%)"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_text_analysis_results(analysis_result, show_visualizations, feedback_style):
    """Display comprehensive text analysis results"""
    
    if "error" in analysis_result:
        st.error(f"Analysis failed: {analysis_result['error']}")
        return
    
    # Overall Score
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = analysis_result.get("overall_score", 0)
        color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overall Score</div>
            <div class="metric-value">{color} {score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        readability = analysis_result.get("readability", {})
        ease_score = readability.get("flesch_reading_ease", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Readability</div>
            <div class="metric-value">{ease_score:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        coherence = analysis_result.get("coherence", 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Coherence</div>
            <div class="metric-value">{coherence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        vocab_richness = analysis_result.get("vocabulary_analysis", {}).get("vocabulary_richness", 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Vocabulary Richness</div>
            <div class="metric-value">{vocab_richness:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Writing Quality", "💭 Sentiment", "📊 Statistics", "🎯 Issues", "🔍 Insights"
    ])
    
    with tab1:
        display_writing_quality_analysis(analysis_result, show_visualizations)
    
    with tab2:
        display_sentiment_analysis(analysis_result, show_visualizations)
    
    with tab3:
        display_text_statistics(analysis_result, show_visualizations)
    
    with tab4:
        display_issues_and_improvements(analysis_result, feedback_style)
    
    with tab5:
        display_insights_and_recommendations(analysis_result, feedback_style)

def display_writing_quality_analysis(analysis_result, show_visualizations):
    """Display writing quality metrics"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📖 Readability Metrics")
        readability = analysis_result.get("readability", {})
        
        st.write(f"**Reading Ease:** {readability.get('flesch_reading_ease', 0):.1f}")
        st.write(f"**Grade Level:** {readability.get('flesch_kincaid_grade', 0):.1f}")
        st.write(f"**Difficulty:** {readability.get('readability_level', 'Unknown')}")
        
        # Progress bars for scores
        ease_score = readability.get('flesch_reading_ease', 0)
        st.progress(min(1.0, ease_score / 100))
    
    with col2:
        st.markdown("#### 🎨 Style Analysis")
        style = analysis_result.get("style_analysis", {})
        
        formality = style.get('formality_score', 0) * 100
        complexity = style.get('complexity_score', 0) * 100
        
        st.write(f"**Formality:** {formality:.1f}%")
        st.write(f"**Complexity:** {complexity:.1f}%")
        
        st.progress(formality / 100)
        st.progress(complexity / 100)
    
    if show_visualizations:
        st.markdown("#### 📈 Writing Quality Overview")
        fig = go.Figure()
        
        categories = ['Readability', 'Coherence', 'Vocabulary', 'Structure', 'Style']
        scores = [
            min(100, analysis_result.get('readability', {}).get('flesch_reading_ease', 0)),
            analysis_result.get('coherence', 0) * 100,
            analysis_result.get('vocabulary_analysis', {}).get('vocabulary_richness', 0) * 100,
            analysis_result.get('structure_analysis', {}).get('structure_score', 0) * 100,
            analysis_result.get('style_analysis', {}).get('formality_score', 0) * 100
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Writing Quality',
            line_color='#667eea'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Writing Quality Radar Chart",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_sentiment_analysis(analysis_result, show_visualizations):
    """Display sentiment analysis results"""
    
    sentiment = analysis_result.get("sentiment", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 😊 Sentiment Overview")
        
        overall_sentiment = sentiment.get("overall", "neutral")
        if overall_sentiment == "positive":
            st.markdown(f"""
            <div class="success-box">
                <strong>Overall Sentiment:</strong> {overall_sentiment.title()} 😊
            </div>
            """, unsafe_allow_html=True)
        elif overall_sentiment == "negative":
            st.markdown(f"""
            <div class="warning-box">
                <strong>Overall Sentiment:</strong> {overall_sentiment.title()} 😞
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
                <strong>Overall Sentiment:</strong> {overall_sentiment.title()} 😐
            </div>
            """, unsafe_allow_html=True)
        
        # VADER scores
        vader = sentiment.get("vader", {})
        st.write("**VADER Analysis:**")
        st.write(f"• Positive: {vader.get('positive', 0):.3f}")
        st.write(f"• Negative: {vader.get('negative', 0):.3f}")
        st.write(f"• Neutral: {vader.get('neutral', 0):.3f}")
        st.write(f"• Compound: {vader.get('compound', 0):.3f}")
        
        # RoBERTa scores if available
        if sentiment.get("roberta"):
            roberta = sentiment.get("roberta")
            st.write("**RoBERTa Analysis:**")
            st.write(f"• Label: {roberta.get('label', 'N/A')}")
            st.write(f"• Confidence: {roberta.get('confidence', 0):.3f}")
    
    with col2:
        if show_visualizations and vader:
            st.markdown("#### 📊 Sentiment Distribution")
            fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[vader.get('positive', 0), vader.get('negative', 0), vader.get('neutral', 0)],
                hole=.3,
                marker_colors=['#55efc4', '#ff7675', '#74b9ff']
            )])
            
            fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=12)
            fig.update_layout(title="Sentiment Distribution", height=400)
            
            st.plotly_chart(fig, use_container_width=True)

def display_text_statistics(analysis_result, show_visualizations):
    """Display text statistics"""
    
    word_stats = analysis_result.get("word_stats", {})
    sentence_stats = analysis_result.get("sentence_stats", {})
    vocab_analysis = analysis_result.get("vocabulary_analysis", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📝 Word Statistics")
        st.write(f"**Total Words:** {word_stats.get('total_words', 0):,}")
        st.write(f"**Unique Words:** {word_stats.get('unique_words', 0):,}")
        st.write(f"**Avg Word Length:** {word_stats.get('avg_word_length', 0):.1f}")
        st.write(f"**Lexical Diversity:** {word_stats.get('lexical_diversity', 0):.3f}")
    
    with col2:
        st.markdown("#### 📄 Sentence Statistics")
        st.write(f"**Total Sentences:** {sentence_stats.get('total_sentences', 0)}")
        st.write(f"**Avg Sentence Length:** {sentence_stats.get('avg_sentence_length', 0):.1f}")
        st.write(f"**Shortest Sentence:** {sentence_stats.get('shortest_sentence', 0)} words")
        st.write(f"**Longest Sentence:** {sentence_stats.get('longest_sentence', 0)} words")
    
    with col3:
        st.markdown("#### 📚 Vocabulary Analysis")
        st.write(f"**Vocabulary Richness:** {vocab_analysis.get('vocabulary_richness', 0):.3f}")
        st.write(f"**Rare Words:** {vocab_analysis.get('rare_words_count', 0)}")
        st.write(f"**Academic Words:** {vocab_analysis.get('academic_words_count', 0)}")
        
        # Most common words
        common_words = vocab_analysis.get('most_common_words', {})
        if common_words:
            st.write("**Most Common Words:**")
            for word, count in list(common_words.items())[:3]:
                st.write(f"• {word}: {count}")
    
    if show_visualizations:
        st.markdown("#### 📈 Word Length Distribution")
        words = word_tokenize(analysis_result.get('text', ''))
        word_lengths = [len(word) for word in words if word.isalpha()]
        
        if word_lengths:
            fig = px.histogram(
                x=word_lengths,
                nbins=20,
                title="Word Length Distribution",
                labels={'x': 'Word Length', 'y': 'Frequency'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def display_issues_and_improvements(analysis_result, feedback_style):
    """Display identified issues and improvement suggestions"""
    
    grammar_issues = analysis_result.get("grammar_issues", {})
    misconceptions = analysis_result.get("misconceptions", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Grammar & Style Issues")
        
        total_issues = 0
        if grammar_issues.get('run_on_sentences', 0) > 0:
            count = grammar_issues['run_on_sentences']
            total_issues += count
            st.markdown(f"""
            <div class="warning-box">
                <strong>Run-on Sentences:</strong> {count} detected<br>
                <small>Consider breaking long sentences into shorter ones.</small>
            </div>
            """, unsafe_allow_html=True)
        
        if grammar_issues.get('passive_voice_count', 0) > 0:
            count = grammar_issues['passive_voice_count']
            total_issues += count
            st.markdown(f"""
            <div class="warning-box">
                <strong>Passive Voice:</strong> {count} instances<br>
                <small>Try using active voice for clearer writing.</small>
            </div>
            """, unsafe_allow_html=True)
        
        if grammar_issues.get('weak_words', 0) > 0:
            count = grammar_issues['weak_words']
            total_issues += count
            st.markdown(f"""
            <div class="warning-box">
                <strong>Weak Words:</strong> {count} found<br>
                <small>Replace words like 'very', 'really' with stronger alternatives.</small>
            </div>
            """, unsafe_allow_html=True)
        
        if grammar_issues.get('repeated_words', 0) > 0:
            count = grammar_issues['repeated_words']
            total_issues += count
            st.markdown(f"""
            <div class="warning-box">
                <strong>Repeated Words:</strong> {count} instances<br>
                <small>Use synonyms to improve variety.</small>
            </div>
            """, unsafe_allow_html=True)
        
        if total_issues == 0:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Great job!</strong><br>
                No major grammar issues detected.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Subject-Specific Issues")
        
        if misconceptions:
            for misconception in misconceptions:
                severity_color = "error" if misconception.get('severity') == 'high' else "warning"
                severity_icon = "🚨" if misconception.get('severity') == 'high' else "⚠️"
                
                if severity_color == "error":
                    st.error(f"{severity_icon} **{misconception.get('type', 'Issue').replace('_', ' ').title()}**\n\n{misconception.get('description', '')}")
                else:
                    st.warning(f"{severity_icon} **{misconception.get('type', 'Issue').replace('_', ' ').title()}**\n\n{misconception.get('description', '')}")
        else:
            st.markdown("""
            <div class="success-box">
                <strong>✅ No Issues Found!</strong><br>
                No subject-specific misconceptions detected.
            </div>
            """, unsafe_allow_html=True)

def display_insights_and_recommendations(analysis_result, feedback_style):
    """Display insights and personalized recommendations"""
    
    st.markdown("#### 💡 Personalized Recommendations")
    
    overall_score = analysis_result.get("overall_score", 0)
    readability = analysis_result.get("readability", {})
    coherence = analysis_result.get("coherence", 0)
    grammar_issues = analysis_result.get("grammar_issues", {})
    
    recommendations = []
    
    # Generate recommendations based on analysis
    if overall_score >= 85:
        recommendations.append({
            "type": "success",
            "title": "Excellent Work!",
            "description": "Your text demonstrates strong writing skills across all areas.",
            "action": "Consider sharing your writing techniques with others or exploring advanced writing topics."
        })
    elif overall_score >= 70:
        recommendations.append({
            "type": "info",
            "title": "Good Foundation",
            "description": "Your writing has a solid foundation with room for refinement.",
            "action": "Focus on the specific areas highlighted in the issues section for improvement."
        })
    else:
        recommendations.append({
            "type": "warning",
            "title": "Needs Improvement",
            "description": "Your writing would benefit from focused practice in several areas.",
            "action": "Start with grammar and sentence structure, then work on vocabulary and coherence."
        })
    
    # Specific recommendations
    if readability.get('flesch_reading_ease', 0) < 30:
        recommendations.append({
            "type": "info",
            "title": "Simplify Your Writing",
            "description": "Your text is quite complex and may be difficult for readers to follow.",
            "action": "Use shorter sentences and simpler vocabulary where possible."
        })
    
    if coherence < 0.5:
        recommendations.append({
            "type": "warning",
            "title": "Improve Coherence",
            "description": "Your text lacks coherence between paragraphs and ideas.",
            "action": "Use transition words and ensure logical flow between concepts."
        })
    
    if sum([v for v in grammar_issues.values() if isinstance(v, (int, float))]) > 10:
        recommendations.append({
            "type": "warning",
            "title": "Grammar Focus Needed",
            "description": "Multiple grammar and style issues were detected.",
            "action": "Review grammar basics and consider using writing assistance tools."
        })
    
    # Display recommendations
    for rec in recommendations:
        if rec["type"] == "success":
            st.markdown(f"""
            <div class="success-box">
                <strong>{rec['title']}</strong><br>
                {rec['description']}<br>
                <small><em>Next Steps:</em> {rec['action']}</small>
            </div>
            """, unsafe_allow_html=True)
        elif rec["type"] == "info":
            st.markdown(f"""
            <div class="info-box">
                <strong>{rec['title']}</strong><br>
                {rec['description']}<br>
                <small><em>Recommendation:</em> {rec['action']}</small>
            </div>
            """, unsafe_allow_html=True)
        elif rec["type"] == "warning":
            st.markdown(f"""
            <div class="warning-box">
                <strong>{rec['title']}</strong><br>
                {rec['description']}<br>
                <small><em>Action Required:</em> {rec['action']}</small>
            </div>
            """, unsafe_allow_html=True)

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text content"""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        else:
            st.error("Only plain text (.txt) files are supported.")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def display_batch_analysis_results(batch_results):
    """Display results from batch analysis"""
    st.markdown("#### 📊 Batch Analysis Results")
    
    if not batch_results:
        st.warning("No files were successfully analyzed.")
        return
    
    # Create comparison dataframe
    comparison_data = []
    for result in batch_results:
        analysis = result['analysis']
        comparison_data.append({
            'File': result['filename'],
            'Overall Score': analysis.get('overall_score', 0),
            'Readability': analysis.get('readability', {}).get('flesch_reading_ease', 0),
            'Coherence': analysis.get('coherence', 0) * 100,
            'Word Count': analysis.get('word_stats', {}).get('total_words', 0),
            'Issues': sum([v for v in analysis.get('grammar_issues', {}).values() if isinstance(v, (int, float))])
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.markdown("#### 📋 Comparison Table")
    st.dataframe(df, use_container_width=True)
    
    # Visualizations
    if len(batch_results) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Score Comparison")
            fig = px.bar(df, x='File', y='Overall Score', 
                        title='Overall Scores Comparison',
                        color='Overall Score',
                        color_continuous_scale='viridis')
            fig.update_xaxis(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Word Count vs Score")
            fig = px.scatter(df, x='Word Count', y='Overall Score', 
                            size='Issues', hover_name='File',
                            title='Score vs Word Count',
                            size_max=20,
                            color='Readability',
                            color_continuous_scale='blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()