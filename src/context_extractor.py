"""
Context extraction and aggregation from video frames.

This module handles the aggregation of context from multiple video frames
using various strategies and similarity metrics.
"""

import numpy as np
import os
from typing import List, Dict, Tuple
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ContextExtractor:
    """
    Extracts and aggregates context from video frames.
    
    Leverages sentence transformers and similarity metrics to create
    coherent context from multiple frame captions.
    """
    
    def __init__(
        self,
        aggregation_method: str = "weighted_average",
        similarity_threshold: float = 0.7
    ):
        """
        Initialize context extractor.
        
        Args:
            aggregation_method: Method for aggregating contexts
            similarity_threshold: Threshold for context relevance
        """
        self.aggregation_method = aggregation_method
        self.similarity_threshold = similarity_threshold
        
        # Initialize sentence transformer for text similarity (optional)
        self.sentence_model = None
        try:
            # Only load if explicitly requested to avoid network dependency
            if os.getenv('LOAD_SENTENCE_TRANSFORMER', 'false').lower() == 'true':
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info("Sentence transformer loaded successfully")
            else:
                logging.info("Sentence transformer disabled (set LOAD_SENTENCE_TRANSFORMER=true to enable)")
        except Exception as e:
            logging.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        logging.info(f"ContextExtractor initialized with {aggregation_method} method")
    
    def aggregate_context(
        self,
        frame_captions: List[str],
        frame_features: List[np.ndarray],
        max_length: int = 512
    ) -> Dict[str, any]:
        """
        Aggregate context from multiple frame captions.
        
        Args:
            frame_captions: List of captions from video frames
            frame_features: List of feature vectors from frames
            max_length: Maximum length of aggregated context
            
        Returns:
            Dictionary containing aggregated context and metadata
        """
        if not frame_captions:
            return {
                "context_text": "",
                "features": np.array([]),
                "temporal_consistency": 0.0
            }
        
        if self.aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(
                frame_captions, frame_features, max_length
            )
        elif self.aggregation_method == "attention":
            return self._attention_aggregation(
                frame_captions, frame_features, max_length
            )
        elif self.aggregation_method == "concatenation":
            return self._concatenation_aggregation(
                frame_captions, frame_features, max_length
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _weighted_average_aggregation(
        self,
        frame_captions: List[str],
        frame_features: List[np.ndarray],
        max_length: int
    ) -> Dict[str, any]:
        """Aggregate using weighted average based on caption similarity."""
        # Calculate similarities between captions
        similarities = self._calculate_caption_similarities(frame_captions)
        
        # Calculate weights based on similarity and position
        weights = self._calculate_weights(similarities, len(frame_captions))
        
        # Weighted average of features
        valid_features = [f for f in frame_features if f is not None and len(f) > 0]
        if valid_features and len(valid_features) > 0:
            weighted_features = np.average(
                valid_features, 
                axis=0, 
                weights=weights[:len(valid_features)]
            )
        else:
            weighted_features = np.array([])
        
        # Select most representative captions
        top_indices = np.argsort(weights)[-3:]  # Top 3 captions
        selected_captions = [frame_captions[i] for i in top_indices]
        
        # Combine selected captions
        context_text = " ".join(selected_captions)
        if len(context_text) > max_length:
            context_text = context_text[:max_length]
        
        # Calculate temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(similarities)
        
        return {
            "context_text": context_text,  # Changed from "text" to "context_text"
            "features": weighted_features,
            "temporal_consistency": temporal_consistency,
            "weights": weights.tolist(),
            "selected_captions": selected_captions
        }
    
    def _attention_aggregation(
        self,
        frame_captions: List[str],
        frame_features: List[np.ndarray],
        max_length: int
    ) -> Dict[str, any]:
        """Aggregate using attention mechanism."""
        if self.sentence_model is None:
            logging.warning("Sentence transformer not available, falling back to concatenation")
            return self._concatenation_aggregation(frame_captions, frame_features, max_length)
        
        # Encode captions
        caption_embeddings = self.sentence_model.encode(frame_captions)
        
        # Calculate attention weights
        attention_weights = self._calculate_attention_weights(
            caption_embeddings, frame_features
        )
        
        # Apply attention to features
        if frame_features and len(frame_features) > 0:
            attended_features = np.average(
                frame_features,
                axis=0,
                weights=attention_weights
            )
        else:
            attended_features = np.array([])
        
        # Generate context text using attention
        context_text = self._generate_attention_context(
            frame_captions, attention_weights, max_length
        )
        
        return {
            "context_text": context_text,
            "features": attended_features,
            "temporal_consistency": 0.8,  # Placeholder
            "attention_weights": attention_weights.tolist()
        }
    
    def _concatenation_aggregation(
        self,
        frame_captions: List[str],
        frame_features: List[np.ndarray],
        max_length: int
    ) -> Dict[str, any]:
        """Simple concatenation of captions."""
        context_text = " ".join(frame_captions)
        if len(context_text) > max_length:
            context_text = context_text[:max_length]
        
        # Concatenate features
        if frame_features and len(frame_features) > 0:
            concatenated_features = np.concatenate(frame_features, axis=0)
        else:
            concatenated_features = np.array([])
        
        return {
            "context_text": context_text,
            "features": concatenated_features,
            "temporal_consistency": 1.0,  # Perfect consistency for concatenation
            "method": "concatenation"
        }
    
    def _calculate_caption_similarities(self, captions: List[str]) -> np.ndarray:
        """Calculate similarity matrix between captions."""
        if len(captions) <= 1:
            return np.array([[1.0]])
        
        if self.sentence_model is None:
            # Fallback to simple word overlap similarity
            return self._calculate_word_overlap_similarity(captions)
        
        try:
            embeddings = self.sentence_model.encode(captions)
            similarities = cosine_similarity(embeddings)
            return similarities
        except Exception as e:
            logging.warning(f"Error calculating similarities with sentence transformer: {e}")
            return self._calculate_word_overlap_similarity(captions)
    
    def _calculate_word_overlap_similarity(self, captions: List[str]) -> np.ndarray:
        """Fallback similarity calculation using word overlap."""
        n = len(captions)
        similarities = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                words_i = set(captions[i].lower().split())
                words_j = set(captions[j].lower().split())
                
                if len(words_i) == 0 or len(words_j) == 0:
                    similarity = 0.0
                else:
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    similarity = intersection / union if union > 0 else 0.0
                
                similarities[i, j] = similarity
                similarities[j, i] = similarity
        
        return similarities
    
    def _calculate_weights(
        self, 
        similarities: np.ndarray, 
        num_captions: int
    ) -> np.ndarray:
        """Calculate weights for caption aggregation."""
        # Base weights from similarity scores
        base_weights = np.mean(similarities, axis=1)
        
        # Add temporal weighting (later frames get slightly higher weight)
        temporal_weights = np.linspace(0.8, 1.2, num_captions)
        
        # Combine weights
        combined_weights = base_weights * temporal_weights
        
        # Normalize
        combined_weights = combined_weights / np.sum(combined_weights)
        
        return combined_weights
    
    def _calculate_attention_weights(
        self,
        caption_embeddings: np.ndarray,
        frame_features: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate attention weights for caption aggregation."""
        # Simple attention based on embedding norms
        attention_scores = np.linalg.norm(caption_embeddings, axis=1)
        
        # Softmax normalization
        attention_weights = np.exp(attention_scores)
        attention_weights = attention_weights / np.sum(attention_weights)
        
        return attention_weights
    
    def _generate_attention_context(
        self,
        captions: List[str],
        attention_weights: np.ndarray,
        max_length: int
    ) -> str:
        """Generate context text using attention weights."""
        # Sort captions by attention weight
        sorted_indices = np.argsort(attention_weights)[::-1]
        
        # Select top captions
        selected_captions = []
        current_length = 0
        
        for idx in sorted_indices:
            caption = captions[idx]
            if current_length + len(caption) <= max_length:
                selected_captions.append(caption)
                current_length += len(caption)
            else:
                break
        
        return " ".join(selected_captions)
    
    def _calculate_temporal_consistency(self, similarities: np.ndarray) -> float:
        """Calculate temporal consistency of captions."""
        if similarities.shape[0] <= 1:
            return 1.0
        
        # Calculate average similarity between consecutive frames
        consecutive_similarities = []
        for i in range(similarities.shape[0] - 1):
            consecutive_similarities.append(similarities[i, i+1])
        
        return np.mean(consecutive_similarities)
    
    def extract_key_phrases(self, context_text: str, max_phrases: int = 5) -> List[str]:
        """
        Extract key phrases from context text.
        
        Args:
            context_text: The aggregated context text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of key phrases
        """
        if not context_text:
            return []
        
        # Simple key phrase extraction based on word frequency
        words = context_text.lower().split()
        word_freq = {}
        
        for word in words:
            # Remove punctuation
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 2:  # Only consider words longer than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top phrases
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        key_phrases = [word for word, freq in sorted_words[:max_phrases]]
        
        return key_phrases
