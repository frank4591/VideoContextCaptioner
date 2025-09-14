"""
Text processing utilities.

This module provides utility functions for text processing,
caption analysis, and text similarity calculations.
"""

import re
import string
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
import numpy as np

class TextUtils:
    """
    Utility class for text processing operations.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Clean text first
        text = TextUtils.clean_text(text)
        
        # Split into words
        tokens = text.split()
        
        # Remove punctuation from tokens
        tokens = [token.strip(string.punctuation) for token in tokens]
        
        # Remove empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
    
    @staticmethod
    def calculate_word_frequency(text: str) -> Dict[str, int]:
        """
        Calculate word frequency in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of word frequencies
        """
        tokens = TextUtils.tokenize_text(text)
        return Counter(tokens)
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Tokenize texts
        tokens1 = set(TextUtils.tokenize_text(text1))
        tokens2 = set(TextUtils.tokenize_text(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[Tuple[str, int]]:
        """
        Extract keywords from text based on frequency.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of (keyword, frequency) tuples sorted by frequency
        """
        word_freq = TextUtils.calculate_word_frequency(text)
        
        # Sort by frequency (descending)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:max_keywords]
    
    @staticmethod
    def calculate_text_diversity(text: str) -> float:
        """
        Calculate text diversity (unique words / total words).
        
        Args:
            text: Input text
            
        Returns:
            Diversity score between 0.0 and 1.0
        """
        tokens = TextUtils.tokenize_text(text)
        
        if not tokens:
            return 0.0
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        return unique_tokens / total_tokens if total_tokens > 0 else 0.0
    
    @staticmethod
    def calculate_caption_quality(caption: str) -> Dict[str, float]:
        """
        Calculate various quality metrics for a caption.
        
        Args:
            caption: Input caption
            
        Returns:
            Dictionary containing quality metrics
        """
        if not caption:
            return {
                "length_score": 0.0,
                "diversity_score": 0.0,
                "completeness_score": 0.0,
                "overall_score": 0.0
            }
        
        # Length score (optimal length around 10-15 words)
        word_count = len(TextUtils.tokenize_text(caption))
        length_score = 1.0 - abs(word_count - 12.5) / 12.5
        length_score = max(0.0, min(1.0, length_score))
        
        # Diversity score
        diversity_score = TextUtils.calculate_text_diversity(caption)
        
        # Completeness score (check for common caption elements)
        tokens = TextUtils.tokenize_text(caption)
        has_objects = any(word in tokens for word in 
                         ['person', 'people', 'man', 'woman', 'child', 'animal', 'dog', 'cat', 'car', 'house'])
        has_actions = any(word in tokens for word in 
                         ['walking', 'running', 'sitting', 'standing', 'playing', 'eating', 'looking', 'holding'])
        has_scenes = any(word in tokens for word in 
                        ['outdoor', 'indoor', 'street', 'park', 'room', 'building', 'garden', 'kitchen'])
        
        completeness_score = (has_objects + has_actions + has_scenes) / 3.0
        
        # Overall score
        overall_score = (length_score * 0.3 + diversity_score * 0.3 + completeness_score * 0.4)
        
        return {
            "length_score": length_score,
            "diversity_score": diversity_score,
            "completeness_score": completeness_score,
            "overall_score": overall_score
        }
    
    @staticmethod
    def merge_captions(captions: List[str], method: str = "weighted_average") -> str:
        """
        Merge multiple captions into a single caption.
        
        Args:
            captions: List of captions to merge
            method: Merging method ("weighted_average", "concatenation", "best")
            
        Returns:
            Merged caption
        """
        if not captions:
            return ""
        
        if len(captions) == 1:
            return captions[0]
        
        if method == "concatenation":
            return " ".join(captions)
        
        elif method == "best":
            # Select the caption with highest quality score
            best_caption = ""
            best_score = 0.0
            
            for caption in captions:
                quality = TextUtils.calculate_caption_quality(caption)
                if quality["overall_score"] > best_score:
                    best_score = quality["overall_score"]
                    best_caption = caption
            
            return best_caption
        
        elif method == "weighted_average":
            # Weight captions by their quality scores
            weighted_captions = []
            
            for caption in captions:
                quality = TextUtils.calculate_caption_quality(caption)
                weight = quality["overall_score"]
                weighted_captions.append((caption, weight))
            
            # Sort by weight and select top captions
            weighted_captions.sort(key=lambda x: x[1], reverse=True)
            top_captions = [caption for caption, _ in weighted_captions[:3]]
            
            return " ".join(top_captions)
        
        else:
            # Default to concatenation
            return " ".join(captions)
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """
        Extract basic entities from text (simple implementation).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing extracted entities
        """
        tokens = TextUtils.tokenize_text(text)
        
        # Simple entity extraction based on keywords
        people_words = ['person', 'people', 'man', 'woman', 'child', 'boy', 'girl', 'baby']
        animal_words = ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'animal']
        object_words = ['car', 'house', 'building', 'tree', 'flower', 'book', 'phone', 'computer']
        action_words = ['walking', 'running', 'sitting', 'standing', 'playing', 'eating', 'looking', 'holding']
        scene_words = ['outdoor', 'indoor', 'street', 'park', 'room', 'garden', 'kitchen', 'bedroom']
        
        entities = {
            "people": [word for word in tokens if word in people_words],
            "animals": [word for word in tokens if word in animal_words],
            "objects": [word for word in tokens if word in object_words],
            "actions": [word for word in tokens if word in action_words],
            "scenes": [word for word in tokens if word in scene_words]
        }
        
        return entities
    
    @staticmethod
    def calculate_semantic_similarity(text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between texts using word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Semantic similarity score
        """
        if not text1 or not text2:
            return 0.0
        
        # Extract entities from both texts
        entities1 = TextUtils.extract_entities(text1)
        entities2 = TextUtils.extract_entities(text2)
        
        # Calculate similarity for each entity type
        similarities = []
        
        for entity_type in entities1:
            if entity_type in entities2:
                set1 = set(entities1[entity_type])
                set2 = set(entities2[entity_type])
                
                if set1 or set2:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union > 0 else 0.0
                    similarities.append(similarity)
        
        # Return average similarity across all entity types
        return np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def format_caption(caption: str, max_length: int = 100) -> str:
        """
        Format caption for display.
        
        Args:
            caption: Input caption
            max_length: Maximum length for display
            
        Returns:
            Formatted caption
        """
        if not caption:
            return ""
        
        # Clean and normalize
        caption = TextUtils.clean_text(caption)
        
        # Capitalize first letter
        caption = caption.capitalize()
        
        # Ensure it ends with punctuation
        if not caption.endswith(('.', '!', '?')):
            caption += '.'
        
        # Truncate if too long
        if len(caption) > max_length:
            caption = caption[:max_length-3] + "..."
        
        return caption


