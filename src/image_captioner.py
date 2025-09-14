"""
Image captioning with video context integration.

This module handles the generation of image captions using video context
and leverages Liquid AI's adaptive state capabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import time

from .liquid_integration import LiquidAIIntegration

class ImageCaptioner:
    """
    Image captioner that integrates video context using Liquid AI.
    
    This class handles the generation of image captions with video context
    by leveraging the LNN's adaptive state and continuous-time processing.
    """
    
    def __init__(
        self,
        liquid_integration: LiquidAIIntegration,
        context_weight: float = 0.7
    ):
        """
        Initialize image captioner.
        
        Args:
            liquid_integration: Liquid AI integration instance
            context_weight: Weight of video context in caption generation
        """
        self.liquid_integration = liquid_integration
        self.context_weight = context_weight
        
        logging.info(f"ImageCaptioner initialized with context weight: {context_weight}")
    
    def generate_caption(
        self,
        image_path: str,
        video_context: Dict[str, any],
        context_weight: Optional[float] = None,
        text_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate image caption with video context.
        
        Args:
            image_path: Path to the input image
            video_context: Video context extracted from frames
            context_weight: Override default context weight
            text_prompt: Custom text prompt for caption generation
            
        Returns:
            Dictionary containing caption and metadata
        """
        start_time = time.time()
        
        try:
            # Use the specified context weight or default
            effective_context_weight = context_weight or self.context_weight
            
            # Generate caption with context using LNN's adaptive state
            result = self.liquid_integration.generate_with_context(
                image=image_path,
                context=video_context,
                context_weight=effective_context_weight,
                text_prompt=text_prompt
            )
            
            # Calculate additional metrics
            processing_time = time.time() - start_time
            context_relevance = self._calculate_context_relevance(
                result["caption"], video_context
            )
            
            return {
                "raw_output": result.get("raw_output", ""),
                "context": result.get("context", ""),
                "prompt": result.get("prompt", ""),
                "instagram_caption": result.get("instagram_caption", result["caption"]),
                "caption": result["caption"],  # Keep for backward compatibility
                "confidence": result["confidence"],
                "context_relevance": context_relevance,
                "processing_time": processing_time,
                "context_weight_used": effective_context_weight
            }
            
        except Exception as e:
            logging.error(f"Error generating image caption: {str(e)}")
            raise
    
    def generate_caption_without_context(
        self,
        image_path: str,
        text_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate image caption without video context.
        
        Args:
            image_path: Path to the input image
            text_prompt: Optional text prompt for conditioning
            
        Returns:
            Dictionary containing caption and metadata
        """
        start_time = time.time()
        
        try:
            # Generate caption without context
            result = self.liquid_integration.generate_caption(
                image=image_path,
                text_prompt=text_prompt,
                return_features=True
            )
            
            processing_time = time.time() - start_time
            
            return {
                "caption": result["caption"],
                "confidence": result["confidence"],
                "processing_time": processing_time,
                "context_weight_used": 0.0
            }
            
        except Exception as e:
            logging.error(f"Error generating caption without context: {str(e)}")
            raise
    
    def compare_captions(
        self,
        image_path: str,
        video_context: Dict[str, any],
        context_weights: List[float] = [0.0, 0.3, 0.5, 0.7, 1.0]
    ) -> Dict[str, any]:
        """
        Compare captions generated with different context weights.
        
        Args:
            image_path: Path to the input image
            video_context: Video context extracted from frames
            context_weights: List of context weights to test
            
        Returns:
            Dictionary containing comparison results
        """
        results = {}
        
        for weight in context_weights:
            try:
                if weight == 0.0:
                    # Generate without context
                    result = self.generate_caption_without_context(image_path)
                else:
                    # Generate with context
                    result = self.generate_caption(
                        image_path, video_context, context_weight=weight
                    )
                
                results[f"weight_{weight}"] = {
                    "caption": result["caption"],
                    "confidence": result["confidence"],
                    "context_relevance": result.get("context_relevance", 0.0),
                    "processing_time": result["processing_time"]
                }
                
            except Exception as e:
                logging.error(f"Error generating caption with weight {weight}: {str(e)}")
                results[f"weight_{weight}"] = {"error": str(e)}
        
        return results
    
    def _calculate_context_relevance(
        self,
        caption: str,
        video_context: Dict[str, any]
    ) -> float:
        """
        Calculate how relevant the video context is to the generated caption.
        
        Args:
            caption: Generated caption
            video_context: Video context information
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            context_text = video_context.get("context_text", "")
            if not context_text or not caption:
                return 0.0
            
            # Simple word overlap calculation
            caption_words = set(caption.lower().split())
            context_words = set(context_text.lower().split())
            
            if not caption_words or not context_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(caption_words.intersection(context_words))
            union = len(caption_words.union(context_words))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Also consider temporal consistency
            temporal_consistency = video_context.get("temporal_consistency", 0.5)
            
            # Combine metrics
            relevance = (jaccard_similarity * 0.7) + (temporal_consistency * 0.3)
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating context relevance: {str(e)}")
            return 0.5  # Default moderate relevance
    
    def analyze_caption_quality(
        self,
        caption: str,
        video_context: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Analyze the quality of the generated caption.
        
        Args:
            caption: Generated caption
            video_context: Video context information
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            # Basic caption metrics
            word_count = len(caption.split())
            char_count = len(caption)
            
            # Context relevance
            context_relevance = self._calculate_context_relevance(caption, video_context)
            
            # Caption diversity (unique words)
            unique_words = len(set(caption.lower().split()))
            diversity_ratio = unique_words / word_count if word_count > 0 else 0.0
            
            # Check for common caption patterns
            has_objects = any(word in caption.lower() for word in 
                            ['person', 'people', 'man', 'woman', 'child', 'animal', 'dog', 'cat'])
            has_actions = any(word in caption.lower() for word in 
                            ['walking', 'running', 'sitting', 'standing', 'playing', 'eating'])
            has_scenes = any(word in caption.lower() for word in 
                           ['outdoor', 'indoor', 'street', 'park', 'room', 'building'])
            
            # Quality score based on multiple factors
            quality_score = (
                min(word_count / 10, 1.0) * 0.2 +  # Length appropriateness
                context_relevance * 0.3 +           # Context relevance
                diversity_ratio * 0.2 +              # Word diversity
                (1.0 if has_objects else 0.5) * 0.1 +  # Object detection
                (1.0 if has_actions else 0.5) * 0.1 +  # Action detection
                (1.0 if has_scenes else 0.5) * 0.1     # Scene detection
            )
            
            return {
                "word_count": word_count,
                "char_count": char_count,
                "unique_words": unique_words,
                "diversity_ratio": diversity_ratio,
                "context_relevance": context_relevance,
                "has_objects": has_objects,
                "has_actions": has_actions,
                "has_scenes": has_scenes,
                "quality_score": quality_score
            }
            
        except Exception as e:
            logging.error(f"Error analyzing caption quality: {str(e)}")
            return {
                "word_count": 0,
                "char_count": 0,
                "unique_words": 0,
                "diversity_ratio": 0.0,
                "context_relevance": 0.0,
                "has_objects": False,
                "has_actions": False,
                "has_scenes": False,
                "quality_score": 0.0
            }
    
    def generate_multiple_captions(
        self,
        image_path: str,
        video_context: Dict[str, any],
        num_captions: int = 3,
        context_weight: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """
        Generate multiple captions for the same image with slight variations.
        
        Args:
            image_path: Path to the input image
            video_context: Video context extracted from frames
            num_captions: Number of captions to generate
            context_weight: Override default context weight
            
        Returns:
            List of caption results
        """
        captions = []
        
        for i in range(num_captions):
            try:
                # Slightly vary the context weight for diversity
                varied_weight = (context_weight or self.context_weight) + (i * 0.1)
                varied_weight = min(varied_weight, 1.0)
                
                result = self.generate_caption(
                    image_path, video_context, context_weight=varied_weight
                )
                
                result["variation_index"] = i
                result["varied_weight"] = varied_weight
                captions.append(result)
                
            except Exception as e:
                logging.error(f"Error generating caption variation {i}: {str(e)}")
                captions.append({
                    "variation_index": i,
                    "error": str(e)
                })
        
        return captions
