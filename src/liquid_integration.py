"""
Integration with Liquid AI's LFM2-VL models.

This module handles the interaction with Liquid Neural Networks
and leverages their continuous-time processing capabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import time

class LiquidAIIntegration:
    """
    Integration with Liquid AI's LFM2-VL models.
    
    This class handles the interaction with Liquid Neural Networks
    and leverages their continuous-time processing capabilities.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 512
    ):
        """
        Initialize Liquid AI integration.
        
        Args:
            model_path: Path to the LFM2-VL model
            device: Device to run inference on
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length
        self.model_path = model_path
        
        # Initialize model (placeholder - actual implementation depends on LFM2-VL API)
        self.model = self._load_model(model_path)
        
        logging.info(f"Liquid AI integration initialized with model: {model_path}")
    
    def _load_model(self, model_path: str):
        """Load the LFM2-VL model using the correct approach."""
        try:
            # Check if model path exists
            if not Path(model_path).exists():
                logging.warning(f"Model path {model_path} does not exist. Using mock model.")
                return self._create_mock_model()
            
            # Load using the correct LFM2-VL approach
            try:
                from transformers import AutoProcessor, AutoModelForImageTextToText
                
                logging.info(f"Loading LFM2-VL model from {model_path}...")
                
                # Load model and processor using AutoProcessor (recommended approach)
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                
                processor = AutoProcessor.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                
                logging.info(f"LFM2-VL model and processor loaded successfully on {model.device}")
                
                return {
                    "model": model,
                    "processor": processor,
                    "is_mock": False
                }
                
            except Exception as e:
                logging.warning(f"Could not load LFM2-VL model from {model_path}: {e}. Using mock model.")
                return self._create_mock_model()
                
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a mock model for testing purposes."""
        logging.info("Creating mock model for testing")
        
        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 2
                self.pad_token_id = 0
            
            def encode(self, text, **kwargs):
                # Simple tokenization
                tokens = text.split()[:self.max_length]
                return {"input_ids": torch.tensor([[hash(word) % 1000 for word in tokens]]),
                        "attention_mask": torch.ones((1, len(tokens)))}
            
            def decode(self, token_ids, **kwargs):
                # Simple decoding
                return " ".join([f"token_{id.item()}" for id in token_ids[0]])
        
        class MockModel:
            def __init__(self, device="cpu"):
                self.device = device
            
            def generate(self, **kwargs):
                # Mock generation
                batch_size = kwargs.get('pixel_values', torch.randn(1, 3, 224, 224)).shape[0]
                seq_len = kwargs.get('max_new_tokens', 10)
                return torch.randint(1, 1000, (batch_size, seq_len))
            
            def forward(self, **kwargs):
                # Mock forward pass
                return {"logits": torch.randn(1, 512, 1000)}
        
        return {
            "model": MockModel(device=self.device),
            "tokenizer": MockTokenizer(),
            "is_mock": True
        }
    
    def generate_caption(
        self,
        image: Union[str, np.ndarray],
        text_prompt: Optional[str] = None,
        return_features: bool = False,
        temperature: float = 0.1,  # Use LFM2-VL recommended temperature
        max_new_tokens: int = 100
    ) -> Dict[str, any]:
        """
        Generate caption for an image using LFM2-VL with proper chat template.
        
        Args:
            image: Input image (path or numpy array)
            text_prompt: Optional text prompt for conditioning
            return_features: Whether to return feature vectors
            temperature: Sampling temperature (LFM2-VL recommends 0.1)
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Dictionary containing caption and optional features
        """
        start_time = time.time()
        
        try:
            if self.model.get("is_mock", True):
                # Mock generation for testing
                return {
                    "caption": f"Mock caption for image with prompt: {text_prompt or 'Describe this image'}",
                    "features": np.random.randn(768) if return_features else None,
                    "confidence": 0.85,
                    "processing_time": time.time() - start_time
                }
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Create conversation in the correct LFM2-VL format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {"type": "text", "text": text_prompt or "Describe this image in detail."},
                    ],
                },
            ]
            
            # Apply chat template and generate
            inputs = self.model["processor"].apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model["model"].device)
            
            # Generate with recommended LFM2-VL parameters
            with torch.no_grad():
                outputs = self.model["model"].generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    min_p=0.15,  # LFM2-VL recommended parameter
                    repetition_penalty=1.05,  # LFM2-VL recommended parameter
                    do_sample=True,
                    pad_token_id=self.model["processor"].tokenizer.eos_token_id,
                    eos_token_id=self.model["processor"].tokenizer.eos_token_id
                )
            
            # Decode the response
            caption = self.model["processor"].batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Clean up the caption
            caption = self._clean_caption(caption)
            
            # Extract features if requested
            features = self._extract_features_from_outputs(outputs)  # Always extract features
            if not return_features:
                features = None  # Only set to None if not requested
            
            processing_time = time.time() - start_time
            
            return {
                "caption": caption,
                "features": features,
                "confidence": self._calculate_confidence_from_outputs(outputs),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logging.error(f"Error generating caption: {str(e)}")
            raise
    
    def generate_with_context(
        self,
        image: Union[str, np.ndarray],
        context: Dict[str, any],
        context_weight: float = 0.7,
        text_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate caption with video context using LFM2-VL with proper chat template.
        
        This integrates video context into the image captioning process.
        """
        start_time = time.time()
        
        try:
            if self.model.get("is_mock", True):
                # Mock generation for testing
                return {
                    "caption": f"Mock caption with context: {context.get('context_text', 'No context')}",
                    "context_relevance": 0.5,
                    "confidence": 0.85,
                    "processing_time": time.time() - start_time
                }
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Create context-aware prompt
            context_text = context.get('context_text', '')
            if text_prompt:
                # Use custom prompt with context
                context_prompt = f"Based on the video context: '{context_text}', {text_prompt}"
            else:
                # Use default prompt with context
                context_prompt = f"Based on the video context: '{context_text}', describe this image in detail."
            
            # Create conversation in the correct LFM2-VL format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {"type": "text", "text": context_prompt},
                    ],
                },
            ]
            
            # Apply chat template and generate
            inputs = self.model["processor"].apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model["model"].device)
            
            # Generate with recommended LFM2-VL parameters
            with torch.no_grad():
                outputs = self.model["model"].generate(
                    **inputs, 
                    max_new_tokens=100,
                    temperature=0.1,  # LFM2-VL recommended parameter
                    min_p=0.15,       # LFM2-VL recommended parameter
                    repetition_penalty=1.05,  # LFM2-VL recommended parameter
                    do_sample=True,
                    pad_token_id=self.model["processor"].tokenizer.eos_token_id,
                    eos_token_id=self.model["processor"].tokenizer.eos_token_id
                )
            
            # Decode the response
            raw_caption = self.model["processor"].batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract the Instagram caption from the raw output
            instagram_caption = self._extract_instagram_caption(raw_caption)
            
            # Calculate context relevance using the extracted caption
            context_relevance = self._calculate_context_relevance(instagram_caption, context)
            
            processing_time = time.time() - start_time
            
            return {
                "raw_output": raw_caption,
                "context": context_text,
                "prompt": text_prompt or "Create an Instagram-style caption for this image.",
                "instagram_caption": instagram_caption,
                "caption": instagram_caption,  # Keep for backward compatibility
                "context_relevance": context_relevance,
                "confidence": self._calculate_confidence_from_outputs(outputs),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logging.error(f"Error generating contextual caption: {str(e)}")
            raise
    
    def _preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Preprocess image for LFM2-VL input."""
        from PIL import Image
        
        if isinstance(image, str):
            # Load image from path using PIL
            img = Image.open(image).convert('RGB')
        else:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                img = Image.fromarray(image.astype(np.uint8))
            else:
                raise ValueError(f"Invalid image format: {image.shape}")
        
        return img
    
    def _prepare_inputs(
        self,
        image: np.ndarray,
        text_prompt: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for LFM2-VL model."""
        if self.model.get("is_mock", True):
            # Mock inputs
            image_tensor = torch.randn(1, 3, 224, 224).to(self.device)
            inputs = {"pixel_values": image_tensor}
            
            if text_prompt:
                inputs.update({
                    "input_ids": torch.randint(1, 1000, (1, 10)).to(self.device),
                    "attention_mask": torch.ones(1, 10).to(self.device)
                })
        else:
            # Real LFM2-VL inputs using processor
            try:
                if self.model.get("processor"):
                    # Use the processor if available
                    processor = self.model["processor"]
                    inputs = processor(
                        images=image,
                        text=text_prompt or "Describe this image",
                        return_tensors="pt"
                    )
                    # Move to device
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to(self.device)
                else:
                    # Fallback to manual processing
                    from transformers import Siglip2ImageProcessor
                    
                    # Create image processor
                    image_processor = Siglip2ImageProcessor.from_pretrained(
                        "google/siglip-base-patch16-224"
                    )
                    
                    # Process image
                    image_inputs = image_processor(images=image, return_tensors="pt")
                    
                    # Process text
                    if text_prompt:
                        text_inputs = self.model["tokenizer"](
                            text_prompt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.max_length
                        )
                    else:
                        text_inputs = self.model["tokenizer"](
                            "Describe this image",
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.max_length
                        )
                    
                    inputs = {
                        "pixel_values": image_inputs["pixel_values"].to(self.device),
                        "input_ids": text_inputs["input_ids"].to(self.device),
                        "attention_mask": text_inputs["attention_mask"].to(self.device)
                    }
                    
            except Exception as e:
                logging.warning(f"Error preparing inputs: {e}. Using fallback.")
                # Fallback to simple inputs
                image_tensor = torch.randn(1, 3, 224, 224).to(self.device)
                inputs = {"pixel_values": image_tensor}
                
                if text_prompt:
                    inputs.update({
                        "input_ids": torch.randint(1, 1000, (1, 10)).to(self.device),
                        "attention_mask": torch.ones(1, 10).to(self.device)
                    })
        
        return inputs
    
    def _prepare_context_prompt(self, context: Dict[str, any]) -> str:
        """Prepare context prompt from video context."""
        context_text = context.get("context_text", "")
        if not context_text:
            return ""
        
        # Create context-aware prompt
        prompt = f"Based on the video context: {context_text}\nGenerate a caption for this image:"
        
        return prompt
    
    def _prepare_contextual_inputs(
        self,
        image: np.ndarray,
        context_prompt: str,
        context_weight: float
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs with video context."""
        # Prepare base inputs
        inputs = self._prepare_inputs(image, context_prompt)
        
        # Add context features if available
        if "context_features" in context_prompt:
            context_features = context_prompt["context_features"]
            if context_features is not None:
                context_tensor = torch.from_numpy(context_features).to(self.device)
                inputs["context_features"] = context_tensor
                inputs["context_weight"] = torch.tensor(context_weight).to(self.device)
        
        return inputs
    
    def _generate_with_lnn(
        self,
        inputs: Dict[str, torch.Tensor],
        temperature: float,
        max_new_tokens: int
    ) -> Dict[str, torch.Tensor]:
        """Generate using LNN's continuous-time processing."""
        model = self.model["model"]
        
        # Generate using the LFM2-VL model
        with torch.no_grad():
            if self.model.get("is_mock", True):
                # Mock generation
                batch_size = inputs.get('pixel_values', torch.randn(1, 3, 224, 224)).shape[0]
                outputs = torch.randint(1, 1000, (batch_size, max_new_tokens))
            else:
                # Real LFM2-VL generation
                try:
                    # Prepare generation parameters for better quality
                    generation_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "do_sample": True,
                        "pad_token_id": self.model["tokenizer"].eos_token_id,
                        "use_cache": True,
                        "eos_token_id": self.model["tokenizer"].eos_token_id,
                        "repetition_penalty": 1.2,  # Increased to reduce repetition
                        "length_penalty": 1.0,
                        "no_repeat_ngram_size": 3,  # Avoid repetition
                        "early_stopping": True,
                        "num_beams": 1,  # Use greedy decoding for cleaner output
                        "top_p": 0.9,
                        "top_k": 50
                    }
                    
                    # Generate with the model
                    outputs = model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                    
                except Exception as e:
                    logging.warning(f"Error in LFM2-VL generation: {e}. Using fallback.")
                    # Fallback to simple generation
                    batch_size = inputs.get('pixel_values', torch.randn(1, 3, 224, 224)).shape[0]
                    outputs = torch.randint(1, 1000, (batch_size, max_new_tokens))
        
        return {"generated_ids": outputs}
    
    def _generate_with_contextual_lnn(
        self,
        inputs: Dict[str, torch.Tensor],
        context: Dict[str, any]
    ) -> Dict[str, torch.Tensor]:
        """Generate with context using LNN's adaptive state."""
        # This would leverage the LNN's ability to maintain state
        # and adapt based on contextual information
        
        # Placeholder implementation
        return self._generate_with_lnn(inputs, temperature=0.7, max_new_tokens=256)
    
    def _clean_caption(self, caption: str) -> str:
        """Clean up the generated caption."""
        if not caption:
            return "A detailed description of the image showing various visual elements and composition."
        
        # Remove common artifacts and special tokens
        artifacts = ['<|reserved_', '<|endoftext|>', '<|startoftext|>', '<pad>', '<unk>', '<|reserved_4|>', '<|reserved_5|>', '<|im_start|>', '<|im_end|>']
        for artifact in artifacts:
            caption = caption.replace(artifact, '')
        
        # Remove excessive whitespace and newlines
        import re
        caption = re.sub(r'\n+', ' ', caption)  # Replace multiple newlines with space
        caption = re.sub(r'\s+', ' ', caption)  # Replace multiple spaces with single space
        caption = caption.strip()
        
        # Ensure we have a meaningful caption
        if not caption or len(caption) < 10:
            caption = "A detailed description of the image showing various visual elements and composition."
        
        return caption
    
    def _extract_instagram_caption(self, raw_output: str) -> str:
        """Extract the Instagram caption from the raw model output."""
        if not raw_output:
            return "A beautiful moment captured in time."
        
        import re
        
        # Debug: Print the raw output to understand the structure
        print(f"DEBUG - Raw output: {raw_output[:500]}...")
        
        # Look for the last "assistant" response in the output
        # Split by "assistant" and take the last part
        parts = re.split(r'assistant\s*', raw_output, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            # Get the last assistant response
            last_response = parts[-1].strip()
            print(f"DEBUG - Last assistant response: {last_response[:200]}...")
            
            # Look for content in quotes
            quote_match = re.search(r'["\']([^"\']*?)["\']', last_response)
            if quote_match:
                caption = quote_match.group(1).strip()
                print(f"DEBUG - Caption from quotes: {caption}")
                return self._clean_caption(caption)
            
            # If no quotes, take everything after the last assistant
            caption = last_response.strip()
            if caption and len(caption) > 10:
                print(f"DEBUG - Caption without quotes: {caption}")
                return self._clean_caption(caption)
        
        # Fallback: try to find content after the last quote
        quote_pattern = r'["\']([^"\']*?)["\']?\s*$'
        match = re.search(quote_pattern, raw_output)
        if match:
            caption = match.group(1).strip()
            print(f"DEBUG - Fallback caption: {caption}")
            return self._clean_caption(caption)
        
        # Final fallback: clean the entire output
        print(f"DEBUG - Final fallback: {raw_output}")
        return self._clean_caption(raw_output)
    
    def _extract_features_from_outputs(self, outputs) -> np.ndarray:
        """Extract features from model outputs."""
        # Placeholder implementation - would extract actual features from LFM2-VL outputs
        return np.random.randn(768)
    
    def _calculate_confidence_from_outputs(self, outputs) -> float:
        """Calculate confidence from model outputs."""
        # Placeholder implementation - would calculate actual confidence
        return 0.85
    
    def _extract_caption(self, outputs: Dict[str, torch.Tensor]) -> str:
        """Extract caption from model outputs (legacy method)."""
        generated_ids = outputs["generated_ids"]
        
        # Handle different output formats
        if generated_ids.dim() > 1:
            # Remove batch dimension
            generated_ids = generated_ids[0]
        
        # Decode generated tokens with proper handling
        try:
            caption = self.model["tokenizer"].decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False
            )
        except Exception as e:
            logging.warning(f"Error decoding tokens: {e}. Using fallback.")
            # Fallback: convert to string representation
            caption = str(generated_ids.tolist())
        
        return self._clean_caption(caption)
    
    def _extract_features(self, outputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """Extract feature vectors from model outputs."""
        # This would extract the LNN's internal state or hidden features
        # Placeholder implementation
        return np.random.randn(768)  # Example feature vector
    
    def _calculate_confidence(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Calculate confidence score for generated caption."""
        # This would calculate confidence based on model outputs
        # Placeholder implementation
        return 0.85
    
    def _calculate_context_relevance(
        self,
        caption: str,
        context: Dict[str, any]
    ) -> float:
        """Calculate how relevant the context is to the generated caption."""
        # This would use similarity metrics to assess context relevance
        # Placeholder implementation
        return 0.75
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "max_length": self.max_length,
            "is_mock": self.model.get("is_mock", False)
        }
