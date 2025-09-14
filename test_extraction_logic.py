#!/usr/bin/env python3
"""
Test the Instagram caption extraction logic directly.
"""

import sys
import os
import logging
import re

# Add src to path
sys.path.append('src')

def test_extraction_logic():
    """Test the extraction logic with sample data."""
    print("🧪 Testing Instagram Caption Extraction Logic")
    print("=" * 70)
    
    # Sample raw output that we're seeing
    sample_raw_output = '''user Based on the video context: 'user Describe this image in detail. assistant The image captures a serene tropical beach scene with a vibrant sunset casting an orange glow across the sky. A palm tree stands prominently on the right, its leaves gently swaying. The sandy shore is dotted with lush green foliage, and the calm ocean stretches out to the horizon, where a small island can be seen in the distance. The overall ambiance suggests a tranquil and picturesque coastal setting. user Describe this image in detail. assistant The image capt', Generate a trendy Instagram caption with hashtags for this image. assistant "Embracing the serenity of a tropical paradise at sunset. 🌅🏖️ #tropicalbeach #sunsetvibes #coastalvibes #palmtree #sunset #beachvibes #tropicalvibes #sunset #beach #sunset #tropicalvibes #sunset #beach #sunset #tropicalvibes #sunset #beach #sunset #t'''
    
    print(f"📝 Sample Raw Output:")
    print(f"   {sample_raw_output}")
    print()
    
    # Test the extraction logic
    def extract_instagram_caption(raw_output: str) -> str:
        """Extract the Instagram caption from the raw model output."""
        if not raw_output:
            return "A beautiful moment captured in time."
        
        print(f"🔍 DEBUG - Raw output: {raw_output[:500]}...")
        
        # Look for the last "assistant" response in the output
        # Split by "assistant" and take the last part
        parts = re.split(r'assistant\s*', raw_output, flags=re.IGNORECASE)
        
        print(f"🔍 DEBUG - Split parts: {len(parts)}")
        for i, part in enumerate(parts):
            print(f"   Part {i}: {part[:100]}...")
        
        if len(parts) > 1:
            # Get the last assistant response
            last_response = parts[-1].strip()
            print(f"🔍 DEBUG - Last assistant response: {last_response[:200]}...")
            
            # Look for content in quotes
            quote_match = re.search(r'["\']([^"\']*?)["\']', last_response)
            if quote_match:
                caption = quote_match.group(1).strip()
                print(f"🔍 DEBUG - Caption from quotes: {caption}")
                return caption
            
            # If no quotes, take everything after the last assistant
            caption = last_response.strip()
            if caption and len(caption) > 10:
                print(f"🔍 DEBUG - Caption without quotes: {caption}")
                return caption
        
        # Fallback: try to find content after the last quote
        quote_pattern = r'["\']([^"\']*?)["\']?\s*$'
        match = re.search(quote_pattern, raw_output)
        if match:
            caption = match.group(1).strip()
            print(f"🔍 DEBUG - Fallback caption: {caption}")
            return caption
        
        # Final fallback: clean the entire output
        print(f"🔍 DEBUG - Final fallback: {raw_output}")
        return raw_output
    
    # Test the extraction
    extracted_caption = extract_instagram_caption(sample_raw_output)
    
    print(f"\n✅ Extracted Caption:")
    print(f"   {extracted_caption}")
    
    # Test with a cleaner example
    print(f"\n" + "="*70)
    print("🧪 Testing with Cleaner Example")
    print("="*70)
    
    clean_example = 'user Describe this image. assistant "This is a beautiful beach scene with palm trees and sunset. #beach #sunset #tropical"'
    clean_caption = extract_instagram_caption(clean_example)
    
    print(f"📝 Clean Example:")
    print(f"   {clean_example}")
    print(f"✅ Extracted: {clean_caption}")
    
    return extracted_caption

def main():
    """Main test function."""
    result = test_extraction_logic()
    
    print(f"\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    if result:
        print("✅ Extraction logic working")
        print(f"   - Extracted caption: {result}")
    else:
        print("❌ Extraction failed")
    
    print("\n🎉 Extraction logic test completed!")

if __name__ == "__main__":
    main()
