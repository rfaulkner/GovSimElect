"""
Core LLM Judge functionality for text classification using custom taxonomies.
"""

import json
import re
from typing import Dict, List, Optional, Any


class LLMJudge:
    """
    Main class for performing LLM-based text classification.
    """
    
    def __init__(self, api_client, taxonomy, temperature: float = 0):
        """
        Initialize the LLM Judge.
        
        Args:
            api_client: API client instance (OpenAI, Azure, etc.)
            taxonomy: Taxonomy instance with categories and definitions
            temperature: Temperature for LLM generation (0 = deterministic)
        """
        self.api_client = api_client
        self.taxonomy = taxonomy
        self.temperature = temperature
    
    def classify_text(self, text: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Classify a single text using the LLM judge.
        
        Args:
            text: Text to classify
            max_tokens: Maximum tokens for LLM response
            
        Returns:
            Dictionary with classification results
        """
        prompt = self._build_classification_prompt(text)
        
        try:
            response = self.api_client.send_request(
                model_name=self.api_client.deployment_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            
            # Extract JSON from response
            json_result = self._extract_json_from_response(response)
            
            if json_result:
                result = json.loads(json_result)
                # Ensure confidence is float
                result["Confidence"] = float(result.get("Confidence", 0))
                return result
            else:
                # Return raw response if JSON extraction fails
                return {
                    "Reasoning_behind_classification": response,
                    "Confidence": 0.0,
                    "justification_type": "Other"
                }
                
        except Exception as e:
            return {
                "Reasoning_behind_classification": f"Error in analysis: {str(e)}",
                "Confidence": 0.0,
                "justification_type": "Failed classification"
            }
    
    def _build_classification_prompt(self, text: str) -> str:
        """Build the classification prompt using the taxonomy."""
        
        # Get taxonomy sections
        taxonomy_text = self.taxonomy.get_formatted_taxonomy()
        
        prompt = f"""Analyze the following text and categorize the decision-making strategy used.
You may choose one, multiple or none of the classes. If none apply, classify as other.

{taxonomy_text}

Text to analyze:
\"\"\"{text}\"\"\"

IMPORTANT: Your response MUST be in valid JSON format EXACTLY as shown below. Do not include any explanatory text outside of the JSON structure.

Example of the required JSON format:
{{
  "Reasoning_behind_classification": "Explanation of your classification reasoning",
  "Confidence": 0.85,
  "justification_type": "Category1, Category2"
}}

Ensure that:
1. Your JSON is properly formatted with no trailing commas
2. "Confidence" is a decimal number between 0 and 1, not a string
3. For multiple justification types, list them as a comma-separated string
4. Don't include any text outside the JSON object
"""
        return prompt
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Extract JSON object from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            JSON string if found, None otherwise
        """
        try:
            # Find JSON boundaries
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    # Validate JSON
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass
            
            # Try fixing single quotes
            fixed_response = response.replace("'", '"')
            json_start = fixed_response.find('{')
            json_end = fixed_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = fixed_response[json_start:json_end]
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass
            
            return None
            
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            return None