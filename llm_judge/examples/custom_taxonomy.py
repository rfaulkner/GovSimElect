#!/usr/bin/env python3
"""
Example showing how to create and use a custom taxonomy.

This example demonstrates:
1. Creating a taxonomy programmatically
2. Saving and loading taxonomies
3. Using different taxonomies for different tasks
"""

import os
import sys
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api_clients import AzureOpenAIClient, OpenRouterClient, DeepSeekClient
from taxonomy import Taxonomy
from judge import LLMJudge
from processors import BatchProcessor


def create_custom_taxonomy():
    """Create a custom taxonomy for email classification."""
    
    # Create empty taxonomy
    taxonomy = Taxonomy()
    
    # Add categories for email classification
    taxonomy.add_category(
        "Support request", 
        "Customer is asking for help with a technical issue, bug report, or needs assistance using a product or service."
    )
    
    taxonomy.add_category(
        "Sales inquiry", 
        "Customer is interested in purchasing a product or service, asking about pricing, features, or availability."
    )
    
    taxonomy.add_category(
        "Billing question", 
        "Customer has questions about their bill, charges, payment methods, or account billing information."
    )
    
    taxonomy.add_category(
        "Feature request", 
        "Customer is suggesting new features, improvements, or enhancements to existing products or services."
    )
    
    taxonomy.add_category(
        "Complaint", 
        "Customer is expressing dissatisfaction, reporting a problem, or filing a formal complaint about service or product quality."
    )
    
    taxonomy.add_category(
        "Praise or feedback", 
        "Customer is providing positive feedback, testimonials, or expressing satisfaction with products or services."
    )
    
    taxonomy.add_category(
        "Account management", 
        "Customer needs help with account settings, password resets, profile updates, or other account-related tasks."
    )
    
    taxonomy.add_category(
        "General inquiry", 
        "Customer is asking general questions about the company, policies, or seeking basic information."
    )
    
    return taxonomy


def setup_api_client():
    """Set up API client - simplified version."""
    
    # Try different providers in order of preference
    if os.getenv('AZURE_API_KEY'):
        return AzureOpenAIClient(
            api_key=os.getenv('AZURE_API_KEY'),
            endpoint=os.getenv('AZURE_ENDPOINT'),
            deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME')
        )
    elif os.getenv('OPENROUTER_API_KEY'):
        return OpenRouterClient(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            model_name=os.getenv('OPENROUTER_MODEL_NAME', 'openai/gpt-4o-mini')
        )
    elif os.getenv('DEEPSEEK_API_KEY'):
        return DeepSeekClient(
            api_key=os.getenv('DEEPSEEK_API_KEY')
        )
    else:
        print("No API credentials found. Please set environment variables.")
        sys.exit(1)


def main():
    """Main example function."""
    
    print("=== Custom Taxonomy Example ===\n")
    
    # 1. Create custom taxonomy
    print("1. Creating custom email classification taxonomy...")
    taxonomy = create_custom_taxonomy()
    print(f"✓ Created taxonomy with {len(taxonomy)} categories")
    
    # Save the taxonomy for future use
    taxonomy_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'email_taxonomy.json')
    taxonomy.save_to_json(taxonomy_path)
    print(f"✓ Saved taxonomy to {taxonomy_path}")
    
    # Show the formatted taxonomy
    print("\nTaxonomy preview:")
    print(taxonomy.get_formatted_taxonomy())
    
    # 2. Set up API client
    print("2. Setting up API client...")
    api_client = setup_api_client()
    
    # 3. Create judge with custom taxonomy
    print("3. Creating LLM judge with custom taxonomy...")
    judge = LLMJudge(api_client, taxonomy, temperature=0)
    
    # 4. Test with email-like examples
    print("4. Testing with email examples:\n")
    
    email_examples = [
        "Hi, I'm having trouble logging into my account. I keep getting an error message when I enter my password. Can you help?",
        "I'm interested in upgrading to your premium plan. What features are included and what's the monthly cost?",
        "I was charged twice for my subscription this month. Can you please refund the duplicate charge?",
        "It would be great if you could add dark mode to your mobile app. Many users have been requesting this feature.",
        "I'm extremely disappointed with the service quality. My order was delayed by two weeks without any communication.",
        "Just wanted to say thanks for the excellent customer support. The team was very helpful and professional!",
        "I need to update my email address in my profile but I can't find the option in settings. Where is it located?",
        "What are your business hours? Also, do you offer phone support or only email?"
    ]
    
    for i, email in enumerate(email_examples, 1):
        print(f"Email {i}: {email[:80]}...")
        result = judge.classify_text(email)
        
        print(f"  → Classification: {result['justification_type']}")
        print(f"  → Confidence: {result['Confidence']:.2f}")
        print(f"  → Reasoning: {result['Reasoning_behind_classification'][:100]}...\n")
    
    # 5. Demonstrate taxonomy modification
    print("5. Demonstrating taxonomy modification:")
    
    # Add a new category
    taxonomy.add_category(
        "Urgent request", 
        "Customer request that requires immediate attention due to critical business impact or emergency situation."
    )
    print("✓ Added 'Urgent request' category")
    
    # Remove a category
    taxonomy.remove_category("General inquiry")
    print("✓ Removed 'General inquiry' category")
    
    print(f"Updated taxonomy now has {len(taxonomy)} categories")
    
    # Test urgent email
    urgent_email = "URGENT: Our production system is down and we need immediate assistance. This is costing us thousands per minute!"
    
    # Create new judge with updated taxonomy
    updated_judge = LLMJudge(api_client, taxonomy, temperature=0)
    result = updated_judge.classify_text(urgent_email)
    
    print(f"\nUrgent email classification: {result['justification_type']}")
    print(f"Confidence: {result['Confidence']:.2f}")
    
    # 6. Load the cooperation taxonomy and compare
    print("\n6. Comparing with cooperation taxonomy:")
    
    coop_taxonomy_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'cooperation_taxonomy.json')
    if os.path.exists(coop_taxonomy_path):
        coop_taxonomy = Taxonomy.from_json_file(coop_taxonomy_path)
        print(f"Cooperation taxonomy has {len(coop_taxonomy)} categories")
        print("Sample categories:", ", ".join(list(coop_taxonomy.get_categories())[:3]))
        
        # Test the same urgent email with cooperation taxonomy
        coop_judge = LLMJudge(api_client, coop_taxonomy, temperature=0)
        coop_result = coop_judge.classify_text(urgent_email)
        
        print(f"Same email with cooperation taxonomy: {coop_result['justification_type']}")
        print("(Shows how different taxonomies produce different classifications)")
    
    print(f"\nAPI Statistics:")
    print(f"Total cost: ${api_client.get_total_cost():.4f}")
    print(f"Success rate: {api_client.get_success_rate():.1f}%")
    
    print("\n=== Custom taxonomy example completed! ===")


if __name__ == "__main__":
    main()