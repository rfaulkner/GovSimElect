#!/usr/bin/env python3
"""
Basic usage example for the LLM Judge package.

This example shows how to:
1. Set up an API client
2. Load a taxonomy
3. Create an LLM judge
4. Classify individual texts
5. Process multiple texts in batch
"""

import os
import sys
import pandas as pd

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api_clients import AzureOpenAIClient, OpenRouterClient, DeepSeekClient
from taxonomy import Taxonomy
from judge import LLMJudge
from processors import BatchProcessor


def setup_api_client():
    """
    Set up your API client. Choose one of the following:
    """
    
    # Option 1: Azure OpenAI
    if os.getenv('AZURE_API_KEY'):
        client = AzureOpenAIClient(
            api_key=os.getenv('AZURE_API_KEY'),
            endpoint=os.getenv('AZURE_ENDPOINT'),
            deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'),
            api_version='2024-12-01-preview'
        )
        return client
    
    # Option 2: OpenRouter
    elif os.getenv('OPENROUTER_API_KEY'):
        client = OpenRouterClient(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            model_name=os.getenv('OPENROUTER_MODEL_NAME', 'openai/gpt-4o-mini')
        )
        return client
    
    # Option 3: DeepSeek
    elif os.getenv('DEEPSEEK_API_KEY'):
        client = DeepSeekClient(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            model_name='deepseek-chat'
        )
        return client
    
    else:
        print("No API credentials found in environment variables.")
        print("Please set one of the following:")
        print("- AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME")
        print("- OPENROUTER_API_KEY, OPENROUTER_MODEL_NAME")
        print("- DEEPSEEK_API_KEY")
        sys.exit(1)


def main():
    """Main example function."""
    
    print("=== LLM Judge Basic Usage Example ===\n")
    
    # 1. Set up API client
    print("1. Setting up API client...")
    api_client = setup_api_client()
    
    # Verify connection
    success, message = api_client.verify_connection()
    if not success:
        print(f"API connection failed: {message}")
        return
    print(f"✓ API connection successful: {message}\n")
    
    # 2. Load taxonomy
    print("2. Loading taxonomy...")
    taxonomy_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'example_taxonomy.json')
    taxonomy = Taxonomy.from_json_file(taxonomy_path)
    print(f"✓ Loaded taxonomy with {len(taxonomy)} categories")
    print("Categories:", ", ".join(taxonomy.get_categories()))
    print()
    
    # 3. Create LLM judge
    print("3. Creating LLM judge...")
    judge = LLMJudge(api_client, taxonomy, temperature=0)
    print("✓ LLM judge created\n")
    
    # 4. Single text classification example
    print("4. Single text classification example:")
    
    example_texts = [
        "I'm really disappointed with this product. It broke after just one week!",
        "This is absolutely amazing! Best purchase I've ever made.",
        "Can you tell me what the return policy is for this item?",
        "The package arrived on time and was well packaged. The item matches the description.",
        "I love the design but hate the functionality. It's a mixed bag really."
    ]
    
    for i, text in enumerate(example_texts, 1):
        print(f"\nExample {i}: '{text}'")
        result = judge.classify_text(text)
        print(f"Classification: {result['justification_type']}")
        print(f"Confidence: {result['Confidence']:.2f}")
        print(f"Reasoning: {result['Reasoning_behind_classification'][:100]}...")
    
    print("\n" + "="*60 + "\n")
    
    # 5. Batch processing example
    print("5. Batch processing example:")
    
    # Create sample data
    sample_data = {
        'text': [
            "The service was excellent and the staff was very helpful.",
            "I had to wait 2 hours for my appointment. This is unacceptable!",
            "What are your business hours on weekends?",
            "The product quality is decent for the price point.",
            "This is the worst experience I've ever had with any company!",
            "Thank you so much for the quick response to my inquiry.",
            "The website is down. When will it be fixed?",
            "Everything worked perfectly as expected.",
            "I'm not sure if this product is right for me. Can you help?",
            "Outstanding customer service! I'm impressed."
        ],
        'customer_id': [f'CUST_{i:03d}' for i in range(1, 11)],
        'date': ['2024-01-01'] * 10
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Processing {len(df)} texts using batch processor...")
    
    # Create batch processor
    batch_processor = BatchProcessor(judge, max_workers=2, batch_size=3)
    
    # Process the dataframe
    df_with_results = batch_processor.process_dataframe(
        df, 
        text_column='text',
        metadata_columns=['customer_id', 'date']
    )
    
    # Display results
    print("\nBatch processing results:")
    for _, row in df_with_results.iterrows():
        print(f"Customer {row['customer_id']}: {row['classification_justification']} "
              f"(confidence: {row['classification_confidence']:.2f})")
    
    # Save results
    output_file = 'classification_results.csv'
    df_with_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Show API usage statistics
    print(f"\nAPI Statistics:")
    print(f"Total cost: ${api_client.get_total_cost():.4f}")
    print(f"Success rate: {api_client.get_success_rate():.1f}%")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()