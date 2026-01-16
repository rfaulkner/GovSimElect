"""
Text processing utilities for batch operations and data handling.
"""

import pandas as pd
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable
from tqdm.auto import tqdm


class TextProcessor:
    """
    Handles individual text processing operations.
    """
    
    def __init__(self, judge):
        """
        Initialize text processor.
        
        Args:
            judge: LLMJudge instance
        """
        self.judge = judge
    
    def process_text(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a single text with optional metadata.
        
        Args:
            text: Text to classify
            metadata: Optional metadata to include in result
            
        Returns:
            Dictionary with classification results and metadata
        """
        if not text or len(text.strip()) == 0:
            result = {
                "Reasoning_behind_classification": "No text provided.",
                "Confidence": 0.0,
                "justification_type": "Other"
            }
        else:
            result = self.judge.classify_text(text)
        
        # Add metadata if provided
        if metadata:
            result.update(metadata)
        
        # Rename fields for consistency
        final_result = {
            "classification_explanation": result.get("Reasoning_behind_classification", ""),
            "classification_confidence": result.get("Confidence", 0.0),
            "classification_justification": result.get("justification_type", "Other")
        }
        
        # Add metadata
        if metadata:
            final_result.update(metadata)
        
        return final_result


class BatchProcessor:
    """
    Handles batch processing of multiple texts with parallel execution.
    """
    
    def __init__(self, judge, max_workers: int = 4, batch_size: int = 10):
        """
        Initialize batch processor.
        
        Args:
            judge: LLMJudge instance
            max_workers: Number of parallel workers
            batch_size: Size of each processing batch
        """
        self.judge = judge
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.text_processor = TextProcessor(judge)
    
    def process_texts(self, texts: List[str], metadata_list: Optional[List[Dict]] = None,
                     progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Process a list of texts in parallel.
        
        Args:
            texts: List of texts to process
            metadata_list: Optional list of metadata dicts (same length as texts)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of classification results
        """
        if not texts:
            return []
        
        # Prepare metadata
        if metadata_list is None:
            metadata_list = [{"index": i} for i in range(len(texts))]
        elif len(metadata_list) != len(texts):
            raise ValueError("metadata_list must have same length as texts")
        
        # Create batches
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]
            batches.append(list(zip(batch_texts, batch_metadata)))
        
        print(f"Processing {len(texts)} texts in {len(batches)} batches with {self.max_workers} workers")
        
        all_results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                              total=len(batches), desc="Processing batches"):
                try:
                    results = future.result()
                    all_results.extend(results)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(len(all_results), len(texts))
                        
                except Exception as e:
                    batch_idx = future_to_batch[future]
                    print(f"Batch {batch_idx} generated an exception: {e}")
        
        # Sort results by index to maintain order
        all_results.sort(key=lambda x: x.get('index', 0))
        
        elapsed = time.time() - start_time
        print(f"Completed processing {len(texts)} texts in {elapsed:.2f} seconds")
        print(f"Average speed: {len(texts)/elapsed:.2f} texts/second")
        
        return all_results
    
    def _process_batch(self, batch: List[tuple], delay: float = 0.5) -> List[Dict[str, Any]]:
        """
        Process a single batch of texts.
        
        Args:
            batch: List of (text, metadata) tuples
            delay: Delay between requests to prevent rate limiting
            
        Returns:
            List of results for this batch
        """
        results = []
        
        for text, metadata in batch:
            try:
                result = self.text_processor.process_text(text, metadata)
                time.sleep(delay)  # Rate limiting
            except Exception as e:
                print(f"Error processing text: {e}")
                result = {
                    "classification_explanation": f"Error in analysis: {str(e)}",
                    "classification_confidence": 0.0,
                    "classification_justification": "Failed classification"
                }
                if metadata:
                    result.update(metadata)
            
            results.append(result)
        
        return results
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                         metadata_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process texts from a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing texts to classify
            metadata_columns: List of columns to include as metadata
            
        Returns:
            DataFrame with classification results added
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # Prepare metadata
        metadata_list = []
        for i, row in df.iterrows():
            metadata = {"index": i, "original_index": row.name}
            
            if metadata_columns:
                for col in metadata_columns:
                    if col in df.columns:
                        metadata[col] = row[col]
            
            metadata_list.append(metadata)
        
        # Process texts
        results = self.process_texts(texts, metadata_list)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by original index and merge with original DataFrame
        results_df = results_df.sort_values('original_index')
        
        # Add classification columns to original DataFrame
        for col in ['classification_explanation', 'classification_confidence', 'classification_justification']:
            if col in results_df.columns:
                df[col] = results_df[col].values
        
        return df