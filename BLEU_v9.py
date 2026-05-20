import torch
from pathlib import Path
from transformers import MT5Tokenizer, BlipProcessor
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Optional
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.data import DataLoader
import argparse

# Import necessary classes from the training script
from kaggle_no_validation_space_issue import (
    GujaratiCaptioningModelV9,
    ModelConfigV9,
    GujaratiCaptioningDataset,
    create_train_val_datasets
)

def load_checkpoint(
    checkpoint_path: str,
    config: ModelConfigV9,
    device: torch.device
) -> GujaratiCaptioningModelV9:
    """Load model from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = GujaratiCaptioningModelV9(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}")
        raise

def calculate_bleu_scores(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU-1,2,3,4 scores for the generated captions.
    """
    # Tokenize hypotheses
    hypothesis_tokens = [nltk.word_tokenize(hyp.lower()) for hyp in hypotheses]
    
    # Tokenize references
    reference_tokens = [[nltk.word_tokenize(ref.lower()) for ref in refs] for refs in references]
    
    # Calculate BLEU scores with smoothing
    smooth = SmoothingFunction()
    bleu_scores = {}
    
    for i in range(1, 5):
        weights = tuple([1.0 / i] * i)
        score = corpus_bleu(
            reference_tokens,
            hypothesis_tokens,
            weights=weights,
            smoothing_function=smooth.method1
        )
        bleu_scores[f'bleu{i}'] = score * 100
    
    return bleu_scores

def test_model(
    model: GujaratiCaptioningModelV9,
    test_loader: DataLoader,
    tokenizer: MT5Tokenizer,
    device: torch.device,
    max_length: int = 64
) -> Dict[str, float]:
    """
    Test the model and calculate BLEU scores.
    """
    model.eval()
    all_references = []
    all_hypotheses = []
    
    try:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch['image'].to(device)
                captions = batch['caption']
                
                # Generate captions for each image
                for img, ref_caption in zip(images, captions):
                    # Generate caption
                    input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)
                    outputs = model.generate_caption(
                        image=img.unsqueeze(0),
                        tokenizer=tokenizer,
                        max_length=max_length
                    )
                    
                    # Store reference and hypothesis
                    all_references.append([ref_caption])
                    all_hypotheses.append(outputs)
        
        # Calculate BLEU scores
        bleu_scores = calculate_bleu_scores(all_references, all_hypotheses)
        
        # Log some example outputs
        num_examples = min(5, len(all_hypotheses))
        logging.info("\nExample Generations:")
        for i in range(num_examples):
            logging.info(f"\nReference: {all_references[i][0]}")
            logging.info(f"Generated: {all_hypotheses[i]}")
        
        # Log BLEU scores
        logging.info("\nBLEU Scores:")
        for metric, score in bleu_scores.items():
            logging.info(f"{metric}: {score:.2f}")
        
        return bleu_scores
    
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test Gujarati Image Captioning Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--output', type=str, default='test_results.json', help='Path to save test results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()
    
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('test.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Load configuration
        config = ModelConfigV9.load(args.config)
        config.batch_size = args.batch_size
        config.num_workers = args.num_workers
        
        # Initialize tokenizer and processor
        tokenizer = MT5Tokenizer.from_pretrained(config.mt5_model)
        
        # Create test dataset and loader
        _, test_dataset = create_train_val_datasets(config)  # Using validation set as test set
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Load model
        model = load_checkpoint(args.checkpoint, config, device)
        model.eval()
        
        # Test model
        results = test_model(model, test_loader, tokenizer, device, config.max_length)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"\nTest results saved to {args.output}")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == '__main__':
    main()