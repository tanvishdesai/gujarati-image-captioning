import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import logging
import os
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from tqdm import tqdm
import re
from dataclasses import dataclass
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
from collections import Counter
import math
from torch.cuda.amp import autocast, GradScaler
from contextlib import contextmanager
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ModelConfig:
    """Configuration class for model parameters and dataset paths."""
    # Model parameters
    batch_size: int = 16
    max_length: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 10
    num_workers: int = 4
    pin_memory: bool = True
    train_val_split: float = 0.9
    save_frequency: int = 1
    early_stopping_patience: int = 5
    gradient_clip_val: float = 1.0
    validation_frequency: int = 2
    max_validation_samples: int = 100
    blip_model: str = 'Salesforce/blip-image-captioning-base'
    mt5_model: str = 'google/mt5-base'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True
    
    # Dataset and checkpoint paths
    image_dir: str = 'flickr8k/Flickr_Data/Flickr_Data/Images'
    captions_file: str = 'gujarati_captions.txt'
    checkpoint_dir: str = 'checkpoints_v8'
    
    @classmethod
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            # Explicit conversion to float
            config_dict['learning_rate'] = float(config_dict['learning_rate'])
        return cls(**config_dict)

    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
class MemoryManager:
    """Manages GPU memory usage and cleanup."""
    @staticmethod
    @contextmanager
    def autocast_if_needed(enabled: bool = True):
        """Context manager for automatic mixed precision."""
        if enabled and torch.cuda.is_available():
            with autocast():
                yield
        else:
            yield
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

class EvaluationMetrics:
    """Efficient implementation of evaluation metrics."""
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.smooth = SmoothingFunction()
    
    def calculate_metrics_batch(self, 
                              references: List[str], 
                              hypotheses: List[str],
                              batch_size: int = 32) -> Dict[str, float]:
        """Calculate metrics in batches to manage memory."""
        metrics_sum = {
            'bleu1': 0., 'bleu2': 0., 'bleu3': 0., 'bleu4': 0.,
            'meteor': 0., 'rouge1': 0., 'rouge2': 0., 'rougeL': 0.
        }
        
        for i in range(0, len(references), batch_size):
            batch_refs = references[i:i + batch_size]
            batch_hyps = hypotheses[i:i + batch_size]
            
            for ref, hyp in zip(batch_refs, batch_hyps):
                # Calculate BLEU scores
                for n in range(1, 5):
                    metrics_sum[f'bleu{n}'] += self._compute_bleu(ref, hyp, n)
                
                # Calculate METEOR
                metrics_sum['meteor'] += self._compute_meteor(ref, hyp)
                
                # Calculate ROUGE scores
                rouge_scores = self._compute_rouge(ref, hyp)
                for key, value in rouge_scores.items():
                    metrics_sum[key] += value
        
        # Average the metrics
        num_samples = len(references)
        return {k: (v / num_samples) * 100 for k, v in metrics_sum.items()}
    
    def _compute_bleu(self, reference: str, hypothesis: str, n: int) -> float:
        """Compute BLEU-N score."""
        try:
            ref_tokens = nltk.word_tokenize(reference)
            hyp_tokens = nltk.word_tokenize(hypothesis)
            weights = tuple([1.0/n] * n)
            return sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=self.smooth.method1
            )
        except Exception as e:
            logging.warning(f"BLEU calculation error: {str(e)}")
            return 0.0
    
    def _compute_meteor(self, reference: str, hypothesis: str) -> float:
        """Compute METEOR score."""
        try:
            return meteor_score([reference], hypothesis)
        except Exception as e:
            logging.warning(f"METEOR calculation error: {str(e)}")
            return 0.0
    
    def _compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logging.warning(f"ROUGE calculation error: {str(e)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

class GujaratiPreprocessor:
    """Enhanced Gujarati text preprocessing."""
    def __init__(self):
        self.normalizer = IndicNormalizerFactory().get_normalizer("gu")
    
    def normalize_text(self, text: str) -> str:
        """Normalize Gujarati text."""
        try:
            normalized = self.normalizer.normalize(text)
            return re.sub(r'\s+', ' ', normalized).strip()
        except Exception as e:
            logging.error(f"Text normalization error: {str(e)}")
            return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Gujarati text."""
        try:
            return indic_tokenize.trivial_tokenize(text, lang='gu')
        except Exception as e:
            logging.error(f"Tokenization error: {str(e)}")
            return text.split()

class EnhancedDataset(Dataset):
    """Memory-efficient dataset implementation."""
    def __init__(self, 
                 image_dir: Union[str, Path], 
                 captions_file: Union[str, Path],
                 processor: BlipProcessor,
                 tokenizer: MT5Tokenizer,
                 max_length: int = 64,
                 is_train: bool = True):
        super().__init__()
        self.gujarati_processor = GujaratiPreprocessor()
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Load and validate data
        self._load_and_validate_data(captions_file)
    
    def _load_and_validate_data(self, captions_file: Union[str, Path]):
        """Load and validate dataset entries."""
        self.samples = []
        skipped_images = 0
        
        try:
            with open(captions_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    try:
                        image_name, caption = line.strip().split('\t')
                        image_path = self.image_dir / image_name.split('#')[0]
                        
                        if image_path.exists():
                            normalized_caption = self.gujarati_processor.normalize_text(caption)
                            self.samples.append((str(image_path), normalized_caption))
                        else:
                            skipped_images += 1
                    except Exception as e:
                        logging.warning(f"Error processing line: {line.strip()}, Error: {str(e)}")
                        continue
        
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")
        
        logging.info(f"Loaded {len(self.samples)} valid image-caption pairs")
        logging.info(f"Skipped {skipped_images} images due to missing files")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, caption = self.samples[idx]
        
        try:
            # Load and process image
            with Image.open(image_path).convert('RGB') as image:
                processed_image = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
            
            # Process caption
            tokens = self.gujarati_processor.tokenize(caption)
            processed_caption = " ".join(tokens)
            
            encoding = self.tokenizer(
                processed_caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'image': processed_image,
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'caption': processed_caption
            }
        
        except Exception as e:
            logging.error(f"Error loading item {idx}, image: {image_path}, Error: {str(e)}")
            # Return zero tensors as fallback
            return {
                'image': torch.zeros((3, 224, 224)),
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'caption': ""
            }

class GujaratiCaptioningModelV8(nn.Module):
    """Improved image captioning model with memory efficiency."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize BLIP with memory efficiency
        self.blip = BlipForConditionalGeneration.from_pretrained(
            config.blip_model,
            torch_dtype=torch.float16 if config.mixed_precision else torch.float32
        )
        self.blip_processor = BlipProcessor.from_pretrained(config.blip_model)
        
        # Freeze BLIP parameters
        for param in self.blip.parameters():
            param.requires_grad = False
        
        # Initialize mT5 with memory efficiency
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(
            config.mt5_model,
            torch_dtype=torch.float16 if config.mixed_precision else torch.float32
        )
        
        # Get correct hidden sizes from models
        blip_hidden_size = self.blip.config.vision_config.hidden_size  # Use vision_config for BLIP
        mt5_hidden_size = self.mt5.config.hidden_size
        
        # Feature processing layers
        self.feature_projection = nn.Sequential(
            nn.Linear(blip_hidden_size, mt5_hidden_size),
            nn.LayerNorm(mt5_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=mt5_hidden_size,
            num_heads=8,
            batch_first=True
        )
    
    @torch.cuda.amp.autocast()
    def forward(self, 
                images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Any:
        """Forward pass with automatic mixed precision."""
        try:
            # Get BLIP features
            with torch.no_grad():
                blip_features = self.blip.vision_model(
                    pixel_values=images
                ).last_hidden_state
            
            # Project features
            projected_features = self.feature_projection(blip_features)
            
            # Get mT5 encoder outputs
            encoder_outputs = self.mt5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Combine features using cross-attention
            enhanced_features, _ = self.cross_attention(
                query=encoder_outputs.last_hidden_state,
                key=projected_features,
                value=projected_features
            )
            
            # Generate captions
            outputs = self.mt5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=(enhanced_features,),
                labels=labels,
                return_dict=True
            )
            
            return outputs
        
        except Exception as e:
            logging.error(f"Forward pass error: {str(e)}")
            raise
    
    @torch.no_grad()
    def generate_caption(self, 
                        image: torch.Tensor,
                        tokenizer: MT5Tokenizer,
                        max_length: int = 64,
                        num_beams: int = 4) -> str:
        """Generate caption with error handling."""
        try:
            self.eval()
            
            # Process image with BLIP
            blip_features = self.blip.vision_model(
                pixel_values=image.unsqueeze(0)
            ).last_hidden_state
            
            # Project features
            projected_features = self.feature_projection(blip_features)
            
            # Generate caption
            input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(image.device)
            
            outputs = self.mt5.generate(
                input_ids=input_ids,
                encoder_outputs=torch.nn.utils.rnn.pad_sequence(
                    [projected_features], 
                    batch_first=True
                ),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return ""
def train_model(
    model: GujaratiCaptioningModelV8,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfig,
    tokenizer: MT5Tokenizer,
    checkpoint_dir: Path
) -> List[Dict[str, Any]]:
    """
    Enhanced training function with memory management and error handling.
    """
    # Initialize components
    device = torch.device(config.device)
    evaluator = EvaluationMetrics()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.mixed_precision)
    
    # Calculate training steps
    total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = total_steps // 10
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='linear'
    )
    
    # Training state
    best_meteor = -float('inf')
    train_history = []
    patience_counter = 0
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for epoch in range(config.num_epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs} [Train]')
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    # Move batch to device
                    images = batch['image'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Forward pass with mixed precision
                    with MemoryManager.autocast_if_needed(config.mixed_precision):
                        outputs = model(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        loss = outputs.loss / config.gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.gradient_clip_val
                        )
                        
                        # Optimizer and scheduler steps
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    # Update progress bar
                    epoch_loss += loss.item() * config.gradient_accumulation_steps
                    train_pbar.set_postfix({
                        'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}'
                    })
                
                except Exception as e:
                    logging.error(f"Error in training batch: {str(e)}")
                    continue
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation phase
            if (epoch + 1) % config.validation_frequency == 0:
                model.eval()
                metrics = validate_model(
                    model=model,
                    val_loader=val_loader,
                    tokenizer=tokenizer,
                    evaluator=evaluator,
                    device=device,
                    max_samples=config.max_validation_samples
                )
                
                # Log metrics
                logging.info(f"\nEpoch {epoch + 1} Summary:")
                logging.info(f"Training Loss: {avg_train_loss:.4f}")
                for metric, value in metrics.items():
                    logging.info(f"{metric}: {value:.2f}")
                
                # Save checkpoint if improvement
                if metrics['meteor'] > best_meteor:
                    best_meteor = metrics['meteor']
                    patience_counter = 0
                    
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        metrics=metrics,
                        config=config,
                        checkpoint_dir=checkpoint_dir,
                        is_best=True
                    )
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= config.early_stopping_patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Save training history
                train_history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'metrics': metrics,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Regular checkpoint save
            if (epoch + 1) % config.save_frequency == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=None,
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    is_best=False
                )
            
            # Clear memory
            MemoryManager.clear_gpu_memory()
    
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise
    
    return train_history

def validate_model(
    model: GujaratiCaptioningModelV8,
    val_loader: DataLoader,
    tokenizer: MT5Tokenizer,
    evaluator: EvaluationMetrics,
    device: torch.device,
    max_samples: int
) -> Dict[str, float]:
    """
    Validate model with efficient memory usage.
    """
    references = []
    hypotheses = []
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                if batch_idx >= max_samples:
                    break
                
                images = batch['image'].to(device)
                captions = batch['caption']
                
                # Generate captions
                for img, ref_caption in zip(images, captions):
                    hyp_caption = model.generate_caption(
                        image=img,
                        tokenizer=tokenizer,
                        max_length=64
                    )
                    
                    references.append(ref_caption)
                    hypotheses.append(hyp_caption)
        
        # Calculate metrics in batches
        metrics = evaluator.calculate_metrics_batch(
            references=references,
            hypotheses=hypotheses,
            batch_size=32
        )
        
        return metrics
    
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return {
            'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0,
            'meteor': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0
        }

def save_checkpoint(
    model: GujaratiCaptioningModelV8,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Optional[Dict[str, float]],
    config: ModelConfig,
    checkpoint_dir: Path,
    is_best: bool
) -> None:
    """
    Save model checkpoint with error handling.
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config.to_dict(),
            'metrics': metrics
        }
        
        if is_best:
            path = checkpoint_dir / f'best_model_epoch_{epoch + 1}.pth'
        else:
            path = checkpoint_dir / f'model_epoch_{epoch + 1}.pth'
        
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to {path}")
    
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
        raise

def main():
    """
    Main training pipeline with error handling.
    """
    try:
        # Load configuration
        config = ModelConfig.from_yaml('config.yaml')
        
        # Initialize components
        tokenizer = MT5Tokenizer.from_pretrained(config.mt5_model)
        processor = BlipProcessor.from_pretrained(config.blip_model)
        
        # Create datasets
        full_dataset = EnhancedDataset(
            image_dir=config.image_dir,
            captions_file=config.captions_file,
            processor=processor,
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        
        # Split dataset
        train_size = int(config.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        # Initialize model
        model = GujaratiCaptioningModelV8(config).to(config.device)
        
        # Train model
        checkpoint_dir = Path('checkpoints_v8')
        train_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            tokenizer=tokenizer,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save training history
        history_path = checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2, default=str)
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Fatal error in main function: {str(e)}")
        raise
    
    finally:
        MemoryManager.clear_gpu_memory()

if __name__ == '__main__':
    main()