# Importing all required libraries from V8 plus additional ones
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
from torchvision import transforms
import logging
import os
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from tqdm import tqdm
import re
from dataclasses import dataclass, field
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
from torch.nn.utils import clip_grad_norm_
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F


from colab import EnhancedDataset, ModelConfig, MemoryManager, save_checkpoint, validate_model,   EvaluationMetrics



# Configure logging to use UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Configuration class updates
@dataclass
class ModelConfigV9(ModelConfig):  # Inherits from V8's ModelConfig
    """Enhanced configuration class with new parameters."""
    # New distributed training parameters
    distributed_training: bool = False
    world_size: int = torch.cuda.device_count()
    dist_backend: str = 'nccl'
    dist_url: str = 'tcp://localhost:23456'
    image_dir = r"flickr8k\Flickr_Data\Flickr_Data\Images"  # Update this to your actual image directory path
    caption_file = r"gujarati_captions.txt"  # Update this to your actual caption file path
    val_split: float = 0.1

    # New model architecture parameters
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    positional_embedding_type: str = 'learned'
    use_residual_connections: bool = True
    gradient_checkpointing: bool = True
    
    # New optimization parameters
    use_gradient_centralization: bool = True
    dynamic_batch_size: bool = True
    min_batch_size: int = 8
    max_batch_size: int = 32
    
    # New generation parameters
    diversity_penalty: float = 0.5
    num_beam_groups: int = 4
    num_return_sequences: int = 3
    
    # New evaluation parameters
    semantic_similarity_threshold: float = 0.7
    caption_diversity_weight: float = 0.3
    
    def __post_init__(self):
        super().__post_init__()
        if self.distributed_training and not torch.cuda.is_available():
            logging.warning("Distributed training requested but CUDA is not available. Falling back to CPU.")
            self.distributed_training = False


import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Tuple
from transformers import MT5Tokenizer

class GujaratiCaptioningDataset(Dataset):
    """Custom dataset for Gujarati image captioning."""
    
    def __init__(self, 
                 image_dir: str,
                 caption_file: str,
                 tokenizer: MT5Tokenizer,
                 transform=None,
                 max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing images
            caption_file (str): Path to caption file
            tokenizer: MT5Tokenizer instance
            transform: Optional image transforms
            max_length (int): Maximum length for tokenization
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and parse caption file
        self.samples = self._load_captions(caption_file)
        logging.info(f"Loaded {len(self.samples)} samples from {caption_file}")
    
    def _load_captions(self, caption_file: str) -> List[Dict]:
        """Load captions from file."""
        samples = []
        invalid_lines = []  # To track invalid lines
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Skip empty lines
                        if not line.strip():
                            continue

                        # Split the line into image name and caption using tab as the delimiter
                        parts = line.strip().split('\t')  # Use '\t' for tab delimiter
                        if len(parts) != 2:
                            invalid_lines.append(line.strip())  # Track invalid lines
                            continue

                        img_name, caption = parts

                        # Remove the #0, #1, etc., suffix from the image name
                        img_name = img_name.split('#')[0]  # Remove everything after '#'

                        img_path = self.image_dir / img_name

                        if img_path.exists():
                            samples.append({
                                'image_path': str(img_path),
                                'caption': caption.strip()
                            })
                        else:
                            logging.warning(f"Image file not found: {img_path}")
                    except Exception as e:
                        invalid_lines.append(line.strip())  # Track invalid lines
                        continue
        except Exception as e:
            logging.error(f"Error loading caption file: {str(e)}")
            raise

        # Log invalid lines to a separate file
        if invalid_lines:
            with open('invalid_lines.log', 'w', encoding='utf-8') as f:
                for line in invalid_lines:
                    f.write(line + '\n')
            logging.warning(f"Found {len(invalid_lines)} invalid lines in the caption file. Check 'invalid_lines.log' for details.")

        logging.info(f"Loaded {len(samples)} valid samples from the caption file.")
        return samples
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        try:
            # Load and process image
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Tokenize caption
            tokenized = self.tokenizer(
                sample['caption'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'image': image,
                'input_ids': tokenized['input_ids'].squeeze(0),
                'attention_mask': tokenized['attention_mask'].squeeze(0),
                'caption': sample['caption']
            }
            
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {str(e)}")
            raise
            
def create_train_val_datasets(
    config: ModelConfigV9,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Create training and validation datasets by splitting a single dataset.
    
    Args:
        config: Model configuration
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    try:
        # Create transforms for the images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size based on your needs
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create the full dataset
        tokenizer = MT5Tokenizer.from_pretrained(config.mt5_model)
        full_dataset = GujaratiCaptioningDataset(
            image_dir=config.image_dir,
            caption_file=config.caption_file,
            tokenizer=tokenizer,
            transform=transform,
            max_length=config.max_length
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        logging.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logging.error(f"Error creating datasets: {str(e)}")
        raise

# Update ModelConfigV9 to include new dataset parameters
def update_config_for_single_dataset(config: ModelConfigV9) -> ModelConfigV9:
    """Update configuration for single dataset setup."""
    config.image_dir = "path/to/your/image/directory"  # Update this
    config.caption_file = "path/to/your/captions.txt"  # Update this
    config.val_split = 0.1  # Validation split ratio
    return config


# Enhanced memory manager
class EnhancedMemoryManager(MemoryManager):
    """Enhanced memory management with better monitoring."""
    
    @staticmethod
    def get_gpu_memory_usage():
        if torch.cuda.is_available():
            return {i: torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())}
        return {}
    
    @staticmethod
    def is_memory_critical():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) > 0.95:
                    return True
        return False
    
    @staticmethod
    def optimize_memory_for_dynamic_batch():
        """Optimize memory for dynamic batch processing."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0

# Enhanced positional embedding
class EnhancedPositionalEmbedding(nn.Module):
    """Advanced positional embedding with multiple encoding options."""
    
    def __init__(self, d_model: int, max_length: int = 512, embedding_type: str = 'learned'):
        super().__init__()
        self.embedding_type = embedding_type
        
        if embedding_type == 'learned':
            self.pe = nn.Parameter(torch.randn(1, max_length, d_model))
        else:  # sinusoidal
            pe = torch.zeros(max_length, d_model)
            position = torch.arange(0, max_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding_type == 'learned':
            return x + self.pe[:, :x.size(1)]
        return x + self.pe[:, :x.size(1)]

# Enhanced feature projection with residual connections
class EnhancedFeatureProjection(nn.Module):
    """Feature projection module with residual connections and layer normalization."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.norm1 = nn.LayerNorm(output_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(output_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        residual = x
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        
        return x

# Enhanced cross-attention with pruning
class EnhancedCrossAttention(nn.Module):
    """Cross-attention module with attention pruning and improved scaling."""
    
    def __init__(self, config: ModelConfigV9):
        super().__init__()
        self.config = config
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.attention_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.attention_threshold = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Scale attention scores
        attn_output, attn_weights = self.attention(query, key, value, key_padding_mask=mask)
        
        # Prune attention weights below threshold
        attention_mask = (attn_weights > self.attention_threshold).float()
        pruned_attention = attn_weights * attention_mask
        
        # Rescale to maintain sum = 1
        pruned_attention = pruned_attention / (pruned_attention.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Apply scaled attention
        attn_output = torch.matmul(pruned_attention, value)
        
        return attn_output, pruned_attention

# Main model class
class GujaratiCaptioningModelV9(nn.Module):
    """Enhanced version of the Gujarati captioning model with improved architecture."""
    
    def __init__(self, config: ModelConfigV9):
        super().__init__()
        self.config = config
        
        # Initialize core models
        self.blip = BlipForConditionalGeneration.from_pretrained(
            config.blip_model,
            torch_dtype=torch.float32
        )
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(
            config.mt5_model,
            torch_dtype=torch.float32
        )
        
        # Freeze BLIP parameters
        for param in self.blip.parameters():
            param.requires_grad = False
        
        # Get model dimensions
        self.blip_hidden_size = self.blip.config.vision_config.hidden_size
        self.mt5_hidden_size = self.mt5.config.hidden_size
        
        # Enhanced components
        self.feature_projection = EnhancedFeatureProjection(
            self.blip_hidden_size,
            self.mt5_hidden_size,
            dropout=config.hidden_dropout_prob
        )
        
        self.positional_embedding = EnhancedPositionalEmbedding(
            self.mt5_hidden_size,
            config.max_length,
            config.positional_embedding_type
        )
        
        self.cross_attention = EnhancedCrossAttention(config)
        
        # Initialize semantic similarity model for evaluation
        if torch.cuda.is_available():
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.mt5.gradient_checkpointing_enable()
    
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Any:
        """Enhanced forward pass with improved feature processing."""
        try:
            # Process images with BLIP
            with torch.no_grad():
                blip_features = self.blip.vision_model(
                    pixel_values=images.to(dtype=torch.float32)
                ).last_hidden_state
            
            # Project features with residual connections
            projected_features = self.feature_projection(blip_features)
            
            # Add positional embeddings
            projected_features = self.positional_embedding(projected_features)
            
            # Process with MT5 encoder
            encoder_outputs = self.mt5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Apply enhanced cross-attention
            enhanced_features, attention_weights = self.cross_attention(
                query=encoder_outputs.last_hidden_state,
                key=projected_features,
                value=projected_features,
                mask=attention_mask
            )
            
            # Generate outputs with MT5
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
    def generate_diverse_captions(self, image: torch.Tensor, tokenizer: MT5Tokenizer,
                                num_captions: int = 3) -> List[str]:
        """Generate diverse captions using beam search groups."""
        try:
            self.eval()
            
            # Process image
            blip_features = self.blip.vision_model(
                pixel_values=image.unsqueeze(0)
            ).last_hidden_state
            
            projected_features = self.feature_projection(blip_features)
            projected_features = self.positional_embedding(projected_features)
            
            # Generate multiple captions
            input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(image.device)
            
            outputs = self.mt5.generate(
                input_ids=input_ids,
                encoder_outputs=torch.nn.utils.rnn.pad_sequence([projected_features], batch_first=True),
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                num_beam_groups=self.config.num_beam_groups,
                num_return_sequences=num_captions,
                diversity_penalty=self.config.diversity_penalty,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=0.8
            )
            
            captions = [tokenizer.decode(output, skip_special_tokens=True) 
                       for output in outputs]
            
            return self._filter_diverse_captions(captions)
        
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return []
    
    def _filter_diverse_captions(self, captions: List[str]) -> List[str]:
        """Filter captions based on semantic similarity."""
        if not hasattr(self, 'semantic_model') or len(captions) <= 1:
            return captions
        
        try:
            # Get embeddings for all captions
            embeddings = self.semantic_model.encode(captions)
            
            # Calculate pairwise similarities
            similarities = F.cosine_similarity(
                torch.tensor(embeddings).unsqueeze(0),
                torch.tensor(embeddings).unsqueeze(1)
            )
            
            # Filter out too similar captions
            filtered_indices = []
            for i in range(len(captions)):
                if not any(similarities[i][j] > self.config.semantic_similarity_threshold 
                          for j in filtered_indices):
                    filtered_indices.append(i)
            
            return [captions[i] for i in filtered_indices]
        
        except Exception as e:
            logging.warning(f"Caption filtering error: {str(e)}")
            return captions

# Training function with distributed support
# Continuing from previous training function

class EnhancedEvaluationMetrics(EvaluationMetrics):
    """Enhanced evaluation metrics with diversity measures."""
    
    def __init__(self, config: ModelConfigV9):
        super().__init__()
        self.config = config
        self.semantic_model = None
        if torch.cuda.is_available():
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    def calculate_metrics_batch(self,
                              references: List[str],
                              hypotheses: List[str],
                              diverse_hypotheses: Optional[List[List[str]]] = None,
                              batch_size: int = 32) -> Dict[str, float]:
        """Calculate metrics including diversity measures."""
        # Get base metrics
        base_metrics = super().calculate_metrics_batch(references, hypotheses, batch_size)
        
        # Add diversity metrics if available
        if diverse_hypotheses and self.semantic_model:
            diversity_metrics = self._calculate_diversity_metrics(diverse_hypotheses)
            base_metrics.update(diversity_metrics)
        
        return base_metrics
    
    def _calculate_diversity_metrics(self, diverse_hypotheses: List[List[str]]) -> Dict[str, float]:
        """Calculate diversity metrics for generated captions."""
        diversity_scores = {
            'distinct_1': 0.0,
            'distinct_2': 0.0,
            'semantic_diversity': 0.0
        }
        
        try:
            # Calculate n-gram diversity
            for captions in diverse_hypotheses:
                if len(captions) > 1:
                    diversity_scores['distinct_1'] += self._distinct_n_grams(captions, 1)
                    diversity_scores['distinct_2'] += self._distinct_n_grams(captions, 2)
            
            # Calculate semantic diversity
            if self.semantic_model:
                diversity_scores['semantic_diversity'] = self._semantic_diversity(diverse_hypotheses)
            
            # Average scores
            num_samples = len(diverse_hypotheses)
            return {k: v / num_samples for k, v in diversity_scores.items()}
        
        except Exception as e:
            logging.error(f"Error calculating diversity metrics: {str(e)}")
            return diversity_scores
    
    def _distinct_n_grams(self, captions: List[str], n: int) -> float:
        """Calculate ratio of distinct n-grams."""
        all_ngrams = []
        for caption in captions:
            tokens = nltk.word_tokenize(caption)
            all_ngrams.extend(list(nltk.ngrams(tokens, n)))
        
        if not all_ngrams:
            return 0.0
        
        return len(set(all_ngrams)) / len(all_ngrams)
    
    def _semantic_diversity(self, diverse_hypotheses: List[List[str]]) -> float:
        """Calculate semantic diversity using sentence embeddings."""
        try:
            total_diversity = 0.0
            for captions in diverse_hypotheses:
                if len(captions) > 1:
                    embeddings = self.semantic_model.encode(captions)
                    similarities = F.cosine_similarity(
                        torch.tensor(embeddings).unsqueeze(0),
                        torch.tensor(embeddings).unsqueeze(1)
                    )
                    # Calculate average pairwise dissimilarity
                    total_diversity += 1.0 - torch.triu(similarities, diagonal=1).mean().item()
            
            return total_diversity / len(diverse_hypotheses)
        
        except Exception as e:
            logging.error(f"Error calculating semantic diversity: {str(e)}")
            return 0.0

def create_optimizer_with_gc(model: nn.Module, config: ModelConfigV9) -> torch.optim.Optimizer:
    """Create optimizer with gradient centralization."""
    
    def get_parameters_with_gc():
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Apply gradient centralization to conv and linear layers
                if 'conv' in name or 'linear' in name:
                    yield {'params': [param], 'apply_gc': True}
                else:
                    yield {'params': [param], 'apply_gc': False}
    
    optimizer = torch.optim.AdamW(
        get_parameters_with_gc(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay
    )
    
    return optimizer


def validate_model_v9(
    model: GujaratiCaptioningModelV9,
    val_loader: DataLoader,
    tokenizer: MT5Tokenizer,
    evaluator: EnhancedEvaluationMetrics,
    config: ModelConfigV9,
    device: torch.device,
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    Enhanced validation function for V9 model with support for diverse caption generation and semantic similarity.
    """
    model.eval()
    references = []
    hypotheses = []
    diverse_hypotheses = []

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                if max_samples and batch_idx >= max_samples:
                    break

                images = batch['image'].to(device)
                captions = batch['caption']

                # Generate standard captions
                for img, ref_caption in zip(images, captions):
                    hyp_caption = model.generate_caption(
                        image=img,
                        tokenizer=tokenizer,
                        max_length=config.max_length
                    )
                    references.append(ref_caption)
                    hypotheses.append(hyp_caption)

                # Generate diverse captions if configured
                if config.num_return_sequences > 1:
                    for img in images:
                        diverse_captions = model.generate_diverse_captions(
                            image=img,
                            tokenizer=tokenizer,
                            num_captions=config.num_return_sequences
                        )
                        diverse_hypotheses.append(diverse_captions)

        # Calculate metrics including diversity if available
        metrics = evaluator.calculate_metrics_batch(
            references=references,
            hypotheses=hypotheses,
            diverse_hypotheses=diverse_hypotheses if diverse_hypotheses else None,
            batch_size=32
        )

        return metrics

    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return {
            'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0,
            'meteor': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'distinct_1': 0.0, 'distinct_2': 0.0, 'semantic_diversity': 0.0
        }



def apply_gradient_centralization(model: nn.Module):
    """Apply gradient centralization to conv and linear layers."""
    for param in model.parameters():
        if getattr(param, 'apply_gc', False) and param.grad is not None:
            if len(param.grad.shape) > 1:
                param.grad.add_(-param.grad.mean(dim=tuple(range(1, len(param.grad.shape))), keepdim=True))

def adjust_batch_size(memory_usage: Dict[int, int], config: ModelConfigV9) -> int:
    """Dynamically adjust batch size based on memory usage."""
    memory_pressure = max(usage / torch.cuda.max_memory_allocated(i) 
                         for i, usage in memory_usage.items())
    
    if memory_pressure > 0.9:
        return max(config.min_batch_size, config.batch_size // 2)
    elif memory_pressure < 0.7:
        return min(config.max_batch_size, config.batch_size * 2)
    
    return config.batch_size


def save_best_checkpoint(
    model: GujaratiCaptioningModelV9,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Optional[Dict[str, float]],
    config: ModelConfig,
    checkpoint_dir: Path
) -> None:
    """
    Save the best model checkpoint.
    """
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

def save_regular_checkpoint(
    model: GujaratiCaptioningModelV9,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Optional[Dict[str, float]],
    config: ModelConfig,
    checkpoint_dir: Path
) -> None:
    """
    Save a regular model checkpoint.
    """
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics=metrics,
        config=config,
        checkpoint_dir=checkpoint_dir,
        is_best=False
    )

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    config: ModelConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy='linear'
    )

def create_datasets(
    config: ModelConfig
) -> Tuple[Dataset, Dataset]:
    """
    Create training and validation datasets.
    """
    full_dataset = EnhancedDataset(
        image_dir=config.image_dir,
        captions_file=config.captions_file,
        processor=BlipProcessor.from_pretrained(config.blip_model),
        tokenizer=MT5Tokenizer.from_pretrained(config.mt5_model),
        max_length=config.max_length
    )
    
    train_size = int(config.train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset

def create_optimizer(
    model: nn.Module,
    config: ModelConfig
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import MT5Tokenizer

# Other imports and code...
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import MT5Tokenizer
from tqdm import tqdm
from typing import Dict, List, Optional
from torch.cuda.amp import autocast, GradScaler


def train_model_single_gpu(
    model: GujaratiCaptioningModelV9,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfigV9,
    tokenizer: MT5Tokenizer,
    checkpoint_dir: Path
) -> List[Dict[str, Any]]:
    """
    Training function for single-GPU training.
    """
    device = torch.device(config.device)  # Use the specified device (e.g., 'cuda' or 'cpu')
    model = model.to(device)  # Move the model to the specified device
    evaluator = EvaluationMetrics()
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=config.mixed_precision)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy='linear'
    )
    
    best_meteor = -float('inf')
    train_history = []
    patience_counter = 0
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for epoch in range(config.num_epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs} [Train]')
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    # Move batch to the specified device
                    images = batch['image'].to(device, dtype=torch.float32)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Mixed precision training
                    with autocast(enabled=config.mixed_precision):
                        outputs = model(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        loss = outputs.loss
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=config.gradient_clip_val
                    )
                    
                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        epoch_loss += loss.item()
                        train_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                            'grad_norm': f'{grad_norm.item():.2f}'
                        })
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Validation step
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
                
                logging.info(f"\nEpoch {epoch + 1} Summary:")
                logging.info(f"Training Loss: {epoch_loss / len(train_loader):.4f}")
                for metric, value in metrics.items():
                    logging.info(f"{metric}: {value:.2f}")
                
                # Save best checkpoint
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
                
                # Early stopping
                if patience_counter >= config.early_stopping_patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Save regular checkpoint
                if (epoch + 1) % config.save_frequency == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        metrics=metrics,
                        config=config,
                        checkpoint_dir=checkpoint_dir,
                        is_best=False
                    )
                
                # Update training history
                train_history.append({
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss / len(train_loader),
                    'metrics': metrics,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Clear GPU memory
            torch.cuda.empty_cache()
    
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise
    
    return train_history

def main():
    """Main function for single-GPU training."""
    try:
        # Load configuration
        config = ModelConfigV9.create_default_config()
        config.image_dir = r"flickr8k\Flickr_Data\Flickr_Data\Images"  # Update this
        config.caption_file = r"gujarati_captions.txt"  # Update this
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        
        # Initialize components
        tokenizer = MT5Tokenizer.from_pretrained(config.mt5_model)
        
        # Create datasets
        train_dataset, val_dataset = create_train_val_datasets(config)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Initialize model
        model = GujaratiCaptioningModelV9(config).to(config.device)
        
        # Train model
        checkpoint_dir = Path(config.checkpoint_dir)
        train_history = train_model_single_gpu(
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

if __name__ == '__main__':
    main()