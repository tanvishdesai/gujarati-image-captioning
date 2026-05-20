# Importing all required libraries from V8 plus additional ones
from logging import config
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
from transformers import get_cosine_schedule_with_warmup
import os
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from tqdm import tqdm
import re
from dataclasses import dataclass, field, asdict

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
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Tuple
from transformers import MT5Tokenizer
import gc
from torch.nn.utils import clip_grad_norm_
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F


# Configure logging to use UTF-8 encoding
logging.basicConfig(
    level=print,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)



from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import json
import logging
import torch
from typing import Dict, Any, Optional

from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import json
import logging
import torch
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """
    Unified configuration class for model parameters, combining the best features
    from both versions while eliminating redundancy and improving organization.
    """
    # First, define fields that don't have default values
    # (in this case, we'll give defaults to all fields to make it more flexible)
        # Add new parameters
    initial_warmup_steps: int = 1000
    min_learning_rate: float = 1e-7
    gradient_clip_val: float = 0.1
    loss_scale_factor: float = 0.1
    # Fields are organized by category, with all fields having default values
    # Training parameters
    batch_size: int = 8
    max_length: int = 197
    learning_rate: float = 1e-5
    num_epochs: int = 5
    gradient_accumulation_steps: int = 4
    num_workers: int = 4
    # Model architecture parameters
    hidden_size: int = 768
    num_attention_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-7
    positional_embedding_type: str = 'learned'
    use_residual_connections: bool = True
    
    # Optimization parameters
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    fp16_opt_level: str = 'O2'
    
    # Distributed training parameters
    distributed_training: bool = False
    dist_backend: str = 'nccl'
    dist_url: str = 'tcp://localhost:23456'
    
    # Device settings
    device: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    world_size: int = field(
        default_factory=lambda: torch.cuda.device_count()
    )
    
    # Dataset and validation parameters
    train_val_split: float = 0.9
    max_validation_samples: int = 500
    validation_frequency: int = 1
    early_stopping_patience: int = 5
    save_frequency: int = 5
    
    # Model paths and settings
    blip_model: str = 'Salesforce/blip-image-captioning-base'
    mt5_model: str = 'google/mt5-base'
    image_dir: str = '/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/'
    captions_file: str = '/kaggle/input/guj-captions/gujarati_captions.txt'
    checkpoint_dir: str = 'checkpoints'
    
    # Generation parameters
    num_beams: int = 4
    num_beam_groups: int = 4
    num_return_sequences: int = 3
    length_penalty: float = 1.0
    diversity_penalty: float = 0.5
    no_repeat_ngram_size: int = 3
    max_generation_length: int = 128
    
    # Logging parameters
    log_level: str = 'INFO'
    log_frequency: int = 100
    experiment_name: str = field(
        default_factory=lambda: f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    def __post_init__(self):
        """Initialize logging, create directories, and validate configuration."""
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.experiment_name}.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create directories and validate configuration
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Log basic configuration
        logging.info(f"Initialized experiment: {self.experiment_name}")
        logging.info(f"Using device: {self.device}")
        
        # Validate distributed training settings
        if self.distributed_training and not torch.cuda.is_available():
            logging.warning("Distributed training requested but CUDA is not available. Falling back to CPU.")
            self.distributed_training = False
        
        # Validate paths
        self._validate_paths()
    
    @classmethod
    def create_default_config(cls) -> 'ModelConfig':
        """Create a configuration with default values."""
        return cls()
    
    @classmethod
    def create_custom_config(cls, **kwargs) -> 'ModelConfig':
        """Create a configuration with custom values."""
        config = cls()
        config.update(**kwargs)
        return config
    
    def _validate_paths(self) -> None:
        """Validate existence of necessary paths and files."""
        paths = {
            'Image directory': Path(self.image_dir),
            'Captions file': Path(self.captions_file),
            'Checkpoint directory': Path(self.checkpoint_dir)
        }
        
        for name, path in paths.items():
            if not path.exists():
                logging.warning(f"{name} not found: {path}")
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters with validation."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated {key} to {value}")
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def save(self, filepath: Optional[str] = None) -> None:
        """Save configuration to JSON file."""
        if filepath is None:
            filepath = f"{self.experiment_name}_config.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(self), f, indent=2, default=str)
            logging.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            config = cls()
            config.update(**config_dict)
            logging.info(f"Configuration loaded from {filepath}")
            return config
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            raise
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get consolidated training parameters."""
        return {
            'optimizer': {
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'betas': (0.9, 0.999)
            },
            'generation': {
                'num_beams': self.num_beams,
                'num_beam_groups': self.num_beam_groups,
                'num_return_sequences': self.num_return_sequences,
                'length_penalty': self.length_penalty,
                'diversity_penalty': self.diversity_penalty,
                'no_repeat_ngram_size': self.no_repeat_ngram_size,
                'max_length': self.max_generation_length,
                'early_stopping': True
            }
        }
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
        print(f"Loaded {len(self.samples)} samples from {caption_file}")
    
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
                            print(f"Image file not found: {img_path}")
                    except Exception as e:
                        invalid_lines.append(line.strip())  # Track invalid lines
                        continue
        except Exception as e:
            print(f"Error loading caption file: {str(e)}")
            raise

        # Log invalid lines to a separate file
        if invalid_lines:
            with open('invalid_lines.log', 'w', encoding='utf-8') as f:
                for line in invalid_lines:
                    f.write(line + '\n')
            print(f"Found {len(invalid_lines)} invalid lines in the caption file. Check 'invalid_lines.log' for details.")

        print(f"Loaded {len(samples)} valid samples from the caption file.")
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
            print(f"Error loading sample {idx}: {str(e)}")
            raise

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
            print(f"BLEU calculation error: {str(e)}")
            return 0.0
    
    def _compute_meteor(self, reference: str, hypothesis: str) -> float:
        """Compute METEOR score."""
        try:
            return meteor_score([reference], hypothesis)
        except Exception as e:
            print(f"METEOR calculation error: {str(e)}")
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
            print(f"ROUGE calculation error: {str(e)}")
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
            print(f"Text normalization error: {str(e)}")
            return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Gujarati text."""
        try:
            return indic_tokenize.trivial_tokenize(text, lang='gu')
        except Exception as e:
            print(f"Tokenization error: {str(e)}")
            return text.split()

     
def create_train_val_datasets(
    config: ModelConfig,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Create training and validation datasets with proper validation checks.
    
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Validate paths before creating dataset
        image_dir = Path(config.image_dir)
        caption_file = Path(config.caption_file)
        
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        if not caption_file.exists():
            raise ValueError(f"Caption file not found: {caption_file}")
            
        # Create the full dataset
        tokenizer = MT5Tokenizer.from_pretrained(config.mt5_model)
        full_dataset = GujaratiCaptioningDataset(
            image_dir=str(image_dir),
            caption_file=str(caption_file),
            tokenizer=tokenizer,
            transform=transform,
            max_length=config.max_length
        )
        
        # Validate dataset size
        total_size = len(full_dataset)
        if total_size == 0:
            raise ValueError("Dataset is empty. Please check the data loading process.")
        
        # Calculate split sizes
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        
        if train_size == 0 or val_size == 0:
            raise ValueError(f"Invalid split sizes - train: {train_size}, val: {val_size}")
        
        # Split the dataset
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        print(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"Error creating datasets: {str(e)}")
        raise

def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: ModelConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders with validation checks.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Model configuration
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    try:
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty")
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        raise

def initialize_training(model, config):
    """Initialize training with stable defaults."""
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "vision_model" in n],
            "lr": config.learning_rate * 0.1,  # Lower learning rate for vision
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "vision_model" not in n],
            "lr": config.learning_rate,
            "weight_decay": 0.01
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=1e-7,
        betas=(0.9, 0.999)
    )
    
    scaler = torch.amp.GradScaler(
        enabled=config.mixed_precision,
        init_scale=2**8,
        growth_factor=1.1,
        backoff_factor=0.5,
        growth_interval=2000
    )
    
    return optimizer, scaler







# Update ModelConfig to include new dataset parameters
def update_config_for_single_dataset(config: ModelConfig) -> ModelConfig:
    """Update configuration for single dataset setup."""
    config.image_dir = "path/to/your/image/directory"  # Update this
    config.caption_file = "path/to/your/captions.txt"  # Update this
    config.val_split = 0.1  # Validation split ratio
    return config

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
        
        # Add the missing input normalization layer
        self.input_norm = nn.LayerNorm(input_dim, eps=1e-6)

        self.linear1 = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(output_dim, output_dim)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        nn.init.zeros_(self.linear2.bias)
        self.norm1 = nn.LayerNorm(output_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(output_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Now input_norm is properly defined
        x = self.input_norm(x)
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
    
    def __init__(self, config: ModelConfig):
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
        # Convert mask to float if it exists
        attention_mask = None
        if mask is not None:
            # Convert to boolean first, then float
            attention_mask = mask.bool().float()
            # Invert the mask as per PyTorch convention (1 = ignore, 0 = attend)
            attention_mask = attention_mask.logical_not().float()
        
        # Forward pass through multi-head attention
        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=attention_mask
        )
        
        # Prune attention weights below threshold
        attention_mask = (attn_weights > self.attention_threshold).float()
        pruned_attention = attn_weights * attention_mask
        
        # Rescale to maintain sum = 1
        pruned_attention = pruned_attention / (pruned_attention.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Apply scaled attention
        output = torch.matmul(pruned_attention, value)
        
        return output, pruned_attention
# Main model class
class GujaratiCaptioningModelV9(nn.Module):
    """Enhanced version of the Gujarati captioning model with improved architecture."""
    
    def __init__(self, config: ModelConfig):
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
    
    def _normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize input images with robust handling of outliers and numerical stability."""
        eps = 1e-8
        images = torch.clamp(images, -5, 5)  # More conservative clamping
        mean = images.mean(dim=(2, 3), keepdim=True)
        std = images.std(dim=(2, 3), keepdim=True) + eps
        images = (images - mean) / std
        return images
    
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Any:
        """Enhanced forward pass with improved feature processing."""
        try:
            # Normalize and validate inputs
            if torch.isnan(images).any():
                raise ValueError("NaN values detected in input images")
                
            # Normalize images using the class method
            images = self._normalize_inputs(images)
            
            # Add small epsilon to prevent division by zero
            images = images / (images.norm(dim=-1, keepdim=True) + 1e-7)

            # Process images with BLIP
            with torch.no_grad():
                blip_features = self.blip.vision_model(
                    pixel_values=images.to(dtype=torch.float32)
                ).last_hidden_state
            
            # Normalize features
            blip_features = F.layer_norm(
                blip_features, 
                normalized_shape=(blip_features.size(-1),)
            )

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
            
            loss = outputs.loss
            if torch.isnan(loss):
                return outputs
                
            loss_scale = min(1.0, 10.0 / (loss.item() + 1e-8))
            outputs.loss = outputs.loss * loss_scale
            
            return outputs
            
        except Exception as e:
            print(f"Forward pass error: {str(e)}")
            raise       

def normalize_inputs(self, images):
    # Add robust normalization
    eps = 1e-8
    images = torch.clamp(images, -5, 5)  # More conservative clamping
    mean = images.mean(dim=(2, 3), keepdim=True)
    std = images.std(dim=(2, 3), keepdim=True) + eps
    images = (images - mean) / std
    return images

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    config: ModelConfig,
    checkpoint_dir: Path,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint with all necessary information for resuming training.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The learning rate scheduler state
        epoch: Current epoch number
        metrics: Dictionary of validation metrics
        config: Model configuration
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint separately
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"Saved best model checkpoint to {best_path}")
    
    print(f"Saved checkpoint for epoch {epoch} to {checkpoint_path}")

def validate_model(
    model: GujaratiCaptioningModelV9,
    val_loader: DataLoader,
    tokenizer: MT5Tokenizer,
    evaluator: EvaluationMetrics,
    device: torch.device,
    max_samples: int
) -> Dict[str, float]:
    """
    Validate model with enhanced error handling and memory management.
    """
    model.eval()
    references = []
    hypotheses = []
    total_val_loss = 0.0
    
    try:
        # Add progress bar with clear description
        val_pbar = tqdm(val_loader, desc="Validating", total=min(len(val_loader), max_samples))
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                if batch_idx >= max_samples:
                    break
                
                try:
                    # Clear cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    # Move batch to device
                    images = batch['image'].to(device, dtype=torch.float32)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Forward pass
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    # Calculate loss
                    total_val_loss += outputs.loss.item()
                    
                    # Generate captions in smaller sub-batches to avoid OOM
                    sub_batch_size = 4  # Adjust based on your GPU memory
                    for i in range(0, len(images), sub_batch_size):
                        sub_images = images[i:i + sub_batch_size]
                        
                        # Generate captions for sub-batch
                        generated_ids = model.mt5.generate(
                            input_ids=input_ids[i:i + sub_batch_size, :1],  # Start token
                            max_length=64,
                            num_beams=4,
                            length_penalty=1.0,
                            early_stopping=True
                        )
                        
                        # Decode generated captions
                        generated_captions = [
                            tokenizer.decode(g, skip_special_tokens=True)
                            for g in generated_ids
                        ]
                        
                        # Store references and hypotheses
                        references.extend(batch['caption'][i:i + sub_batch_size])
                        hypotheses.extend(generated_captions)
                    
                    # Update progress bar with loss
                    val_pbar.set_postfix({'val_loss': f'{total_val_loss/(batch_idx+1):.4f}'})
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"OOM in validation batch {batch_idx}, skipping...")
                        continue
                    raise e
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics_batch(
            references=references,
            hypotheses=hypotheses,
            batch_size=32
        )
        
        # Add validation loss to metrics
        metrics['val_loss'] = total_val_loss / len(val_loader)
        
        # Log metrics
        print("\nValidation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    except Exception as e:
        print(f"Validation error: {str(e)}")
        # Return default metrics in case of error
        return {
            'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0,
            'meteor': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'val_loss': float('inf')
        }


def train_model_single_gpu(
    model: GujaratiCaptioningModelV9,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfig,
    tokenizer: MT5Tokenizer,
    checkpoint_dir: Path
) -> List[Dict[str, Any]]:
    """
    Revised training function with improved stability measures and error handling.
    """
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    device = torch.device(config.device)
    model = model.to(device)
    evaluator = EvaluationMetrics()
    
    # Print model and training info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize optimizer with lower learning rates
    optimizer = create_optimizer_with_layer_decay(
        model,
        vision_lr=5e-6,  # Lower initial learning rate
        text_lr=2e-5,    # Lower initial learning rate
        weight_decay=config.weight_decay,
    )
    
    # Verify optimizer initialization
    for param_group in optimizer.param_groups:
        if param_group['lr'] <= 0:
            raise ValueError("Invalid learning rate detected in optimizer")
    
    # Initialize gradient scaler with conservative settings
    scaler = torch.amp.GradScaler(
        enabled=config.mixed_precision,
        init_scale=2**8,     # Lower initial scale
        growth_factor=1.1,    # More conservative growth
        backoff_factor=0.5,   # Faster backoff
        growth_interval=2000  # Less frequent growth
    )
    
    # Learning rate scheduler with extended warmup
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(0.2 * total_steps)  # 20% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    
    best_meteor = -float('inf')
    train_history = []
    patience_counter = 0
    min_loss = float('inf')
    nan_loss_threshold = 5  # Maximum consecutive NaN losses before triggering early stopping
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for epoch in range(config.num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            optimizer.zero_grad(set_to_none=True)  # More efficient zeroing
            running_loss = 0.0
            running_count = 0
            accumulated_steps = 0
            consecutive_nan_count = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs} [Train]')
            
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    # Clear cache periodically
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    # Move batch to device and validate inputs
                    images = batch['image'].to(device, dtype=torch.float32)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Validate inputs
                    if torch.isnan(images).any() or torch.isinf(images).any():
                        print(f"Warning: Invalid values in input images at batch {batch_idx}")
                        continue
                    
                    # Clamp image values to prevent extremes
                    images = torch.clamp(images, -10, 10)
                    
                    # Calculate loss scale based on accumulation steps
                    loss_scale = 1.0 / config.gradient_accumulation_steps
                    
                    with torch.amp.autocast(device_type='cuda', enabled=config.mixed_precision):
                        outputs = model(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        # Revised loss handling
                        loss = outputs.loss
                        if config.gradient_accumulation_steps > 1:
                            loss = loss / config.gradient_accumulation_steps                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        consecutive_nan_count += 1
                        print(f"Warning: NaN/Inf loss detected at batch {batch_idx}")
                        
                        if consecutive_nan_count >= nan_loss_threshold:
                            raise ValueError("Too many consecutive NaN losses detected")
                        
                        continue
                    else:
                        consecutive_nan_count = 0
                    
                    # Gradient scaling and backward pass
                    scaler.scale(loss).backward()
                    
                    running_loss += loss.item()
                    running_count += 1
                    accumulated_steps += 1
                    
                    if (accumulated_steps % config.gradient_accumulation_steps == 0) or (batch_idx == len(train_loader) - 1):
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                        
                        # Clip gradients with lower threshold
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=0.05  # More aggressive clipping
                        )
                        
                        if torch.isfinite(grad_norm):
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            
                            # Update statistics
                            epoch_loss += running_loss
                            num_batches += 1
                            
                            current_avg_loss = epoch_loss / (num_batches + 1e-8)
                            current_lr = optimizer.param_groups[0]['lr']
                            
                            train_pbar.set_postfix({
                                'loss': f'{running_loss / (running_count + 1e-8):.4f}',
                                'avg_loss': f'{current_avg_loss:.4f}',
                                'lr': f'{current_lr:.2e}',
                                'nan_count': consecutive_nan_count
                            })
                            
                            running_loss = 0.0
                            running_count = 0
                            accumulated_steps = 0
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"\nOOM error in batch {batch_idx}, skipping batch")
                        continue
                    raise e
                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    continue
            
            # End of epoch logging
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"Average Training Loss: {avg_epoch_loss:.4f}")
                
                # Loss improvement tracking
                if avg_epoch_loss < min_loss:
                    min_loss = avg_epoch_loss
                    print(f"New minimum training loss: {min_loss:.4f}")
            
            # Validation step
            if (epoch + 1) % config.validation_frequency == 0:
                try:
                    print("\nStarting validation...")
                    model.eval()
                    
                    metrics = validate_model(
                        model=model,
                        val_loader=val_loader,
                        tokenizer=tokenizer,
                        evaluator=evaluator,
                        device=device,
                        max_samples=config.max_validation_samples
                    )
                    
                    print("\nValidation Metrics:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")
                    
                    # Save checkpoint if improved
                    if metrics['meteor'] > best_meteor:
                        improvement = metrics['meteor'] - best_meteor
                        best_meteor = metrics['meteor']
                        patience_counter = 0
                        print(f"\nMETEOR score improved by {improvement:.4f}")
                        
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
                        print(f"\nNo improvement for {patience_counter} validations")
                    
                    # Record training history
                    train_history.append({
                        'epoch': epoch + 1,
                        'train_loss': avg_epoch_loss,
                        'metrics': metrics,
                        'vision_lr': optimizer.param_groups[0]['lr'],
                        'text_lr': optimizer.param_groups[1]['lr']
                    })
                    
                except Exception as e:
                    print(f"Validation error: {str(e)}")
                    continue
                
                # Early stopping check
                if patience_counter >= config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best METEOR score: {best_meteor:.4f}")
                    break
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise
    finally:
        # Save final training history
        try:
            history_path = checkpoint_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(train_history, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving training history: {str(e)}")
    
    return train_history



# Modify the optimizer creation with lower initial learning rates
def create_optimizer_with_layer_decay(model, vision_lr=5e-6, text_lr=8e-6, weight_decay=0.01):
    vision_params = {"params": [p for n, p in model.named_parameters() if "vision_encoder" in n]}
    text_params = {"params": [p for n, p in model.named_parameters() if "vision_encoder" not in n]}
    
    optimizer = torch.optim.AdamW([
        {**vision_params, "lr": vision_lr, "weight_decay": weight_decay},
        {**text_params, "lr": text_lr, "weight_decay": weight_decay}
    ], eps=1e-8, betas=(0.9,0.999))  # Increased epsilon for numerical stability
    return optimizer



def main():
    """Main function for single-GPU training."""
    try:
        print("Starting main function...")

        # Load configuration
        config = ModelConfig.create_default_config()
        config.image_dir = r"/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/"  # Update this
        config.caption_file = r"/kaggle/input/guj-captions/gujarati_captions.txt"  # Update this
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
        
        print("Training completed successfully")
    
    except Exception as e:
        print(f"Fatal error in main function: {str(e)}")
        raise

if __name__ == '__main__':
    main()