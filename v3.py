from datetime import datetime
from transformers import GPT2Config, GPT2LMHeadModel
import os
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import (
    XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,
    AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer
)
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider  # Import CIDEr scorer
import random
import json

class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        tokenizer: XLMRobertaTokenizer,
        transform=None,
        max_length: int = 128,
        augment: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        # Define default transforms if none provided
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize translation models if augmentation is enabled
        if self.augment:
            try:
                self.gu_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-gu-en")
                self.gu_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-gu-en")
                self.en_gu_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-gu")
                self.en_gu_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-gu")
            except Exception as e:
                print(f"Warning: Translation models could not be loaded. Disabling augmentation. Error: {e}")
                self.augment = False
        
        # Parse captions file
        self.image_captions = {}
        with open(captions_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        image_id, caption = parts
                        image_name = image_id.split('#')[0]
                        if image_name not in self.image_captions:
                            self.image_captions[image_name] = []
                        self.image_captions[image_name].append(caption)
                except Exception as e:
                    print(f"Error processing line: {line.strip()}")
                    print(f"Error: {str(e)}")
                    continue
        
        if not self.image_captions:
            raise ValueError("No valid captions found in file")
            
        self.image_names = list(self.image_captions.keys())
        print(f"Loaded {len(self.image_names)} images with captions")

    def __len__(self) -> int:
        """Return the total number of image-caption pairs in the dataset."""
        return sum(len(captions) for captions in self.image_captions.values())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an image-caption pair by index."""
        # Find the corresponding image and caption
        total = 0
        for image_name, captions in self.image_captions.items():
            if total + len(captions) > idx:
                caption_idx = idx - total
                caption = captions[caption_idx]
                break
            total += len(captions)
        else:
            raise IndexError("Index out of bounds")

        # Load and transform image
        try:
            image_path = self.image_dir / image_name
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_name}: {str(e)}")
            # Return a zero tensor of the correct shape as a fallback
            image = torch.zeros((3, 224, 224))

        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'caption': caption
        }
class ImageEncoder(nn.Module):
    def __init__(self, encoded_dim: int = 2048):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.projection = nn.Linear(2048, encoded_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.flatten(2).transpose(1, 2)
        return self.projection(features)

class GujaratiImageCaptioning(nn.Module):
    def __init__(self, config: XLMRobertaConfig):
        """
        Initialize the Gujarati Image Captioning model with compatible dimensions.
        The architecture consists of three main components:
        1. Image Encoder: ResNet50 backbone with dimension matching projection
        2. Text Encoder: Modified XLM-RoBERTa with reduced complexity
        3. Decoder: Modified GPT-2 with cross-attention
        
        Args:
            config (XLMRobertaConfig): Configuration for the model dimensions
                                     Should have compatible hidden_size and num_attention_heads
        """
        super().__init__()
        
        # Ensure embedding dimension is divisible by number of heads
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({config.hidden_size}) must be divisible by number of attention heads "
                f"({config.num_attention_heads})"
            )
        
        # Image encoder with matching dimensions
        # The output will match the transformer's hidden size
        self.encoder = ImageEncoder(encoded_dim=config.hidden_size)
        
        # Text encoder (XLM-RoBERTa)
        # Using the modified config with compatible dimensions
        self.roberta = XLMRobertaModel(config)
        
        # Initialize GPT-2 with matching dimensions
        decoder_config = GPT2Config.from_pretrained('gpt2')
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        
        # Match GPT-2 dimensions with RoBERTa
        decoder_config.n_embd = config.hidden_size
        decoder_config.n_head = config.num_attention_heads
        decoder_config.n_layer = config.num_hidden_layers
        
        # Vocabulary size should match the tokenizer
        decoder_config.vocab_size = config.vocab_size
        
        self.decoder = GPT2LMHeadModel(decoder_config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            images (torch.Tensor): Input images [batch_size, 3, height, width]
            input_ids (torch.Tensor): Token IDs [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Output logits [batch_size, seq_length, vocab_size]
        """
        # Encode images to match hidden size
        image_features = self.encoder(images)
        
        # Get text embeddings from RoBERTa
        text_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = text_outputs.last_hidden_state
        
        # Pass through decoder with cross-attention to image features
        decoder_outputs = self.decoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=image_features,
            use_cache=False,
            output_hidden_states=True
        )
        
        return decoder_outputs.logits

    def generate(self, image, tokenizer, max_length=128, num_beams=5, temperature=1.0, device=None):
        """
        Generate a caption for a single image.
        
        Args:
            image (torch.Tensor): Input image [1, 3, height, width]
            tokenizer: Tokenizer for text generation
            max_length (int): Maximum caption length
            num_beams (int): Number of beams for beam search
            temperature (float): Sampling temperature
            device (torch.device): Device to use
            
        Returns:
            str: Generated caption
        """
        if device is not None:
            image = image.to(device)
            self.to(device)
            
        with torch.no_grad():
            # Encode image
            image_features = self.encoder(image.unsqueeze(0))
            
            # Generate caption
            input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
            outputs = self.decoder.generate(
                input_ids=input_ids,
                encoder_hidden_states=image_features,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True
            )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)


    
def calculate_metrics(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate CIDEr score
    cider_scorer = Cider()
    cider_score = cider_scorer.compute_score(references, hypotheses)[0]
    
    bleu = corpus_bleu([[ref] for ref in references], hypotheses)
    meteor = np.mean([meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)])
    
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    for ref, hyp in zip(references, hypotheses):
        scores = rouge_scorer_instance.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    
    for key in rouge_scores:
        rouge_scores[key] /= len(references)
    
    return {
        'bleu': bleu,
        'meteor': meteor,
        'cider': cider_score,
        **rouge_scores
    }


def beam_search_with_length_norm(model, image_features, tokenizer, beam_width=5, max_length=128):
    # Initialize all beams at once
    sequences = torch.full((beam_width, 1), tokenizer.bos_token_id, device=image_features.device)
    scores = torch.zeros(beam_width, device=image_features.device)
    
    for step in range(max_length):
        # Get predictions for all beams at once
        with torch.no_grad():
            logits = model.decoder(
                model.roberta(sequences, attention_mask=None).last_hidden_state,
                image_features.repeat(beam_width, 1, 1)
            )[:, -1, :]
            
        # Calculate scores for all possible next tokens
        next_scores = scores.unsqueeze(1) - torch.log_softmax(logits, dim=-1)
        next_scores = next_scores.view(-1)
        
        # Select top k scores and their corresponding tokens
        topk_scores, topk_indices = next_scores.topk(beam_width, largest=False)
        beam_indices = topk_indices // logits.size(-1)
        token_indices = topk_indices % logits.size(-1)
        
        # Update sequences and scores
        sequences = torch.cat([sequences[beam_indices], token_indices.unsqueeze(1)], dim=1)
        scores = topk_scores


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    tokenizer: XLMRobertaTokenizer,
    scheduler=None,
    grad_clip: float = 1.0,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 4  # Accumulate gradients across multiple steps
):
    """
    Memory-optimized training loop for the image captioning model.
    """
    history = {'train_loss': [], 'metrics': []}
    scaler = GradScaler()
    
    # Reduce precision to save memory
    model.half()  # Convert model to half precision
    
    # Track memory usage
    def print_gpu_memory():
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, "
                  f"{torch.cuda.memory_reserved() / 1024**2:.2f}MB reserved")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        # Reduce effective batch size while maintaining same number of samples
        actual_batch_size = batch_size // gradient_accumulation_steps
        
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        print_gpu_memory()
        
        for i, batch in enumerate(tqdm(train_loader)):
            try:
                # Move data to GPU in half precision
                images = batch['image'].to(device, dtype=torch.float16)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Clear cache if memory is running low
                if torch.cuda.memory_allocated() > 3.5 * (1024**3):  # If over 3.5GB used
                    torch.cuda.empty_cache()
                
                with autocast():
                    # Forward pass
                    outputs = model(images, input_ids, attention_mask)
                    
                    # Calculate loss
                    labels = input_ids[:, 1:].contiguous()
                    shift_logits = outputs[:, :-1, :].contiguous()
                    loss = criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        labels.view(-1)
                    )
                    
                    # Scale loss by gradient accumulation steps
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update weights only after accumulating gradients
                if (i + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * gradient_accumulation_steps
                
                # Free up memory
                del outputs, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        if scheduler:
            scheduler.step()
            
        # Calculate epoch metrics
        epoch_loss = train_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}")
        print_gpu_memory()
        
        # Save checkpoint with reduced precision
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'history': history
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
    
    return history

# Modified DataLoader creation with memory optimizations
def create_memory_efficient_loader(dataset, batch_size, num_workers=2):
    """
    Creates a memory-efficient DataLoader with pinned memory and appropriate batch size.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep worker processes alive between iterations
        prefetch_factor=2,  # Reduce prefetching to save memory
        drop_last=True  # Drop incomplete batches to maintain consistent memory usage
    )

def main():
    """
    Main function for training the Gujarati image captioning model.
    Implements memory optimizations, ensures dimensional compatibility, and sets up a robust training pipeline.
    """
    # Set random seeds for reproducibility across all libraries
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    # Device configuration with detailed GPU settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Enable performance optimizations for GPU training
        torch.backends.cudnn.benchmark = True  # Optimize cudnn for fixed input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for better performance
        torch.cuda.set_per_process_memory_fraction(0.9)  # Reserve some GPU memory for system
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Reduce memory fragmentation
    
    # Training hyperparameters - carefully tuned for stability and performance
    hyperparams = {
        'batch_size': 8,  # Smaller batch size for better memory usage
        'gradient_accumulation_steps': 4,  # Effectively increases batch size to 32
        'learning_rate': 2e-5,  # Conservative learning rate for stability
        'weight_decay': 0.01,  # L2 regularization to prevent overfitting
        'num_epochs': 10,
        'warmup_steps': 100,  # Gradual learning rate warmup
        'max_grad_norm': 1.0,  # Prevent exploding gradients
        'num_workers': 2  # Dataloader workers - adjust based on CPU cores
    }
    
    try:
        print("Initializing tokenizer and model configuration...")
        # Initialize tokenizer and base configuration
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
        
        # Modify configuration for efficiency and compatibility
        # These values are carefully chosen to maintain model capacity while reducing memory usage
        config.hidden_size = 512        # Must be divisible by num_attention_heads
        config.num_attention_heads = 8  # Reduced for compatibility with hidden_size
        config.num_hidden_layers = 6    # Reduced for memory efficiency
        config.intermediate_size = 2048 # Set to 4x hidden_size as per transformer literature
        
        # Validate dimensional compatibility
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({config.hidden_size}) must be divisible by number of attention heads "
                f"({config.num_attention_heads})"
            )
            
    except Exception as e:
        print(f"Error initializing tokenizer/config: {str(e)}")
        return
    
    try:
        print("Loading dataset...")
        # Initialize dataset with optimized transform pipeline
        train_dataset = ImageCaptioningDataset(
            image_dir='flickr8k/Flickr_Data/Flickr_Data/Images',
            captions_file='gujarati_captions.txt',
            tokenizer=tokenizer,
            transform=transforms.Compose([
                transforms.Resize((160, 160)),  # Reduced image size for memory efficiency
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                    std=[0.229, 0.224, 0.225]
                )
            ]),
            max_length=64,  # Reduced sequence length for efficiency
            augment=False   # Disable augmentation initially
        )
        
        # Create memory-efficient dataloader
        train_loader = create_memory_efficient_loader(
            dataset=train_dataset,
            batch_size=hyperparams['batch_size'],
            num_workers=hyperparams['num_workers']
        )
        
        print(f"Dataset size: {len(train_dataset)} samples")
        print(f"Number of batches: {len(train_loader)}")
        
    except Exception as e:
        print(f"Error creating dataset/dataloader: {str(e)}")
        return
    
    try:
        print("Initializing model...")
        # Initialize model with validated configuration
        model = GujaratiImageCaptioning(config=config)
        model = model.to(device)
        
        # Enable mixed precision training
        model = model.half()
        print("Model initialized successfully")
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return
    
    try:
        print("Setting up optimizer and scheduler...")
        # Initialize optimizer with weight decay for regularization
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparams['learning_rate'],
            betas=(0.9, 0.999),  # Default Adam betas
            eps=1e-8,
            weight_decay=hyperparams['weight_decay']
        )
        
        # Calculate total training steps for scheduler
        num_training_steps = len(train_loader) * hyperparams['num_epochs']
        
        # Create learning rate scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=hyperparams['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # Initialize loss function, ignoring padding tokens
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
    except Exception as e:
        print(f"Error setting up optimizer/scheduler: {str(e)}")
        return
    
    # Create directory for checkpoints and logs
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save training configuration for reproducibility
    config_path = checkpoint_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'hyperparameters': hyperparams,
            'model_config': config.to_dict(),
            'device': device.type,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    try:
        print("Starting training...")
        # Train model with memory optimizations
        history = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=hyperparams['num_epochs'],
            device=device,
            tokenizer=tokenizer,
            scheduler=scheduler,
            batch_size=hyperparams['batch_size'],
            gradient_accumulation_steps=hyperparams['gradient_accumulation_steps']
        )
        
        # Save final results and training history
        final_results = {
            'training_history': history,
            'final_loss': history['train_loss'][-1],
            'final_metrics': history.get('metrics', [])[-1] if history.get('metrics') else None
        }
        
        with open(checkpoint_dir / 'training_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
            
        print("Training completed successfully!")
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Attempt to save emergency checkpoint
        try:
            emergency_path = checkpoint_dir / 'emergency_checkpoint.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'error': str(e),
                'epoch': history['train_loss'][-1] if 'history' in locals() else None
            }, emergency_path)
            print(f"Emergency checkpoint saved to {emergency_path}")
        except:
            print("Could not save emergency checkpoint")
        return None

if __name__ == "__main__":
    main()