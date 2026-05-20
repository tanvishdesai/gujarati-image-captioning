import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import os
from transformers.modeling_outputs import BaseModelOutput
from tqdm.auto import tqdm
import gc
import logging
import math
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch.nn.functional as F
from collections import defaultdict

# Configuration class for centralized parameter management
@dataclass
class ModelConfig:
    batch_size: int = 16
    eval_batch_size: int = 4
    learning_rate: float = 3e-5
    image_encoder_lr: float = 2e-5
    num_epochs: int = 1
    accumulation_steps: int = 4
    max_length: int = 64
    warmup_ratio: float = 0.15
    dropout_rate: float = 0.2
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4
    mixed_precision: bool = True
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    num_beams: int = 4
    checkpoint_dir: str = 'checkpoints'
    early_stopping_patience: int = 5  # New parameter
    min_learning_rate: float = 1e-6  # New parameter for cyclic learning rate
    cycles: int = 4  # New parameter for cyclic learning rate


class EnhancedFlickrGujaratiDataset(Dataset):
    def __init__(self, image_dir: str, captions_file: str, tokenizer, transform=None, max_length: int = 64):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Dictionary to store multiple captions per image
        self.image_captions: Dict[str, List[str]] = defaultdict(list)
        self.valid_images = set()
        
        print("Validating images and loading captions...")
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    caption = parts[1]
                    
                    image_path = os.path.join(image_dir, img_name)
                    if os.path.exists(image_path):
                        try:
                            # Validate image only once
                            if img_name not in self.valid_images:
                                with Image.open(image_path) as img:
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    # Verify image can be processed
                                    if self.transform:
                                        test_tensor = self.transform(img)
                                        if test_tensor.shape != (3, 224, 224):
                                            continue
                                self.valid_images.add(img_name)
                            self.image_captions[img_name].append(caption)
                        except Exception as e:
                            print(f"Error processing image {img_name}: {str(e)}")
                            continue
        
        # Create final list of image-caption pairs
        self.samples = [(img_name, captions) for img_name, captions in self.image_captions.items()]
        print(f"Dataset initialized with {len(self.samples)} valid images")
        print(f"Average captions per image: {sum(len(c) for _, c in self.samples)/len(self.samples):.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, captions = self.samples[idx]
        image_path = os.path.join(self.image_dir, img_name)
        
        # Randomly select one caption for training
        caption = np.random.choice(captions)
        
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                    
                    # Ensure consistent tensor size
                    if image.shape != (3, 224, 224):
                        raise ValueError(f"Incorrect image shape: {image.shape}")
                        
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return None to be filtered out by the collate_fn
            return None
        
        # Ensure consistent padding and tensor shape for captions
        encoded_caption = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by the tokenizer
        input_ids = encoded_caption['input_ids'].squeeze(0)
        attention_mask = encoded_caption['attention_mask'].squeeze(0)
        
        # Verify tensor sizes
        if input_ids.shape[0] != self.max_length or attention_mask.shape[0] != self.max_length:
            print(f"Incorrect caption tensor shape for {img_name}")
            return None
            
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_id': img_name,
            'all_captions': captions
        }

def custom_collate_fn(batch):
    """Custom collate function to handle None values and ensure consistent tensor sizes."""
    # Filter out None values
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
        
    # Collate the remaining valid samples
    return {
        'image': torch.stack([x['image'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'image_id': [x['image_id'] for x in batch],
        'all_captions': [x['all_captions'] for x in batch]
    }

class CrossAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = 1024  # mBART hidden size
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.attention_dropout = nn.Dropout(config.dropout_rate)
        self.output_dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, query, key_value):
        residual = query
        
        # Multi-head attention
        q = self.query(query)
        k = self.key(key_value)
        v = self.value(key_value)
        
        # Scaled dot-product attention
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_size)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        output = self.output_dropout(output)
        output = self.layer_norm(residual + output)
        
        return output

class EnhancedImageCaptioningModel(nn.Module):
    def __init__(self, mbart_model, config: ModelConfig):
        super().__init__()
        
        # Image encoder with preserved spatial information
        self.image_encoder = models.efficientnet_b0(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        
        self.mbart_hidden_size = mbart_model.config.hidden_size
        
        # Enhanced feature projection with spatial awareness
        self.feature_projection = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(config.dropout_rate),
            nn.Conv2d(512, self.mbart_hidden_size, kernel_size=1)
        )
        
        # Cross-attention mechanism
        self.cross_attention = CrossAttention(config)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.mbart_hidden_size * 2, self.mbart_hidden_size),
            nn.LayerNorm(self.mbart_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.mbart_hidden_size, self.mbart_hidden_size),
            nn.LayerNorm(self.mbart_hidden_size)
        )
        
        self.mbart = mbart_model
        
    def forward(self, images, input_ids=None, attention_mask=None):
        # Process image features
        image_features = self.image_encoder(images)  # [B, C, H, W]
        batch_size = image_features.size(0)
        
        # Project features while maintaining spatial information
        image_features = self.feature_projection(image_features)  # [B, hidden_size, H, W]
        
        # Reshape for attention
        H, W = image_features.shape[-2:]
        image_features = image_features.flatten(2).transpose(1, 2)  # [B, H*W, hidden_size]
        
        if input_ids is None:
            # Inference mode
            return {
                'last_hidden_state': image_features,
                'attentions': None
            }
        
        # Get text features from encoder
        encoder_outputs = self.mbart.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Apply cross-attention between text and image features
        cross_attended_features = self.cross_attention(encoder_hidden_states, image_features)
        
        # Combine features
        combined_features = torch.cat([encoder_hidden_states, cross_attended_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        # Generate caption
        outputs = self.mbart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=[fused_features],
            return_dict=True
        )
        
        return outputs.logits


class CyclicLR:
    def __init__(self, optimizer, max_lr, min_lr, cycles, total_steps):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycles = cycles
        self.total_steps = total_steps
        self.current_step = 0
        
    def step(self):
        # Calculate the current position in the cycle
        cycle_progress = (self.current_step % (self.total_steps // self.cycles)) / (self.total_steps // self.cycles)
        
        # Cosine annealing formula
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        
        # Calculate learning rate
        lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        
        # Update learning rate for each parameter group
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * (param_group['lr'] / self.max_lr)
        
        self.current_step += 1
        
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`."""
        return {
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'cycles': self.cycles,
            'total_steps': self.total_steps,
            'current_step': self.current_step
        }
    
    # Add load_state_dict method
    def load_state_dict(self, state_dict):
        """Loads the scheduler state.
        
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.max_lr = state_dict['max_lr']
        self.min_lr = state_dict['min_lr']
        self.cycles = state_dict['cycles']
        self.total_steps = state_dict['total_steps']
        self.current_step = state_dict['current_step']

class ImageCaptioningTrainer:
    def __init__(self, model, config: ModelConfig, tokenizer, device):
        """Initialize the trainer with model and training configurations.
        
        The trainer now includes support for cyclic learning rates and early stopping.
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize optimizer with parameter groups
        self.optimizer = self._create_optimizer()
        
        # Initialize early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Ensure checkpoint directory exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def _create_optimizer(self):
        """Create optimizer with separate parameter groups for different learning rates."""
        return optim.AdamW([
            {'params': self.model.image_encoder.parameters(), 
             'lr': self.config.image_encoder_lr},
            {'params': list(self.model.feature_projection.parameters()) + 
                      list(self.model.cross_attention.parameters()) + 
                      list(self.model.fusion_layer.parameters()),
             'lr': self.config.learning_rate},
            {'params': self.model.mbart.parameters(), 
             'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay)
    
    def _should_stop_early(self, val_loss):
        """Check if training should stop based on validation loss and patience."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict()
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                return True
        return False
    
    # def _save_checkpoint(self, epoch, filename, metrics=None):
    #     """Save a model checkpoint with current state and metrics."""
    #     checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'scheduler_state_dict': self.scheduler.state_dict(),
    #         'best_val_loss': self.best_val_loss,
    #         'metrics': metrics
    #     }
    #     torch.save(checkpoint, checkpoint_path)
    #     print(f"\nSaved checkpoint: {checkpoint_path}")
    
    def train(self, train_loader, val_loader, test_loader):
        """Execute the complete training loop with improved monitoring and checkpointing."""
        # Calculate total steps for cyclic learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        
        # Initialize cyclic learning rate scheduler
        self.scheduler = CyclicLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            min_lr=self.config.min_learning_rate,
            cycles=self.config.cycles,
            total_steps=total_steps
        )
        
        # Initialize criterion for loss calculation
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        best_bleu4 = 0
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_train_loss = 0
            batch_count = 0
            
            # Training loop with progress bar
            progress_bar = tqdm(train_loader, 
                              desc=f'Training Epoch {epoch+1}/{self.config.num_epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                if batch is None:
                    continue
                    
                self.batch_idx = batch_idx
                loss = self._training_step(batch, criterion)
                
                if loss is not None:
                    total_train_loss += loss
                    batch_count += 1
                    
                    # Update progress bar with current loss
                    progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Periodic memory cleanup
                if batch_count % 100 == 0:
                    torch.cuda.empty_cache()
            
            # Calculate average training loss
            avg_train_loss = total_train_loss / batch_count if batch_count > 0 else float('inf')
            
            # Run validation
            avg_val_loss = self._validate(val_loader, criterion)
            
            # Store training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            # Run evaluation every 5 epochs or on final epoch
            if epoch == self.config.num_epochs - 1:
                print("\nRunning intermediate evaluation...")
                eval_results = self.evaluate(test_loader)
                current_bleu4 = eval_results['bleu4']
                
                print(f"Current BLEU-4: {current_bleu4:.4f}")
                
                # Save best model based on BLEU-4 score
                if current_bleu4 > best_bleu4:
                    best_bleu4 = current_bleu4
                    # self._save_checkpoint(
                    #     epoch,
                    #     'best_bleu4_model.pth',
                    #     metrics={'bleu4': current_bleu4}
                    # )
            
            # Check for early stopping
            if self._should_stop_early(avg_val_loss):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                # Restore best model
                self.model.load_state_dict(self.best_model_state)
                break
            
            # Save regular checkpoint every 10 epochs
            # if (epoch + 1) % 10 == 0:
                # self._save_checkpoint(
                #     epoch,
                #     f'checkpoint_epoch_{epoch+1}.pth',
                #     metrics={'train_loss': avg_train_loss, 'val_loss': avg_val_loss}
                # )
        
        # Save final training history
        history_path = os.path.join(self.config.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        
        # Save final model
        self._save_final_model()
        
        # Run final evaluation
        print("\nTraining completed. Starting final evaluation...")
        evaluation_results = self.evaluate(test_loader)
        
        return evaluation_results
    
    def _training_step(self, batch, criterion):
        """Execute a single training step with mixed precision support."""
        if batch is None:
            return None
        
        # Move batch to device
        images = batch['image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.mixed_precision):
            outputs = self.model(images, input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            loss = loss / self.config.accumulation_steps
        
        # Backward pass with gradient scaling
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation and optimizer step
        if (self.batch_idx + 1) % self.config.accumulation_steps == 0:
            if self.config.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.config.accumulation_steps
    
    def _validate(self, val_loader, criterion):
        """Run validation with memory optimization."""
        self.model.eval()
        total_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                if batch is None:
                    continue
                
                # Move batch to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(images, input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                
                total_val_loss += loss.item()
                val_batch_count += 1
                
                # Periodic memory cleanup
                if val_batch_count % 50 == 0:
                    torch.cuda.empty_cache()
        
        return total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
    
    def _save_final_model(self):
        """Save the final model state with complete training information."""
        final_model_path = os.path.join(
            self.config.checkpoint_dir, 
            'final_model.pth'
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'final_epoch': self.config.num_epochs,
            'best_val_loss': self.best_val_loss
        }, final_model_path)
        print(f"Saved final model: {final_model_path}")
    def generate_caption(self, image):
        """Generate a caption for a single image.
        
        Args:
            image: Input image tensor
            
        Returns:
            caption: Generated caption string
        """
        self.model.eval()
        with torch.no_grad():
            try:
                image = image.to(self.device)
                
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                
                encoder_outputs = self.model(image)
                last_hidden_state = encoder_outputs['last_hidden_state']
                
                # Create dummy input for the decoder
                dummy_input_ids = torch.ones((1, 1), dtype=torch.long, device=self.device)
                
                # Prepare encoder outputs
                proper_encoder_outputs = BaseModelOutput(
                    last_hidden_state=last_hidden_state,
                    hidden_states=None,
                    attentions=None
                )
                
                # Generate caption
                outputs = self.model.mbart.generate(
                    input_ids=dummy_input_ids,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True,
                    encoder_outputs=proper_encoder_outputs,
                    return_dict_in_generate=False
                )
                
                caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return caption
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"WARNING: OOM error. Current GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                    return ""
                raise e
    
    def evaluate(self, test_loader):
        """Evaluate the model using multiple metrics.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            results: Dictionary containing evaluation metrics
        """
        self.model.eval()
        smooth = SmoothingFunction()
        rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Store references and hypotheses
        image_references = defaultdict(list)
        image_hypotheses = {}
        
        print("Generating captions for evaluation...")
        for batch in tqdm(test_loader):
            if batch is None:
                continue
            
            image_id = batch['image_id'][0]
            all_captions = batch['all_captions'][0]
            
            # Generate caption
            generated_caption = self.generate_caption(batch['image'])
            
            if generated_caption:
                image_references[image_id].extend(all_captions)
                image_hypotheses[image_id] = generated_caption
        
        # Prepare for BLEU calculation
        references = []
        hypotheses = []
        
        for image_id in image_hypotheses.keys():
            references.append([self._tokenize_gujarati(ref) for ref in image_references[image_id]])
            hypotheses.append(self._tokenize_gujarati(image_hypotheses[image_id]))
        
        # Calculate BLEU scores
        bleu_scores = {}
        weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
        for i, w in enumerate(weights, 1):
            bleu_scores[f'bleu{i}'] = corpus_bleu(references, hypotheses, 
                                                weights=w, 
                                                smoothing_function=smooth.method1)
        
        # Calculate METEOR and ROUGE scores
        meteor_scores = []
        rouge_scores = defaultdict(list)
        
        for refs, hyp in zip(references, hypotheses):
            meteor_scores.append(max(meteor_score([ref], hyp) for ref in refs))
            
            best_rouge_scores = None
            best_rouge_total = -1
            
            for ref in refs:
                ref_text = ' '.join(ref)
                hyp_text = ' '.join(hyp)
                
                rouge_result = rouge_calculator.score(ref_text, hyp_text)
                rouge_total = sum(score.fmeasure for score in rouge_result.values())
                
                if rouge_total > best_rouge_total:
                    best_rouge_scores = rouge_result
                    best_rouge_total = rouge_total
            
            rouge_scores['rouge1'].append(best_rouge_scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(best_rouge_scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(best_rouge_scores['rougeL'].fmeasure)
        
        # Compile and save results
        results = {
            **bleu_scores,
            'meteor': np.mean(meteor_scores),
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
        
        results_path = os.path.join(self.config.checkpoint_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        return results
    
    def _tokenize_gujarati(self, text):
        """Tokenize Gujarati text with proper handling of punctuation.
        
        Args:
            text: Input text string
            
        Returns:
            tokens: List of tokens
        """
        punctuation = ".,!?।॥''""()[]{}:;-"
        
        for p in punctuation:
            text = text.replace(p, f" {p} ")
        
        tokens = [token.strip() for token in text.split() if token.strip()]
        return tokens


def main():
    # Set environment variables for better GPU memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cudnn.benchmark = True
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize config
    config = ModelConfig()
    
    # Set device and memory optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Set up automatic mixed precision
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # Initialize tokenizer with proper language settings
    tokenizer = MBartTokenizer.from_pretrained(
        "facebook/mbart-large-cc25",
        src_lang="gu_IN",
        tgt_lang="gu_IN"
    )
    
    # Load mBART model with gradient checkpointing and optimizations
    mbart_config = MBartConfig.from_pretrained("facebook/mbart-large-cc25")
    mbart_config.gradient_checkpointing = True
    mbart_config.dropout = config.dropout_rate  # Add dropout from config
    
    print("Loading mBART model...")
    mbart_model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-cc25",
        config=mbart_config
    )
    
    # Enhanced data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size[0] + 32, config.image_size[1] + 32)),
        transforms.RandomCrop(config.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Separate validation/test transform without augmentation
    eval_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("Creating datasets...")
    # Create datasets with appropriate transforms
    train_dataset = EnhancedFlickrGujaratiDataset(
        "/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/",
        "/kaggle/input/guj-captions/gujarati_captions.txt",
        tokenizer,
        train_transform,
        max_length=config.max_length
    )
    
    # Create validation and test datasets with eval transform
    total_size = len(train_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Update transforms for validation and test datasets
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform
    
    print("Creating dataloaders...")
    # Create dataloaders with proper batch sizes and workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,  # Use eval_batch_size for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,  # Use eval_batch_size for testing
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    print("Initializing model...")
    # Initialize model with cross-attention and move to device
    model = EnhancedImageCaptioningModel(mbart_model, config).to(device)
    
    # Enable gradient checkpointing if using cuda
    if torch.cuda.is_available():
        model.mbart.gradient_checkpointing_enable()
    
    print("Setting up trainer...")
    # Create trainer with updated config
    trainer = ImageCaptioningTrainer(model, config, tokenizer, device)
    
    try:
        print("Starting training...")
        # Train and evaluate
        results = trainer.train(train_loader, val_loader, test_loader)
        
        # Save final results
        results_path = os.path.join(config.checkpoint_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print("Training and evaluation completed successfully!")
        print(f"Final results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()