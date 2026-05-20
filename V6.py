import itertools
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForPreTraining, BertConfig
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import os
from typing import Dict, List, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import json
from datetime import datetime
import re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class LightweightImageEncoder(nn.Module):
    """
    Efficient image encoder using MobileNetV2 backbone with custom projection layer.
    Processes images into a format suitable for integration with Gujarati-BERT.
    """
    def __init__(self, encoded_dim=768):  # Changed to match BERT hidden size
        super().__init__()
        # Load pretrained MobileNetV2 and remove classification layer
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # Add normalization and projection layers
        self.bn = nn.BatchNorm1d(1280)
        self.projection = nn.Linear(1280, encoded_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, images):
        features = self.features(images)
        features = features.mean([2, 3])  # Global average pooling
        features = self.bn(features)
        features = self.dropout(features)
        return self.projection(features)

class GujaratiCaptioningModel(nn.Module):
    """
    Enhanced image captioning model using Gujarati-BERT for native caption generation.
    Combines visual features with language understanding through cross-attention.
    """
    def __init__(self, 
                 bert_model_name="ai4bharat/indic-bert",
                 freeze_encoder_layers=6):
        super().__init__()
        
        # Initialize Gujarati-BERT model
        self.bert_config = BertConfig.from_pretrained(bert_model_name)
        self.bert = BertForPreTraining.from_pretrained(bert_model_name)
        self.image_encoder = LightweightImageEncoder(encoded_dim=self.bert_config.hidden_size)
        
        # Freeze initial layers of BERT encoder for stability
        for i, param in enumerate(self.bert.bert.encoder.layer[:freeze_encoder_layers].parameters()):
            param.requires_grad = False
            
        # Cross-attention mechanism for combining image and text features
        self.image_cross_attention = nn.MultiheadAttention(
            embed_dim=self.bert_config.hidden_size,
            num_heads=self.bert_config.num_attention_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Additional layers for feature processing
        self.image_projection = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size),
            nn.LayerNorm(self.bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Layer normalization for feature combination
        self.feature_norm = nn.LayerNorm(self.bert_config.hidden_size)
        
        # Caption generation head
        self.caption_head = nn.Linear(self.bert_config.hidden_size, self.bert_config.vocab_size)
        
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        # Encode and project image features
        image_features = self.image_encoder(images)
        image_features = self.image_projection(image_features).unsqueeze(1)
        
        # Get BERT encoder outputs
        encoder_outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Combine text and image features using cross-attention
        enhanced_features, _ = self.image_cross_attention(
            query=encoder_outputs.last_hidden_state,
            key=image_features,
            value=image_features
        )
        
        # Normalize and combine features
        enhanced_features = self.feature_norm(
            enhanced_features + encoder_outputs.last_hidden_state
        )
        
        # Generate logits for caption prediction
        logits = self.caption_head(enhanced_features)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.bert_config.vocab_size), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
    
    def generate_caption(self, image, tokenizer, max_length=64, num_beams=4, device='cuda'):
        """Generate a caption for a given image using beam search."""
        self.eval()
        with torch.no_grad():
            # Encode and project image features
            image_features = self.image_encoder(image.unsqueeze(0).to(device))
            image_features = self.image_projection(image_features).unsqueeze(1)
            
            # Initialize generation
            input_ids = torch.tensor([[tokenizer.cls_token_id]]).to(device)
            
            # Generate caption using beam search
            for _ in range(max_length):
                attention_mask = torch.ones_like(input_ids)
                
                outputs = self.forward(
                    images=image.unsqueeze(0).to(device),
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                next_token_logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                if next_token.item() == tokenizer.sep_token_id:
                    break
                    
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)

class GujaratiPreprocessor:
    """Handle Gujarati-specific text preprocessing"""
    def __init__(self):
        self.normalizer = IndicNormalizerFactory().get_normalizer("gu")
    
    def normalize_text(self, text):
        # Normalize Gujarati text
        normalized = self.normalizer.normalize(text)
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def tokenize(self, text):
        # Gujarati-specific tokenization
        return indic_tokenize.trivial_tokenize(text, lang='gu')

class EnhancedDataset(Dataset):
    """
    Dataset class with improved data handling and augmentation capabilities.
    Supports Gujarati captions and includes robust error handling.
    """
    def __init__(self, image_dir, captions_file, tokenizer, max_length=64, is_train=True):
        super().__init__()
        self.gujarati_processor = GujaratiPreprocessor()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Enhanced image transforms with augmentation for training
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        
        # Load and preprocess captions
        self.samples = []
        skipped_images = 0
        
        with open(captions_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        image_name, caption = parts
                        image_path = Path(image_dir) / image_name.split('#')[0]
                        
                        if image_path.exists():
                            # Normalize and preprocess Gujarati caption
                            normalized_caption = self.gujarati_processor.normalize_text(caption)
                            self.samples.append((image_name.split('#')[0], normalized_caption))
                        else:
                            skipped_images += 1
                            
                except Exception as e:
                    logging.warning(f"Error processing line: {line.strip()}, Error: {str(e)}")
                    continue
        
        logging.info(f"Loaded {len(self.samples)} valid image-caption pairs")
        logging.info(f"Skipped {skipped_images} images due to missing files")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]
        
        try:
            image = Image.open(Path(self.image_dir) / image_name).convert('RGB')
            image = self.transform(image)
            
            # Tokenize with Gujarati-specific processing
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
                'image': image,
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'caption': processed_caption
            }
        except Exception as e:
            logging.error(f"Error loading item {idx}, image: {image_name}, Error: {str(e)}")
            # Return empty item with same structure
            return {
                'image': torch.zeros((3, 224, 224)),
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'caption': ""
            }

def calculate_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE metrics for generated captions during training.
    """
    try:
        rouge_scorer_inst = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                   use_stemmer=True)
        
        scores = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }
        
        valid_pairs = 0
        for ref, hyp in zip(references, hypotheses):
            try:
                # Calculate ROUGE scores
                rouge_scores = rouge_scorer_inst.score(ref, hyp)
                scores['rouge1'] += rouge_scores['rouge1'].fmeasure
                scores['rouge2'] += rouge_scores['rouge2'].fmeasure
                scores['rougeL'] += rouge_scores['rougeL'].fmeasure
                
                valid_pairs += 1
                
            except Exception as e:
                logging.warning(f"Error calculating metrics for pair: {str(e)}")
                continue
        
        # Average scores
        if valid_pairs > 0:
            for key in scores:
                scores[key] /= valid_pairs
        
        return scores
    
    except Exception as e:
        logging.error(f"Error in calculate_metrics: {str(e)}")
        return {k: 0.0 for k in ['rouge1', 'rouge2', 'rougeL']}



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    tokenizer,
    checkpoint_dir: Path,
    gradient_clip_val: float = 1.0,
    early_stopping_patience: int = 5,
    save_frequency: int = 1,
    gradient_accumulation_steps: int = 8,
    validation_frequency: int = 2  # Only validate every N epochs
):
    """
    Optimized training function with reduced validation overhead
    """
    best_loss = float('inf')
    train_history = []
    patience_counter = 0
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_idx, batch in enumerate(train_pbar):
            try:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                # Calculate loss and scale it
                loss = outputs["loss"] / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip_val
                        )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Clean up GPU memory
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.error(f"Error in training batch: {str(e)}")
                continue
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Only run validation at specified frequency
        do_validation = (val_loader is not None and 
                        ((epoch + 1) % validation_frequency == 0 or epoch == num_epochs - 1))
        
        if do_validation:
            model.eval()
            val_loss = 0.0
            references = []
            hypotheses = []
            
            # Validate on a subset of validation data to speed up training
            subset_size = min(100, len(val_loader))  # Limit validation samples
            val_pbar = tqdm(itertools.islice(val_loader, subset_size), 
                          desc=f'Epoch {epoch + 1}/{num_epochs} [Val]',
                          total=subset_size)
            
            with torch.no_grad():
                for batch in val_pbar:
                    try:
                        images = batch['image'].to(device)
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        captions = batch['caption']
                        
                        # Forward pass for validation
                        outputs = model(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        
                        loss = outputs["loss"]
                        val_loss += loss.item()
                        
                        # Generate caption for only one sample per batch to speed up validation
                        hyp = model.generate_caption(
                            images[0], 
                            tokenizer,
                            device=device
                        )
                        references.append(captions[0])
                        hypotheses.append(hyp)
                            
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                            
                    except Exception as e:
                        logging.error(f"Error in validation batch: {str(e)}")
                        continue
            
            avg_val_loss = val_loss / subset_size
            metrics = calculate_metrics(references, hypotheses)
            
            logging.info(f"\nEpoch {epoch + 1} Summary:")
            logging.info(f"Training Loss: {avg_train_loss:.4f}")
            logging.info(f"Validation Loss: {avg_val_loss:.4f}")
            logging.info("ROUGE Metrics:")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}")
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                
                if (epoch + 1) % save_frequency == 0:
                    checkpoint_path = checkpoint_dir / f'best_model_epoch_{epoch + 1}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_loss,
                        'rouge_metrics': metrics
                    }, checkpoint_path)
                    logging.info(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        else:
            # If not validating this epoch, just save training metrics
            train_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Save checkpoint based on training loss
            if (epoch + 1) % save_frequency == 0:
                checkpoint_path = checkpoint_dir / f'model_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_train_loss,
                }, checkpoint_path)
                logging.info(f"Saved model checkpoint to {checkpoint_path}")
    
    return train_history
def main():
    """
    Optimized main function with improved configuration and resource handling.
    """
    try:
        # Define training configuration with optimized parameters
        config = {
            'batch_size': 8,  # Increased batch size
            'hidden_dim': 768,  # Matches BERT hidden dimension
            'max_length': 48,
            'gradient_accumulation_steps': 4,  # Reduced from 8
            'learning_rate': 0.001,
            'num_epochs': 5,
            'image_dir': Path('flickr8k/Flickr_Data/Flickr_Data/Images'),
            'captions_file': 'gujarati_captions.txt',
            'checkpoint_dir': Path('checkpoints'),
            'num_workers': 0 if os.name == 'nt' else 4,
            'pin_memory': True,
            'train_val_split': 0.9,
            'save_frequency': 1,
            'early_stopping_patience': 5,
            'gradient_clip_val': 1.0,
            'bert_model': 'bert-base-multilingual-cased',
            'validation_frequency': 2,  # New: Validate every 2 epochs
            'max_validation_samples': 100,  # New: Limit validation samples
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
                
        # Validate directories and files
        if not config['image_dir'].exists():
            raise FileNotFoundError(f"Image directory not found: {config['image_dir']}")
        
        if not Path(config['captions_file']).exists():
            raise FileNotFoundError(f"Captions file not found: {config['captions_file']}")
        
        config['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
        
        # Initialize BERT tokenizer
        try:
            logging.info(f"Loading tokenizer from {config['bert_model']}")
            tokenizer = BertTokenizer.from_pretrained(
                'bert-base-multilingual-cased',
                do_lower_case=False,
                strip_accents=False,
                clean_text=False
            )
            
            special_tokens = {
                'pad_token': '[PAD]',
                'unk_token': '[UNK]',
                'sep_token': '[SEP]',
                'cls_token': '[CLS]',
                'mask_token': '[MASK]'
            }
            tokenizer.add_special_tokens(special_tokens)
            
            logging.info("Tokenizer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}")
        
        # Create datasets and dataloaders with memory optimization
        try:
            full_dataset = EnhancedDataset(
                image_dir=config['image_dir'],
                captions_file=config['captions_file'],
                tokenizer=tokenizer,
                max_length=config['max_length']
            )
            
            train_size = int(config['train_val_split'] * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=generator
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory']
            )
            
            logging.info(f"Created dataloaders with {train_size} training and {val_size} validation samples")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create datasets and dataloaders: {str(e)}")
        
        # Initialize model and training components with optimized settings
        try:
            device = torch.device(config['device'])
            model = GujaratiCaptioningModel(
                bert_model_name=config['bert_model']
            ).to(device)

            # Use gradient accumulation for memory efficiency
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            
            # Calculate total steps considering gradient accumulation
            total_steps = (len(train_loader) // config['gradient_accumulation_steps']) * config['num_epochs']
            warmup_steps = total_steps // 10
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['learning_rate'],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='linear'
            )
            
            logging.info("Model components initialized successfully")
            logging.info(f"Training on device: {device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model components: {str(e)}")
        
        # Save configuration
        try:
            config['timestamp'] = datetime.now().isoformat()
            config_path = config['checkpoint_dir'] / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logging.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {str(e)}")
        
        # Train model with optimized settings
        try:
            logging.info("Starting training with optimized settings")
            logging.info(f"Validation will be performed every {config['validation_frequency']} epochs")
            logging.info("Note: BLEU scores will be calculated separately after training using the evaluation script")
            
            train_history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=config['num_epochs'],
                device=device,
                tokenizer=tokenizer,
                checkpoint_dir=config['checkpoint_dir'],
                gradient_clip_val=config['gradient_clip_val'],
                early_stopping_patience=config['early_stopping_patience'],
                save_frequency=config['save_frequency'],
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                validation_frequency=config['validation_frequency']
            )
            
            # Save final model and training artifacts
            final_checkpoint = {
                'epoch': config['num_epochs'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'tokenizer_config': tokenizer.save_pretrained(
                    config['checkpoint_dir'] / 'tokenizer'
                )
            }
            
            final_model_path = config['checkpoint_dir'] / 'final_model.pth'
            torch.save(final_checkpoint, final_model_path)
            
            history_path = config['checkpoint_dir'] / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(train_history, f, indent=2, default=str)
                
            logging.info("Training completed successfully")
            logging.info(f"Saved final model to {final_model_path}")
            logging.info(f"Saved training history to {history_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error during training: {str(e)}")
        
    except Exception as e:
        logging.error(f"Fatal error in main function: {str(e)}")
        raise
    
    finally:
        # Cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Cleaned up resources")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            
if __name__ == '__main__':
    main()                