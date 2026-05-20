import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import MBartTokenizer, MBartForConditionalGeneration
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
    Processes images into a format suitable for integration with mBART.
    """
    def __init__(self, encoded_dim=1024):
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

class MultilingualCaptioningModel(nn.Module):
    """
    Enhanced image captioning model using mBART for multilingual caption generation.
    Combines visual features with language understanding through cross-attention.
    """
    def __init__(self, 
                 mbart_model_name="facebook/mbart-large-cc25",
                 freeze_encoder_layers=6):
        super().__init__()
        
        # Initialize mBART model and image encoder
        self.mbart = MBartForConditionalGeneration.from_pretrained(mbart_model_name)
        self.image_encoder = LightweightImageEncoder(encoded_dim=self.mbart.config.d_model)
        
        # Freeze initial layers of mBART encoder for stability
        for i, param in enumerate(self.mbart.model.encoder.layers[:freeze_encoder_layers].parameters()):
            param.requires_grad = False
            
        # Cross-attention mechanism for combining image and text features
        self.image_cross_attention = nn.MultiheadAttention(
            embed_dim=self.mbart.config.d_model,
            num_heads=self.mbart.config.decoder_attention_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Additional layers for feature processing
        self.image_projection = nn.Sequential(
            nn.Linear(self.mbart.config.d_model, self.mbart.config.d_model),
            nn.LayerNorm(self.mbart.config.d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Layer normalization for feature combination
        self.feature_norm = nn.LayerNorm(self.mbart.config.d_model)
        
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        # Encode and project image features
        image_features = self.image_encoder(images)
        image_features = self.image_projection(image_features).unsqueeze(1)
        
        # Get mBART encoder outputs
        encoder_outputs = self.mbart.get_encoder()(
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
        
        # Update encoder outputs with combined features
        encoder_outputs.last_hidden_state = enhanced_features
        
        # Generate output through mBART decoder
        outputs = self.mbart(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate_caption(self, image, tokenizer, max_length=64, num_beams=4, device='cuda'):
        """Generate a caption for a given image using beam search."""
        self.eval()
        with torch.no_grad():
            # Encode and project image features
            image_features = self.image_encoder(image.unsqueeze(0).to(device))
            image_features = self.image_projection(image_features).unsqueeze(1)
            
            # Initialize with Gujarati language token
            decoder_input_ids = torch.tensor([[tokenizer.lang_code_to_id["gu_IN"]]]).to(device)
            
            # Generate caption using beam search
            outputs = self.mbart.generate(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=torch.nn.utils.rnn.pad_sequence(
                    [image_features.squeeze(1)], batch_first=True
                ).unsqueeze(0),
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,  # Reduced for Gujarati morphology
                length_penalty=0.7,  # Adjusted for Gujarati
                early_stopping=True,
                temperature=0.8,  # Increased for Gujarati diversity
                top_k=50,
                top_p=0.9,
                do_sample=True,  # Enable sampling for more natural Gujarati
                repetition_penalty=1.2  # Help prevent repetitive Gujarati phrases
            )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
    Supports multilingual captions and includes robust error handling.
    """
    def __init__(self, image_dir, captions_file, tokenizer, max_length=64, is_train=True):
        super().__init__()
        self.gujarati_processor = GujaratiPreprocessor()
        
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
                        image_path = self.image_dir / image_name.split('#')[0]
                        
                        if image_path.exists():
                            # Normalize and preprocess Gujarati caption
                            normalized_caption = self.gujarati_processor.normalize_text(caption)
                            processed_caption = f"<gu_IN> {normalized_caption}"
                            self.samples.append((image_name.split('#')[0], processed_caption))
                            
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
            image = Image.open(self.image_dir / image_name).convert('RGB')
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
            return self._get_empty_item()

def calculate_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate various NLP metrics for generated captions.
    Adapted for multilingual evaluation with character-based tokenization.
    """
    try:
        rouge_scorer_inst = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                   use_stemmer=True)
        smooth = SmoothingFunction()
        
        scores = {
            'bleu-1': 0.0,
            'bleu-4': 0.0,
            'meteor': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }
        
        valid_pairs = 0
        for ref, hyp in zip(references, hypotheses):
            try:
                # Character-based tokenization for Gujarati
                ref_tokens = list(ref.replace(" ", ""))
                hyp_tokens = list(hyp.replace(" ", ""))
                
                # Calculate BLEU scores
                scores['bleu-1'] += sentence_bleu([ref_tokens], hyp_tokens,
                                                weights=(1.0,),
                                                smoothing_function=smooth.method1)
                
                scores['bleu-4'] += sentence_bleu([ref_tokens], hyp_tokens,
                                                weights=(0.25, 0.25, 0.25, 0.25),
                                                smoothing_function=smooth.method1)
                
                # Calculate METEOR score
                scores['meteor'] += meteor_score([ref], hyp)
                
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
        return {k: 0.0 for k in ['bleu-1', 'bleu-4', 'meteor', 'rouge1', 'rouge2', 'rougeL']}

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    tokenizer,
    checkpoint_dir: Path,
    gradient_clip_val: float = 1.0,
    early_stopping_patience: int = 5,
    save_frequency: int = 1,
    gradient_accumulation_steps: int = 8  # Added parameter
):
    """
    Memory-optimized training function with gradient accumulation.
    """
    best_loss = float('inf')
    train_history = []
    patience_counter = 0
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
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
                
                # Calculate loss and scale it by gradient accumulation steps
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                # Update weights only after accumulating enough gradients
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
                
                # Update progress bar
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
        
        # Validation loop
# Validation loop
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            references = []
            hypotheses = []
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
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
                        
                        loss = outputs.loss
                        val_loss += loss.item()
                        
                        # Generate captions for metric calculation
                        for img, ref in zip(images, captions):
                            try:
                                hyp = model.generate_caption(
                                    img, 
                                    tokenizer,
                                    device=device
                                )
                                references.append(ref)
                                hypotheses.append(hyp)
                            except Exception as e:
                                logging.warning(f"Error generating caption: {str(e)}")
                                continue
                            
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                            
                    except Exception as e:
                        logging.error(f"Error in validation batch: {str(e)}")
                        continue
            
            # Calculate average validation loss and metrics
            avg_val_loss = val_loss / len(val_loader)
            metrics = calculate_metrics(references, hypotheses)
            
            # Log training progress
            logging.info(f"\nEpoch {epoch + 1} Summary:")
            logging.info(f"Training Loss: {avg_train_loss:.4f}")
            logging.info(f"Validation Loss: {avg_val_loss:.4f}")
            logging.info("Metrics:")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}")
            
            # Check for best model and save checkpoint
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
                        'metrics': metrics
                    }, checkpoint_path)
                    logging.info(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save training history
            train_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
    
    return train_history

def main():
    """
    Main function to set up and run the training process.
    Includes configuration, initialization, and error handling.
    """
    try:
        # Define training configuration
        config = {
            'batch_size': 4,
            'hidden_dim': 768,  # Matches mBART hidden dimension
            'max_length': 48,
            'gradient_accumulation_steps': 8,  # New parameter

            'learning_rate': 1e-4,
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
            'mbart_model': 'facebook/mbart-large-cc25'
        }
        
        # Validate directories and files
        if not config['image_dir'].exists():
            raise FileNotFoundError(f"Image directory not found: {config['image_dir']}")
        
        if not Path(config['captions_file']).exists():
            raise FileNotFoundError(f"Captions file not found: {config['captions_file']}")
        
        config['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
        
        # Initialize mBART tokenizer
        try:
            tokenizer = MBartTokenizer.from_pretrained(config['mbart_model'])
            logging.info("Tokenizer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}")
        
        # Create datasets and dataloaders
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
        
        # Initialize model and training components
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = MultilingualCaptioningModel(
                mbart_model_name=config['mbart_model']
            ).to(device)
            model.mbart.gradient_checkpointing_enable()
            model.mbart.config.use_cache = False

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            
            # Learning rate scheduler with warmup
            total_steps = len(train_loader) * config['num_epochs']
            warmup_steps = total_steps // 10
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['learning_rate'],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='linear'
            )
            
            logging.info("Model components initialized successfully")
            
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
        
        # Train model
        try:
            train_history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id),
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=config['num_epochs'],
                device=device,
                tokenizer=tokenizer,
                checkpoint_dir=config['checkpoint_dir'],
                gradient_clip_val=config['gradient_clip_val'],
                early_stopping_patience=config['early_stopping_patience'],
                save_frequency=config['save_frequency']
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