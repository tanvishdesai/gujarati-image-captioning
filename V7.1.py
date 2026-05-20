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
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
# Replace the imports at the top of the file with these:
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
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
from collections import Counter
import math

# Make sure NLTK data is downloaded
try:
    nltk.download('punkt')
    nltk.download('wordnet')
except Exception as e:
    logging.warning(f"Could not download NLTK data: {str(e)}")

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics calculator implementing BLEU1-4, METEOR, ROUGE, and CIDEr
    using direct implementations rather than the evaluate library.
    """
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.smooth = SmoothingFunction()
        
    def _compute_bleu(self, reference: str, hypothesis: str, n: int) -> float:
        """Compute BLEU-N score."""
        ref_tokens = nltk.word_tokenize(reference)
        hyp_tokens = nltk.word_tokenize(hypothesis)
        weights = tuple([1.0/n] * n)  # Equal weights for n-grams
        try:
            return sentence_bleu(
                [ref_tokens], 
                hyp_tokens, 
                weights=weights,
                smoothing_function=self.smooth.method1
            )
        except Exception as e:
            logging.warning(f"Error computing BLEU-{n}: {str(e)}")
            return 0.0

    def _compute_meteor(self, reference: str, hypothesis: str) -> float:
        """Compute METEOR score."""
        try:
            return meteor_score([reference], hypothesis)
        except Exception as e:
            logging.warning(f"Error computing METEOR: {str(e)}")
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
            logging.warning(f"Error computing ROUGE: {str(e)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def _compute_cider(self, references: List[str], hypothesis: str) -> float:
        """
        Compute CIDEr score using n-gram similarity and TF-IDF weighting.
        This is a simplified version of the CIDEr metric.
        """
        def get_ngrams(tokens, n):
            return Counter(zip(*[tokens[i:] for i in range(n)]))

        def compute_vec(refs, hyp, n):
            # Compute TF-IDF vectors for references and hypothesis
            ref_ngrams = [get_ngrams(nltk.word_tokenize(ref), n) for ref in refs]
            hyp_ngrams = get_ngrams(nltk.word_tokenize(hyp), n)
            
            # Compute document frequency
            doc_freq = Counter()
            for ref_count in ref_ngrams:
                doc_freq.update(ref_count.keys())
            
            # Compute IDF weights
            num_refs = len(refs)
            idf = {k: math.log(num_refs / (v + 1)) for k, v in doc_freq.items()}
            
            # Compute TF-IDF vectors
            ref_vecs = []
            for ref_count in ref_ngrams:
                vec = {k: v * idf.get(k, 0) for k, v in ref_count.items()}
                magnitude = math.sqrt(sum(x*x for x in vec.values()))
                if magnitude > 0:
                    vec = {k: v/magnitude for k, v in vec.items()}
                ref_vecs.append(vec)
            
            hyp_vec = {k: v * idf.get(k, 0) for k, v in hyp_ngrams.items()}
            magnitude = math.sqrt(sum(x*x for x in hyp_vec.values()))
            if magnitude > 0:
                hyp_vec = {k: v/magnitude for k, v in hyp_vec.items()}
            
            return ref_vecs, hyp_vec

        try:
            scores = []
            for n in range(1, 5):  # Use n-grams from 1 to 4
                ref_vecs, hyp_vec = compute_vec(references, hypothesis, n)
                if hyp_vec:
                    # Compute cosine similarity with each reference
                    sims = []
                    for ref_vec in ref_vecs:
                        common_ngrams = set(ref_vec.keys()) & set(hyp_vec.keys())
                        sim = sum(ref_vec[ng] * hyp_vec[ng] for ng in common_ngrams)
                        sims.append(sim)
                    scores.append(np.mean(sims))
                else:
                    scores.append(0.0)
            
            return np.mean(scores) * 10.0  # Scale to be comparable with other metrics
            
        except Exception as e:
            logging.warning(f"Error computing CIDEr: {str(e)}")
            return 0.0

    def calculate_metrics(self, references: List[str], hypotheses: List[str],
                         batch_size: int = 32) -> Dict[str, float]:
        """Calculate all metrics efficiently using batching."""
        metrics = {}
        
        # Calculate metrics for each pair
        for n in range(1, 5):  # BLEU1-4
            bleu_scores = []
            for ref, hyp in zip(references, hypotheses):
                bleu_scores.append(self._compute_bleu(ref, hyp, n))
            metrics[f'bleu{n}'] = np.mean(bleu_scores) * 100
        
        # Calculate METEOR
        meteor_scores = []
        for ref, hyp in zip(references, hypotheses):
            meteor_scores.append(self._compute_meteor(ref, hyp))
        metrics['meteor'] = np.mean(meteor_scores) * 100
        
        # Calculate ROUGE scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for ref, hyp in zip(references, hypotheses):
            scores = self._compute_rouge(ref, hyp)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key])
        
        for key in rouge_scores:
            metrics[key] = np.mean(rouge_scores[key]) * 100
        
        # Calculate CIDEr score
        metrics['cider'] = self._compute_cider(references, hypotheses[0])
        
        return metrics
    
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class GujaratiCaptioningModelV7(nn.Module):
    """
    Enhanced image captioning model using BLIP for visual features and mT5 for 
    multilingual caption generation. Specifically optimized for Gujarati.
    """
    def __init__(self, 
                 blip_model_name="Salesforce/blip-image-captioning-base",
                 mt5_model_name="google/mt5-base"):
        super().__init__()
        
        # Initialize BLIP for image encoding
        self.blip = BlipForConditionalGeneration.from_pretrained(blip_model_name)
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        
        # Freeze BLIP parameters for stability
        for param in self.blip.parameters():
            param.requires_grad = False
            
        # Initialize mT5 for multilingual caption generation
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_model_name)
        
        # Cross-attention for combining BLIP features with mT5
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.mt5.config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Feature processing layers
        self.feature_projection = nn.Sequential(
            nn.Linear(self.blip.config.hidden_size, self.mt5.config.hidden_size),
            nn.LayerNorm(self.mt5.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        # Get BLIP features
        with torch.no_grad():
            blip_features = self.blip.vision_model(
                pixel_values=images
            ).last_hidden_state
        
        # Project BLIP features to mT5 dimension
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
        
        # Generate captions using mT5 decoder
        outputs = self.mt5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=(enhanced_features,),
            labels=labels if labels is not None else None,
            return_dict=True
        )
        
        return outputs
    
    def generate_caption(self, image, tokenizer, max_length=64, num_beams=4, device='cuda'):
        """Generate a caption using beam search."""
        self.eval()
        with torch.no_grad():
            # Process image with BLIP
            blip_features = self.blip.vision_model(
                pixel_values=image.unsqueeze(0).to(device)
            ).last_hidden_state
            
            # Project features
            projected_features = self.feature_projection(blip_features)
            
            # Generate caption using mT5
            input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)
            
            outputs = self.mt5.generate(
                input_ids=input_ids,
                encoder_outputs=torch.nn.utils.rnn.pad_sequence([projected_features], 
                                                              batch_first=True),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

class GujaratiPreprocessor:
    """Gujarati text preprocessing with enhanced normalization."""
    def __init__(self):
        self.normalizer = IndicNormalizerFactory().get_normalizer("gu")
    
    def normalize_text(self, text):
        normalized = self.normalizer.normalize(text)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def tokenize(self, text):
        return indic_tokenize.trivial_tokenize(text, lang='gu')

class EnhancedDataset(Dataset):
    """Dataset class with improved data handling and preprocessing."""
    def __init__(self, image_dir, captions_file, processor, tokenizer, 
                 max_length=64, is_train=True):
        super().__init__()
        self.gujarati_processor = GujaratiPreprocessor()
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Load and preprocess captions
        self.samples = []
        skipped_images = 0
        
        with open(captions_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                try:
                    image_name, caption = line.strip().split('\t')
                    image_path = Path(image_dir) / image_name.split('#')[0]
                    
                    if image_path.exists():
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
            # Load and process image using BLIP processor
            image = Image.open(Path(self.image_dir) / image_name).convert('RGB')
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
            logging.error(f"Error loading item {idx}, image: {image_name}, Error: {str(e)}")
            return {
                'image': torch.zeros((3, 224, 224)),
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'caption': ""
            }


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
    evaluator: EvaluationMetrics,
    config: dict
):
    """
    Optimized training function with efficient evaluation scheduling.
    """
    best_cider = -float('inf')
    train_history = []
    patience_counter = 0
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_idx, batch in enumerate(train_pbar):
            try:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss / config['gradient_accumulation_steps']
                loss.backward()
                
                if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['gradient_clip_val']
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * config['gradient_accumulation_steps']
                train_pbar.set_postfix({
                    'loss': f'{loss.item() * config["gradient_accumulation_steps"]:.4f}'
                })
                
            except Exception as e:
                logging.error(f"Error in training batch: {str(e)}")
                continue
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation and metrics calculation (less frequent)
        if (epoch + 1) % config['validation_frequency'] == 0:
            model.eval()
            val_loss = 0.0
            references = []
            hypotheses = []
            
            with torch.no_grad():
                for batch in tqdm(itertools.islice(val_loader, config['max_validation_samples']),
                                desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                    try:
                        images = batch['image'].to(device)
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        captions = batch['caption']
                        
                        outputs = model(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        
                        val_loss += outputs.loss.item()
                        
                        # Generate caption for evaluation
                        hyp = model.generate_caption(
                            images[0],
                            tokenizer,
                            device=device
                        )
                        references.append(captions[0])
                        hypotheses.append(hyp)
                        
                    except Exception as e:
                        logging.error(f"Error in validation batch: {str(e)}")
                        continue
            
            # Calculate all metrics
            metrics = evaluator.calculate_metrics(references, hypotheses)
            avg_val_loss = val_loss / len(references)
            
            # Logging
            logging.info(f"\nEpoch {epoch + 1} Summary:")
            logging.info(f"Training Loss: {avg_train_loss:.4f}")
            logging.info(f"Validation Loss: {avg_val_loss:.4f}")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.2f}")
            
            # Save checkpoint if improvement in CIDEr score
            if metrics['cider'] > best_cider:
                best_cider = metrics['cider']
                patience_counter = 0
                
                checkpoint_path = checkpoint_dir / f'best_model_epoch_{epoch + 1}.pth'
                torch
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': metrics,
                    'best_cider': best_cider
                }
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= config['early_stopping_patience']:
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
        
        # Save regular checkpoint
        if (epoch + 1) % config['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss
            }, checkpoint_path)
    
    return train_history

def main():
    """
    Main function with comprehensive error handling and optimized training pipeline.
    """
    try:
        # Define training configuration
        config = {
            'batch_size': 16,  # Increased for BLIP+mT5
            'max_length': 64,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2e-5,  # Adjusted for transfer learning
            'num_epochs': 10,
            'image_dir': Path('flickr8k/Flickr_Data/Flickr_Data/Images'),
            'captions_file': 'gujarati_captions.txt',
            'checkpoint_dir': Path('checkpoints_v7'),
            'num_workers': 0 if os.name == 'nt' else 4,
            'pin_memory': True,
            'train_val_split': 0.9,
            'save_frequency': 1,
            'early_stopping_patience': 5,
            'gradient_clip_val': 1.0,
            'validation_frequency': 2,  # Validate every 2 epochs
            'max_validation_samples': 100,  # Limit validation samples for efficiency
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'blip_model': 'Salesforce/blip-image-captioning-base',
            'mt5_model': 'google/mt5-base'
        }
        
        # Validate directories and files
        if not config['image_dir'].exists():
            raise FileNotFoundError(f"Image directory not found: {config['image_dir']}")
        if not Path(config['captions_file']).exists():
            raise FileNotFoundError(f"Captions file not found: {config['captions_file']}")
        
        config['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
        
        # Initialize models and processors
        try:
            logging.info("Initializing BLIP processor and mT5 tokenizer")
            blip_processor = BlipProcessor.from_pretrained(config['blip_model'])
            mt5_tokenizer = MT5Tokenizer.from_pretrained(config['mt5_model'])
            
            logging.info("Model components initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model components: {str(e)}")
        
        # Create datasets and dataloaders
        try:
            full_dataset = EnhancedDataset(
                image_dir=config['image_dir'],
                captions_file=config['captions_file'],
                processor=blip_processor,
                tokenizer=mt5_tokenizer,
                max_length=config['max_length']
            )
            
            train_size = int(config['train_val_split'] * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory']
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
            device = torch.device(config['device'])
            model = GujaratiCaptioningModelV7(
                blip_model_name=config['blip_model'],
                mt5_model_name=config['mt5_model']
            ).to(device)
            
            # Initialize optimizer with weight decay
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            
            # Calculate training steps
            total_steps = (len(train_loader) // config['gradient_accumulation_steps']) * config['num_epochs']
            warmup_steps = total_steps // 10
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['learning_rate'],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='linear'
            )
            
            # Initialize evaluation metrics calculator
            evaluator = EvaluationMetrics()
            
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
        
        # Train model
        try:
            logging.info("Starting training with BLIP + mT5 architecture")
            logging.info(f"Evaluating metrics every {config['validation_frequency']} epochs")
            
            train_history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=config['num_epochs'],
                device=device,
                tokenizer=mt5_tokenizer,
                checkpoint_dir=config['checkpoint_dir'],
                evaluator=evaluator,
                config=config
            )
            
            # Save final model and training artifacts
            final_checkpoint = {
                'epoch': config['num_epochs'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleaned up resources")

if __name__ == '__main__':
    main()