import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torchvision.models as models
from PIL import Image
import os
from tqdm.auto import tqdm
import numpy as np
import gc
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import json
import logging
from transformers import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json
import random
from typing import Dict, List

def process_buffer(results_buffer: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Process a buffer of generated and reference captions to calculate evaluation metrics.
    
    Args:
        results_buffer (dict): Dictionary containing lists of generated and reference captions
            - 'generated': List of generated captions
            - 'reference': List of reference captions
    
    Returns:
        dict: Dictionary containing evaluation metrics (BLEU-1,2,3,4 and ROUGE-L)
    """
    if not results_buffer['generated']:
        return {}
    
    # Prepare references for BLEU scoring
    references = [[ref.split()] for ref in results_buffer['reference']]
    hypotheses = [gen.split() for gen in results_buffer['generated']]
    
    # Calculate BLEU scores with smoothing
    smoother = SmoothingFunction().method7
    bleu_scores = {}
    for n in range(1, 5):
        bleu_scores[f'bleu-{n}'] = corpus_bleu(
            references,
            hypotheses,
            weights=tuple([1.0/n] * n),
            smoothing_function=smoother
        )
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for gen, ref in zip(results_buffer['generated'], results_buffer['reference']):
        scores = scorer.score(ref, gen)
        rouge_scores.append(scores['rougeL'].fmeasure)
    
    # Combine metrics
    metrics = {
        'bleu-1': bleu_scores['bleu-1'],
        'bleu-2': bleu_scores['bleu-2'],
        'bleu-3': bleu_scores['bleu-3'],
        'bleu-4': bleu_scores['bleu-4'],
        'rouge-l': sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    }
    
    # Save intermediate results
    try:
        with open('evaluation_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save results to file: {str(e)}")
    
    return metrics
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    valid_items = [item for item in batch if item is not None]
    if not valid_items:
        # print(f"Batch contained {len(batch)} items, all were None")
        return None
    # print(f"Batch contained {len(batch)} items, {len(valid_items)} were valid")
    return {
        'image': torch.stack([item['image'] for item in valid_items]),
        'caption': torch.stack([item['caption'] for item in valid_items]),
        'raw_caption': [item['raw_caption'] for item in valid_items]
    }


# [Previous Attention and ImageCaptioningModel classes remain the same]


def train_model(model, train_loader, val_loader, criterion, num_epochs, device, 
                accumulation_steps=4, mixed_precision=True, checkpoint_dir='checkpoints'):
    """
    Comprehensive training function with advanced features for stable training of image captioning models.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        accumulation_steps: Number of steps to accumulate gradients
        mixed_precision: Whether to use mixed precision training
        checkpoint_dir: Directory to save model checkpoints
    
    Returns:
        best_val_loss: Best validation loss achieved during training
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize optimizer with different learning rates for different components
    optimizer = optim.AdamW([
        {'params': model.image_encoder.parameters(), 'lr': 1e-4},  # Lower LR for pretrained encoder
        {'params': model.enc_projection.parameters(), 'lr': 3e-4},
        {'params': model.embed.parameters(), 'lr': 3e-4},
        {'params': model.attention.parameters(), 'lr': 3e-4},
        {'params': model.decode_step.parameters(), 'lr': 3e-4},
        {'params': model.fc.parameters(), 'lr': 3e-4}
    ], weight_decay=0.001)  # Weight decay for regularization
    
    # Calculate training steps and warmup period
    total_steps = len(train_loader) * num_epochs
    warmup_steps = total_steps // 10  # 10% of total steps for warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay with minimum learning rate of 10% of initial value
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize mixed precision scaler if enabled
    scaler = GradScaler() if mixed_precision else None
    best_val_loss = float('inf')
        
    print(f"Starting training for {num_epochs} epochs")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"Using mixed precision: {mixed_precision}")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            batch_count = 0
            
            # Progress bar for training
            train_pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')
            optimizer.zero_grad()  # Zero gradients at start of epoch
            
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    # Skip invalid batches
                    if batch is None:
                        print(f"Skipping None batch {batch_idx}")
                        continue
                    
                    # Move data to device
                    images = batch['image'].to(device, non_blocking=True)
                    captions = batch['caption'].to(device, non_blocking=True)
                    
                    # Forward pass with mixed precision
                    with autocast(enabled=mixed_precision):
                        outputs = model(images, captions)
                        targets = captions[:, 1:]  # Remove start token
                        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                        loss = loss / accumulation_steps  # Normalize loss for gradient accumulation
                    
                    # Check for NaN loss early
                    if torch.isnan(loss):
                        current_lr = [group['lr'] for group in optimizer.param_groups]
                        print(f"NaN loss detected at batch {batch_idx}")
                        print(f"Current learning rates: {current_lr}")
                        raise RuntimeError("NaN loss detected")
                    
                    # Backward pass with mixed precision handling
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()
                    else:
                        loss.backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                    
                    # Update training statistics
                    total_train_loss += loss.item() * accumulation_steps
                    batch_count += 1
                    
                    # Update progress bar with current metrics
                    if batch_count % 5 == 0:
                        train_pbar.set_postfix({
                            'loss': f'{total_train_loss/batch_count:.4f}',
                            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                        })
                    
                    # Periodic memory cleanup
                    if batch_idx % 100 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"OOM in batch {batch_idx}. Skipping batch.")
                        if torch.cuda.is_available():
                            print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                        continue
                    raise e
                
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {str(e)}")
                    continue
            
            # Calculate average training loss for epoch
            avg_train_loss = total_train_loss / batch_count
            print(f"Epoch {epoch+1} training completed. Average loss: {avg_train_loss:.4f}")
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            val_batch_count = 0
            
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc='Validation')
            
            with torch.no_grad():
                for batch in val_pbar:
                    try:
                        if batch is None:
                            continue
                        
                        images = batch['image'].to(device, non_blocking=True)
                        captions = batch['caption'].to(device, non_blocking=True)
                        
                        with autocast(enabled=mixed_precision):
                            outputs = model(images, captions)
                            targets = captions[:, 1:]  # Remove start token
                            loss = criterion(
                                outputs.reshape(-1, outputs.size(-1)),
                                targets.reshape(-1)
                            )
                        
                        total_val_loss += loss.item()
                        val_batch_count += 1
                        
                        val_pbar.set_postfix({'val_loss': f'{total_val_loss/val_batch_count:.4f}'})
                    
                    except Exception as e:
                        print(f"Error in validation batch: {str(e)}")
                        continue
            
            # Calculate average validation loss
            avg_val_loss = total_val_loss / val_batch_count
            print(f"Validation completed. Average validation loss: {avg_val_loss:.4f}")
            
            # Save checkpoints
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }
            
            # Save regular checkpoint
            # torch.save(
            #     checkpoint,
            #     os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            # )
            
            # Save best model if validation loss improved
            if epoch == 11:
                best_val_loss = avg_val_loss
                checkpoint['best_val_loss'] = best_val_loss
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, 'best_model.pth')
                )
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            # Memory cleanup after each epoch
            torch.cuda.empty_cache()
            gc.collect()
    
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss if 'avg_val_loss' in locals() else None,
                'train_loss': avg_train_loss if 'avg_train_loss' in locals() else None,
            },
            os.path.join(checkpoint_dir, 'interrupted_checkpoint.pth')
        )
        raise
    
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_val_loss

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # Ensure consistent dtype
        encoder_out = encoder_out.to(dtype=decoder_hidden.dtype)
        
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, attention_dim=512):
        super(ImageCaptioningModel, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_dim = 512  # VGG16 output channels
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.flatten_dim = 49  # 7x7 spatial dimensions from VGG16

        # Initialize VGG16 but remove final MaxPool2d
        vgg16 = models.vgg16(pretrained=True)
        modules = list(vgg16.features.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        
        self.enc_projection = nn.Linear(self.encoder_dim, attention_dim)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(attention_dim, hidden_size, attention_dim)
        self.f_beta = nn.Linear(hidden_size, attention_dim)
        self.decode_step = nn.LSTMCell(embed_size + attention_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()
    def encode_images(self, images):
        """Separate method for encoding images to ensure consistent processing"""
        encoder_out = self.image_encoder(images)
        batch_size = encoder_out.size(0)
        
        # Reshape to [batch_size, 49, 512]
        encoder_out = encoder_out.permute(0, 2, 3, 1)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        
        # Project features
        encoder_out = self.enc_projection(encoder_out)
        return encoder_out
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, images, captions):
        batch_size = images.size(0)
        
        # Get encoded images - VGG16 outputs [batch_size, 512, 7, 7]
        encoder_out = self.image_encoder(images)
        
        # Reshape: [batch_size, 512, 7, 7] -> [batch_size, 49, 512]
        encoder_out = encoder_out.permute(0, 2, 3, 1)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        
        # Project features
        encoder_out = self.enc_projection(encoder_out)
        
        # Initialize LSTM state
        h = torch.zeros(batch_size, self.hidden_size).to(images.device)
        c = torch.zeros(batch_size, self.hidden_size).to(images.device)
        
        captions_in = captions[:, :-1]
        max_length = captions_in.size(1)
        predictions = []
        
        for t in range(max_length):
            attention_weighted_encoding, _ = self.attention(encoder_out, h)
            gate = torch.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            embeddings = self.embed(captions_in[:, t])
            lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            
            h, c = self.decode_step(lstm_input, (h, c))
            preds = self.fc(self.dropout(h))
            predictions.append(preds.unsqueeze(1))
        
        predictions = torch.cat(predictions, dim=1)
        return predictions

    def generate_caption(self, encoder_out, word2idx, idx2word, max_length=20):
        """Generate caption with explicit dtype handling"""
        batch_size = encoder_out.size(0)
        dtype = encoder_out.dtype
        device = encoder_out.device
        
        # Initialize LSTM states with matching dtype
        h = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)
        c = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)
        
        word = torch.tensor([word2idx['<start>']], dtype=torch.long, device=device)
        caption = []
        
        for _ in range(max_length):
            embeddings = self.embed(word).to(dtype=dtype)
            attention_weighted_encoding, _ = self.attention(encoder_out, h)
            
            gate = torch.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            
            # Ensure hidden states maintain dtype
            h, c = h.to(dtype=dtype), c.to(dtype=dtype)
            
            output = self.fc(h)
            predicted = output.argmax(1)
            
            if predicted.item() == word2idx['<end>']:
                break
                
            caption.append(idx2word[predicted.item()])
            word = predicted
        
        return caption

    def encode_images(self, images):
        """Encode images with explicit dtype handling"""
        dtype = images.dtype
        encoder_out = self.image_encoder(images)
        batch_size = encoder_out.size(0)
        
        # Reshape and maintain dtype
        encoder_out = encoder_out.permute(0, 2, 3, 1)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        encoder_out = self.enc_projection(encoder_out)
        
        return encoder_out.to(dtype=dtype)
def evaluate_model(model, test_loader, dataset, device, output_file='evaluation_results.json'):
    """Evaluation function with comprehensive dtype handling and proper layer type checking"""
    model.eval()
    results_buffer = {'generated': [], 'reference': []}
    buffer_size = 50

    def convert_model_to_half():
        """Helper function to ensure all model parameters are float16 with proper layer handling"""
        for param in model.parameters():
            param.data = param.data.half()
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.half()
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.half()
            elif isinstance(module, nn.Embedding):
                module.weight.data = module.weight.data.half()
            elif isinstance(module, nn.LSTMCell):
                # Handle LSTM parameters
                for name, param in module.named_parameters():
                    param.data = param.data.half()

    print("Starting evaluation...")
    try:
        with torch.no_grad():
            # Convert model to half precision
            model = model.half()
            convert_model_to_half()
            
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating captions")):
                if batch is None:
                    continue

                try:
                    # Convert images to half precision
                    images = batch['image'].to(device).half()
                    
                    # Process each image in the batch
                    for i in range(len(images)):
                        image_input = images[i].unsqueeze(0)
                        
                        try:
                            # Encode image with error handling
                            encoder_out = model.encode_images(image_input)
                            if encoder_out.dtype != torch.float16:
                                encoder_out = encoder_out.half()
                            
                            # Generate caption
                            generated_tokens = model.generate_caption(
                                encoder_out,
                                dataset.word2idx,
                                dataset.idx2word
                            )
                            
                            # Process caption
                            generated_caption = ' '.join([word for word in generated_tokens 
                                                        if word not in ['<start>', '<end>', '<pad>', '<unk>']])
                            reference_caption = batch['raw_caption'][i]
                            
                            results_buffer['generated'].append(generated_caption)
                            results_buffer['reference'].append(reference_caption)
                            
                        except RuntimeError as e:
                            print(f"Error processing image {i} in batch {batch_idx}: {str(e)}")
                            if 'encoder_out' in locals():
                                print(f"Encoder output shape: {encoder_out.shape}")
                                print(f"Encoder output dtype: {encoder_out.dtype}")
                            continue
                        
                        # Process results when buffer is full
                        if len(results_buffer['generated']) >= buffer_size:
                            metrics = process_buffer(results_buffer)
                            print(f"\nIntermediate metrics for batch {batch_idx}:")
                            for metric, value in metrics.items():
                                print(f"{metric}: {value:.4f}")
                            results_buffer = {'generated': [], 'reference': []}
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"Out of memory error. Current GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                        continue
                    else:
                        print(f"RuntimeError in batch {batch_idx}: {str(e)}")
                        torch.cuda.empty_cache()
                        continue
                
                # Clear GPU memory after each batch
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if results_buffer['generated']:
            print("Processing remaining results...")
            return process_buffer(results_buffer)
        return {}

    # Process any remaining results
    final_metrics = process_buffer(results_buffer) if results_buffer['generated'] else {}
    
    # Save final results
    try:
        with open(output_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save results to file: {str(e)}")
    
    return final_metrics

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, max_length=64):
        """Initialize the Flickr8k dataset with image-caption pairs.
        
        Args:
            image_dir (str): Directory containing the images
            captions_file (str): Path to the captions file
            transform (callable, optional): Transform to be applied to images
            max_length (int, optional): Maximum length for captions
        """
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        
        # Initialize vocabulary
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.word_freq = {}
        
        # Modified data structure to store multiple captions
        self.image_captions = {}  # {image_name: [caption1, caption2, ...]}
        missing_images = 0
        corrupted_images = 0
        
        # Build vocabulary first
        print("Building vocabulary...")
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    caption = parts[1]
                    words = caption.split()
                    for word in words:
                        self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # Add words appearing more than threshold times
        threshold = 5
        vocab_idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= threshold:
                self.word2idx[word] = vocab_idx
                self.idx2word[vocab_idx] = word
                vocab_idx += 1
        
        print("Loading dataset...")
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    caption = parts[1]
                    
                    image_path = os.path.join(image_dir, img_name)
                    if os.path.exists(image_path):
                        try:
                            with Image.open(image_path) as img:
                                img.verify()
                                if img_name not in self.image_captions:
                                    self.image_captions[img_name] = []
                                self.image_captions[img_name].append(caption)
                        except Exception as e:
                            corrupted_images += 1
                            continue
                    else:
                        missing_images += 1
        
        # Convert to list of (image_name, captions) tuples for indexing
        self.data = [(img_name, captions) for img_name, captions in self.image_captions.items()]
        
        print(f"Dataset initialized with {len(self.data)} valid images")
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Average captions per image: {sum(len(caps) for _, caps in self.data)/len(self.data):.1f}")
        print(f"Skipped {missing_images} missing images")
        print(f"Skipped {corrupted_images} corrupted images")
    
    def __len__(self):
        return len(self.data)

    def tokenize_caption(self, caption):
        """Tokenize a caption using the dataset's vocabulary."""
        words = caption.split()
        tokens = []
        tokens.append(self.word2idx['<start>'])
        tokens.extend([self.word2idx.get(word, self.word2idx['<unk>']) for word in words])
        tokens.append(self.word2idx['<end>'])
        
        if len(tokens) < self.max_length:
            tokens.extend([self.word2idx['<pad>']] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length-1] + [self.word2idx['<end>']]
            
        return torch.LongTensor(tokens)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        img_name, captions = self.data[idx]
        image_path = os.path.join(self.image_dir, img_name)
        
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                if self.transform:
                    image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading verified image {image_path}: {str(e)}")
        
        # Randomly select one caption and tokenize it
        chosen_caption = random.choice(captions)
        tokenized_caption = self.tokenize_caption(chosen_caption)
        
        return {
            'image': image,
            'caption': tokenized_caption,
            'raw_caption': chosen_caption,
            'all_captions': captions  # Store all reference captions
        }

def main():
    try:
        # Set environment variables for better GPU memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        torch.backends.cudnn.benchmark = True
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset with error handling
        try:
            dataset = Flickr8kDataset(
                image_dir='/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/',
                captions_file='/kaggle/input/captionsss/gujarati_captions.txt',
                transform=transform
            )
        except Exception as e:
            print(f"Error creating dataset: {str(e)}")
            raise
        
        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # Initialize model with vocabulary size from dataset
        model = ImageCaptioningModel(
            vocab_size=len(dataset.word2idx),
            embed_size=256,
            hidden_size=512,
            attention_dim=512
        ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx['<pad>'])
        
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            num_epochs=1,
            device=device,
            accumulation_steps=4,
            mixed_precision=True
        )
        
        # Evaluate model
        print("Starting final evaluation...")
        evaluation_results = evaluate_model(
            model=model,
            test_loader=test_loader,
            dataset = dataset,
            device=device,
            output_file='final_evaluation_results.json'
        )
        
        print("\nFinal Evaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()