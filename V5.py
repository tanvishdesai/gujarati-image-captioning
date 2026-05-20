import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import logging
import json
from datetime import datetime
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
from typing import Dict, List

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LightweightCaptioningModel(nn.Module):
    """
    Simplified image captioning model using ResNet18 and GRU decoder.
    More memory-efficient than the mBART version.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        # Use ResNet18 instead of MobileNetV2 for better efficiency
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection for image features - now projects to hidden_dim to match GRU
        self.image_projection = nn.Linear(512, hidden_dim)
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Additional projection for embedded words to match hidden_dim
        self.embed_projection = nn.Linear(embed_dim, hidden_dim)
        
        # GRU decoder (simpler than Transformer)
        self.decoder = nn.GRU(
            input_size=hidden_dim,  # Changed from embed_dim to hidden_dim
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, images, captions):
        batch_size = images.size(0)
        
        # Encode images
        image_features = self.image_encoder(images)
        image_features = image_features.squeeze(-1).squeeze(-1)
        image_features = self.dropout(self.image_projection(image_features))
        
        # Embed captions and project to hidden_dim
        embedded = self.dropout(self.embedding(captions))
        embedded = self.embed_projection(embedded)
        
        # Initialize hidden state with image features
        hidden = image_features.unsqueeze(0)  # Shape: [1, batch_size, hidden_dim]
        
        # Decode
        output, _ = self.decoder(embedded, hidden)
        output = self.output(output)
        
        return output

class SimpleDataset(Dataset):
    """
    Simplified dataset class focusing on essential functionality.
    """
    def __init__(self, image_dir, captions_file, vocab, max_length=30):
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.max_length = max_length
        
        # Basic image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load caption data
        self.samples = []
        
        logging.info("Loading dataset...")
        with open(captions_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Loading captions"):
                try:
                    image_name, caption = line.strip().split('\t')
                    image_path = self.image_dir / image_name.split('#')[0]
                    if image_path.exists():
                        self.samples.append((image_name.split('#')[0], caption))
                except:
                    continue
                    
        logging.info(f"Loaded {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]
        
        # Load and transform image
        image = Image.open(self.image_dir / image_name).convert('RGB')
        image = self.transform(image)
        
        # Convert caption to indices
        caption_indices = [self.vocab['<start>']]
        caption_indices.extend(self.vocab.get(token, self.vocab['<unk>']) 
                             for token in caption.split())
        caption_indices.append(self.vocab['<end>'])
        
        # Pad or truncate
        if len(caption_indices) < self.max_length:
            caption_indices.extend([self.vocab['<pad>']] * 
                                 (self.max_length - len(caption_indices)))
        else:
            caption_indices = caption_indices[:self.max_length]
            
        return {
            'image': image,
            'caption': torch.tensor(caption_indices, dtype=torch.long),
            'raw_caption': caption
        }

def create_vocabulary(captions_file, min_freq=2):
    """Create a simple vocabulary from the caption file."""
    word_freq = {}
    
    logging.info("Creating vocabulary...")
    with open(captions_file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Building vocabulary"):
            try:
                _, caption = line.strip().split('\t')
                for word in caption.split():
                    word_freq[word] = word_freq.get(word, 0) + 1
            except:
                continue
    
    # Create vocabulary
    vocab = {
        '<pad>': 0,
        '<start>': 1,
        '<end>': 2,
        '<unk>': 3
    }
    
    idx = 4
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    logging.info(f"Created vocabulary with {len(vocab)} tokens")        
    return vocab

def evaluate_model(model, val_loader, vocab, device):
    """Evaluate model on validation set."""
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            raw_captions = batch['raw_caption']
            
            # Generate captions
            batch_size = images.size(0)
            hidden = model.image_encoder(images)
            hidden = hidden.squeeze(-1).squeeze(-1)
            hidden = model.image_projection(hidden).unsqueeze(0)
            
            # Initialize with start token
            decoder_input = torch.tensor([[vocab['<start>']]] * batch_size).to(device)
            captions = [[] for _ in range(batch_size)]
            
            # Generate caption word by word
            for _ in range(30):  # max length
                embedded = model.embedding(decoder_input)
                embedded = model.embed_projection(embedded)
                
                output, hidden = model.decoder(embedded, hidden)
                output = model.output(output)
                
                # Get next word
                predicted = output.argmax(2)
                
                # Store predictions for each sample in batch
                for i in range(batch_size):
                    captions[i].append(predicted[i].item())
                
                # Break if all sequences in batch predict end token
                if all(pred.item() == vocab['<end>'] for pred in predicted[:, 0]):
                    break
                    
                decoder_input = predicted
            
            # Convert indices to words
            idx_to_word = {v: k for k, v in vocab.items()}
            for i in range(batch_size):
                pred_caption = []
                for idx in captions[i]:
                    word = idx_to_word[idx]
                    if word == '<end>':
                        break
                    if word not in ['<start>', '<pad>', '<unk>']:
                        pred_caption.append(word)
                
                hypotheses.append(' '.join(pred_caption))
                references.append(raw_captions[i])
    
    # Calculate metrics
    metrics = calculate_metrics(references, hypotheses)
    
    return metrics, hypotheses
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Run one training epoch with progress bar."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        captions = batch['caption'].to(device)
        
        outputs = model(images, captions[:, :-1])
        
        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            captions[:, 1:].reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
    return total_loss / len(train_loader)

def calculate_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate various NLP metrics for generated captions in Gujarati.
    Properly handles tokenization and scoring for Gujarati text.
    """
    try:
        rouge_scorer_inst = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                   use_stemmer=False)
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
                # Proper tokenization for Gujarati text
                ref_tokens = ref.strip().split()
                hyp_tokens = hyp.strip().split()
                
                if len(hyp_tokens) == 0:  # Skip empty hypotheses
                    continue
                
                # Calculate BLEU scores with single reference
                bleu1 = sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    weights=(1.0,),
                    smoothing_function=smooth.method1
                )
                
                bleu4 = sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth.method1
                )
                
                # Calculate METEOR score with pre-tokenized inputs
                meteor = meteor_score(
                    [ref_tokens],  # Pass tokenized reference
                    hyp_tokens,    # Pass tokenized hypothesis
                    preprocess=str  # Ensure proper string handling
                )
                
                # Calculate ROUGE scores
                rouge_scores = rouge_scorer_inst.score(ref, hyp)
                
                # Accumulate scores
                scores['bleu-1'] += bleu1
                scores['bleu-4'] += bleu4
                scores['meteor'] += meteor
                scores['rouge1'] += rouge_scores['rouge1'].fmeasure
                scores['rouge2'] += rouge_scores['rouge2'].fmeasure
                scores['rougeL'] += rouge_scores['rougeL'].fmeasure
                
                valid_pairs += 1
                
            except Exception as e:
                logging.warning(
                    f"Error calculating metrics for pair:\n"
                    f"Reference: {ref}\n"
                    f"Hypothesis: {hyp}\n"
                    f"Error: {str(e)}\n"
                    f"Tokens - Ref: {ref_tokens}, Hyp: {hyp_tokens}"
                )
                continue
        
        # Average scores
        if valid_pairs > 0:
            for key in scores:
                scores[key] = scores[key] / valid_pairs
                # Convert to percentage for easier interpretation
                scores[key] = scores[key] * 100
        
        return scores
    
    except Exception as e:
        logging.error(f"Error in calculate_metrics: {str(e)}")
        return {k: 0.0 for k in ['bleu-1', 'bleu-4', 'meteor', 'rouge1', 'rouge2', 'rougeL']}

def main():
    """Main training function."""
    try:
        # Configuration
        config = {
            'image_dir': 'flickr8k/Flickr_Data/Flickr_Data/Images',
            'captions_file': 'gujarati_captions.txt',
            'batch_size': 32,
            'embed_dim': 256,
            'hidden_dim': 512,
            'num_epochs': 10,
            'learning_rate': 1e-3,
            'train_val_split': 0.9  # Added train-val split ratio
        }
        
        # Create vocabulary
        vocab = create_vocabulary(config['captions_file'])
        
        # Create dataset
        full_dataset = SimpleDataset(
            config['image_dir'],
            config['captions_file'],
            vocab
        )
        
        # Split dataset
        train_size = int(config['train_val_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model and training components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        model = LightweightCaptioningModel(
            vocab_size=len(vocab),
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Training loop with evaluation
        best_bleu4 = 0
        for epoch in range(config['num_epochs']):
            # Training
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, 
                device, epoch, config['num_epochs']
            )
            
            # Evaluation
            metrics, _ = evaluate_model(model, val_loader, vocab, device)
            
            # Log progress
            logging.info(f'\nEpoch {epoch + 1} Summary:')
            logging.info(f'Training Loss: {train_loss:.4f}')
            logging.info('Validation Metrics:')
            for metric, value in metrics.items():
                logging.info(f'{metric}: {value:.4f}')
            
            # Save best model
            if metrics['bleu-4'] > best_bleu4:
                best_bleu4 = metrics['bleu-4']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab': vocab,
                    'config': config,
                    'metrics': metrics
                }, 'best_gujarati_captioner.pth')
                logging.info(f'Saved new best model with BLEU-4: {best_bleu4:.4f}')
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise
if __name__ == '__main__':
    main()