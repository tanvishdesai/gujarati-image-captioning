import torch
import os
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
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

# Configure basic logging
logging.basicConfig(
    level=print,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LightweightCaptioningModel(nn.Module):
    """
    Enhanced version with:
    - Spatial attention mechanism
    - Fine-tuned ResNet encoder
    - Stacked GRU with dropout
    - Layer normalization
    - Beam search support
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Image Encoder ---
        resnet = models.resnet18(pretrained=True)
        
        # Freeze early layers (first 4 blocks)
        for param in list(resnet.children())[:5]:
            param.requires_grad = False
            
        # Train last 3 blocks
        for param in list(resnet.children())[5:-2]:
            param.requires_grad = True
            
        # Remove avgpool and fc layers
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Image feature projection
        self.image_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # --- Attention Mechanism ---
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # --- Decoder ---
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(0.3)
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Stacked GRU with attention input
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,  # [embedded + context]
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize gates
        self._init_gru_weights()

    def _init_gru_weights(self):
        """Orthogonal initialization for GRU weights"""
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, images, captions):
        # --- Image Encoding ---
        batch_size = images.size(0)
        
        # Extract spatial features [B, 512, H, W]
        img_features = self.image_encoder(images)
        img_features = self.adaptive_pool(img_features)
        img_features = self.image_proj(img_features)  # [B, hidden_dim, 7, 7]
        
        # Reshape for attention: [B, 7*7, hidden_dim]
        spatial_features = img_features.view(batch_size, self.hidden_dim, -1).permute(0, 2, 1)
        
        # --- Text Decoding ---
        embeddings = self.embed_dropout(self.embedding(captions))  # [B, seq_len, embed_dim]
        embeddings = self.embed_proj(embeddings)  # [B, seq_len, hidden_dim]
        
        # Initial hidden state from image features
        hidden = self.init_hidden(batch_size, images.device)
        
        outputs = []
        # Process sequence length matching the target length we'll use in the loss function
        seq_length = captions.size(1) 
        
        for t in range(seq_length):
            # Compute attention context
            context, _ = self._apply_attention(hidden, spatial_features)
            
            # Combine embedded input with context
            gru_input = torch.cat([
                embeddings[:, t, :].unsqueeze(1),  # [B, 1, hidden_dim]
                context.unsqueeze(1)               # [B, 1, hidden_dim]
            ], dim=2)
            
            # GRU step
            out, hidden = self.gru(gru_input, hidden)
            out = self.layer_norm(out)
            outputs.append(out)
            
        outputs = torch.cat(outputs, dim=1)
        return self.output(outputs)
        
    def _apply_attention(self, hidden, spatial_features):
        """Compute attention weights and context vector"""
        # hidden: [2, B, hidden_dim] (stacked GRU)
        # spatial_features: [B, 49, hidden_dim]
        
        # Use last layer's hidden state
        h = hidden[-1].unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Expand to match spatial features
        h_expanded = h.expand(-1, spatial_features.size(1), -1)  # [B, 49, hidden_dim]
        
        # Concatenate features
        combined = torch.cat([h_expanded, spatial_features], dim=2)  # [B, 49, 2*hidden]
        
        # Compute attention scores
        attn_weights = F.softmax(self.attention(combined), dim=1)  # [B, 49, 1]
        
        # Compute context vector
        context = torch.sum(attn_weights * spatial_features, dim=1)  # [B, hidden_dim]
        
        return context, attn_weights

    def init_hidden(self, batch_size, device):
        """Initialize hidden state from image features"""
        return torch.zeros(2, batch_size, self.hidden_dim, device=device)

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
        
        print("Loading dataset...")
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
                    
        print(f"Loaded {len(self.samples)} samples")
        
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
            'raw_caption': caption,
            'image_name': image_name  # Added image_name to help with debugging and evaluation
        }

class UniqueImageDataset(Dataset):
    """
    Dataset class for evaluation, providing unique images with all their reference captions.
    """
    def __init__(self, full_dataset, image_to_captions):
        self.full_dataset = full_dataset
        self.image_to_captions = image_to_captions
        self.image_names = list(image_to_captions.keys())
        # Find one index for each image_name to load the image
        self.image_indices = {}
        for idx, (image_name, _) in enumerate(full_dataset.samples):
            if image_name not in self.image_indices:
                self.image_indices[image_name] = idx
                
        # Print statistics
        print(f"UniqueImageDataset created with {len(self.image_names)} unique images")
        avg_captions = sum(len(caps) for caps in image_to_captions.values()) / len(image_to_captions) if image_to_captions else 0
        print(f"Average captions per image: {avg_captions:.2f}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        try:
            image_name = self.image_names[idx]
            # Load image using the full_dataset's method
            image_idx = self.image_indices.get(image_name)
            
            if image_idx is None:
                # Fallback if image index not found
                print(f"Warning: Image index not found for {image_name}. Using default image.")
                sample = self.full_dataset[0]
            else:
                sample = self.full_dataset[image_idx]
                
            image = sample['image']
            
            # Get captions, ensuring we don't return None or empty list
            captions = self.image_to_captions.get(image_name, [])
            
            # Ensure captions is always a list (not None)
            if captions is None:
                captions = []
                
            # Filter out empty captions
            captions = [cap for cap in captions if cap and isinstance(cap, str)]
            
            # If no valid captions found, add a dummy caption to avoid errors
            if not captions:
                # Use a dummy caption "no description available"
                captions = ["no description available"]
                
            # Return structure: 
            # - 'image': tensor of shape [3, 224, 224]
            # - 'captions': list of strings (typically 5 captions for Flickr8k)
            # - 'image_name': string
            return {
                'image': image,
                'captions': captions,
                'image_name': image_name
            }
        except Exception as e:
            print(f"Error in UniqueImageDataset.__getitem__ for index {idx}: {str(e)}")
            # Return a default item to avoid crashing
            default_image = torch.zeros((3, 224, 224))
            return {
                'image': default_image,
                'captions': ["error loading caption"],
                'image_name': f"error_{idx}"
            }

def create_vocabulary(captions_file, min_freq=2):
    """Create a simple vocabulary from the caption file."""
    word_freq = {}
    
    print("Creating vocabulary...")
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
    
    print(f"Created vocabulary with {len(vocab)} tokens")
    return vocab


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Run one training epoch with progress bar."""
    model.train()
    total_loss = 0
    batch_losses = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        captions = batch['caption'].to(device)
        
        # Process the input captions excluding the last token (which is the target)
        outputs = model(images, captions[:, :-1])
        
        # Calculate loss using outputs and target captions (excluding the first token)
        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            captions[:, 1:].reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        batch_losses.append(loss.item())
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
    epoch_loss = total_loss / len(train_loader)
    return epoch_loss, batch_losses

def evaluate_model(model, val_loader, criterion, vocab, device):
    """
    Evaluate the image captioning model on validation data.
    
    Args:
        model: The trained captioning model
        val_loader: DataLoader for validation data
        criterion: Loss criterion (same as used in training)
        vocab: Vocabulary dictionary
        device: The device to run the model on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    id_to_word = {v: k for k, v in vocab.items()}
    total_loss = 0
    all_references = []
    all_hypotheses = []
    
    # Setup metric calculators
    rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            
            # For evaluation with UniqueImageDataset
            if 'captions' in batch:
                # This is for evaluation with multiple reference captions
                references = batch['captions']  # List of lists of reference captions
                
                # Get actual batch size (in case of partial last batch)
                # Ensure we don't try to access more references than we have
                batch_size = min(images.size(0), len(references))
                
                # Generate captions for each image
                for i in range(batch_size):
                    image = images[i].unsqueeze(0)  # Add batch dimension
                    # Generate caption
                    generated_caption = generate_caption_beam(model, image, vocab, device)                    
                    # Convert to word tokens and remove special tokens
                    pred_tokens = [id_to_word[idx] for idx in generated_caption 
                                  if idx not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
                    
                    # Reference captions for this image
                    img_references = references[i]
                    ref_tokens_list = []
                    
                    # Process each reference caption
                    for ref in img_references:
                        if isinstance(ref, str):
                            # If reference is a string, split into tokens
                            ref_tokens = ref.split()
                        else:
                            # If reference is already tokenized
                            ref_tokens = ref
                        ref_tokens_list.append(ref_tokens)
                    
                    all_references.append(ref_tokens_list)
                    all_hypotheses.append(pred_tokens)
            
            # For evaluation with SimpleDataset (during training)
            else:
                captions = batch['caption'].to(device)
                
                # Forward pass through the model
                outputs = model(images, captions[:, :-1])
                
                # Calculate loss
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    captions[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
                
                # Generate captions for BLEU calculation
                for i, image in enumerate(images):
                    # Get ground truth caption
                    target_caption = captions[i].cpu().numpy()
                    # Remove padding, start, and end tokens
                    target_tokens = [id_to_word[idx] for idx in target_caption 
                                    if idx not in [vocab['<pad>'], vocab['<start>'], vocab['<end>']]]
                    
                    # Generate caption
                    generated_caption = generate_caption_beam(model, image.unsqueeze(0), vocab, device)
                    pred_tokens = [id_to_word[idx] for idx in generated_caption 
                                  if idx not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
                    
                    all_references.append([target_tokens])
                    all_hypotheses.append(pred_tokens)
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate ROUGE scores
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for hyp, refs in zip(all_hypotheses, all_references):
        hyp_str = ' '.join(hyp)
        # Use the first reference to calculate ROUGE
        ref_str = ' '.join(refs[0])
        scores = rouge_calculator.score(ref_str, hyp_str)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    
    # Average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= len(all_hypotheses)
    
    # Calculate METEOR score (if nltk.translate.meteor_score is available)
    meteor_score_value = 0
    try:
        for hyp, refs in zip(all_hypotheses, all_references):
            # Use the first reference for simplicity
            meteor_score_value += meteor_score([refs[0]], hyp)
        meteor_score_value /= len(all_hypotheses)
    except:
        print("NLTK METEOR score calculation failed. Skipping.")
    
    # Get validation loss if we had captions
    val_loss = total_loss / len(val_loader) if 'caption' in batch else None
    
    # Collect all metrics
    metrics = {
        'bleu-1': bleu1 * 100,
        'bleu-2': bleu2 * 100,
        'bleu-3': bleu3 * 100,
        'bleu-4': bleu4 * 100,
        'rouge-1': rouge_scores['rouge1'] * 100,
        'rouge-2': rouge_scores['rouge2'] * 100,
        'rouge-L': rouge_scores['rougeL'] * 100,
        'meteor': meteor_score_value * 100,
    }
    
    if val_loss is not None:
        metrics['val_loss'] = val_loss
    
    return metrics

def generate_caption_beam(model, image, vocab, device, max_length=30, beam_size=3):
    """
    Generate a caption for an image using beam search.
    
    Args:
        model: The trained captioning model
        image: A single image tensor [1, C, H, W]
        vocab: Vocabulary dictionary
        device: The device to run inference on
        max_length: Maximum caption length
        beam_size: Beam width for search
        
    Returns:
        List of token indices representing the generated caption
    """
    model.eval()
    
    with torch.no_grad():
        # Extract image features and get spatial features
        img_features = model.image_encoder(image)
        img_features = model.adaptive_pool(img_features)
        img_features = model.image_proj(img_features)
     
        batch_size = image.size(0)
        spatial_features = img_features.view(batch_size, model.hidden_dim, -1).permute(0, 2, 1)
        
        # Initialize hidden state
        hidden = model.init_hidden(batch_size, device)
        
        # Start with the start token
        k = beam_size
        start_token = torch.tensor([[vocab['<start>']]]).to(device)
        
        # Initial input
        embeddings = model.embed_dropout(model.embedding(start_token))
        embeddings = model.embed_proj(embeddings)
        
        # Get context from spatial features and hidden state
        context, _ = model._apply_attention(hidden, spatial_features)
        
        # Combine embedded input with context
        gru_input = torch.cat([
            embeddings,  # [1, 1, hidden_dim]
            context.unsqueeze(1)  # [1, 1, hidden_dim]
        ], dim=2)
        
        # GRU step
        output, hidden = model.gru(gru_input, hidden)
        output = model.layer_norm(output)
        
        # Get initial predictions
        output = model.output(output)
        log_probs = F.log_softmax(output, dim=-1)
        
        # Get top-k candidates
        topk_probs, topk_indices = log_probs.view(-1).topk(k)
        
        # Create beam candidates: each is (log_prob, sequence, hidden_state)
        beams = []
        for i in range(k):
            token_idx = topk_indices[i].item()
            beam = {
                'log_prob': topk_probs[i].item(),
                'sequence': [vocab['<start>'], token_idx],
                'hidden': hidden
            }
            beams.append(beam)
        
        # Beam search iterations
        for step in range(2, max_length):
            candidates = []
            
            # Extend each beam
            for beam in beams:
                sequence = beam['sequence']
                hidden = beam['hidden']
                
                # If beam ended with <end> token, keep it as is
                if sequence[-1] == vocab['<end>']:
                    candidates.append(beam)
                    continue
                
                # Get last token from sequence
                last_token = torch.tensor([[sequence[-1]]]).to(device)
                
                # Embed token
                embeddings = model.embed_dropout(model.embedding(last_token))
                embeddings = model.embed_proj(embeddings)
                
                # Get context from spatial features and hidden state
                context, _ = model._apply_attention(hidden, spatial_features)
                
                # Combine embedded input with context
                gru_input = torch.cat([
                    embeddings,  # [1, 1, hidden_dim]
                    context.unsqueeze(1)  # [1, 1, hidden_dim]
                ], dim=2)
                
                # GRU step
                output, new_hidden = model.gru(gru_input, hidden)
                output = model.layer_norm(output)
                
                # Get predictions
                output = model.output(output)
                log_probs = F.log_softmax(output, dim=-1)
                
                # Get top-k candidates
                topk_probs, topk_indices = log_probs.view(-1).topk(k)
                
                # Create new candidates
                for i in range(k):
                    token_idx = topk_indices[i].item()
                    candidate = {
                        'log_prob': beam['log_prob'] + topk_probs[i].item(),
                        'sequence': sequence + [token_idx],
                        'hidden': new_hidden
                    }
                    candidates.append(candidate)
            
            # Sort candidates by log probability and keep top-k
            candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            beams = candidates[:k]
            
            # Check if all beams end with <end> token
            if all(beam['sequence'][-1] == vocab['<end>'] for beam in beams):
                break
        
        # Return the best beam's sequence
        return beams[0]['sequence']


def generate_caption(image_path, model, vocab, device='cpu', max_length=30):
    """
    Generate a caption for an image file using the model.
    
    Args:
        image_path: Path to the image file
        model: The trained captioning model
        vocab: Vocabulary dictionary
        device: Device to run inference on
        max_length: Maximum caption length
        
    Returns:
        Generated caption as a string
    """
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Generate caption using beam search
    model.eval()
    with torch.no_grad():
        idx_sequence = generate_caption_beam(model, image, vocab, device, max_length)
    
    # Convert indices to words
    id_to_word = {v: k for k, v in vocab.items()}
    words = [id_to_word.get(idx, '<unk>') for idx in idx_sequence 
             if idx not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
    
    return ' '.join(words)

# Add the missing import for F (nn.functional)
import torch.nn.functional as F

# This function needs to be updated to properly create image-to-captions mapping for evaluation
def prepare_evaluation_data(dataset):
    """
    Create mapping from image names to lists of reference captions.
    
    Args:
        dataset: SimpleDataset instance
        
    Returns:
        Dictionary mapping image names to lists of reference captions
    """
    image_to_captions = defaultdict(list)
    
    # Check that we have a valid dataset structure
    if not hasattr(dataset, 'samples'):
        print("Warning: Dataset does not have 'samples' attribute")
        return image_to_captions
    
    # Get unique image names from dataset
    unique_image_names = set()
    for sample in dataset.samples:
        unique_image_names.add(sample[0])
    
    print(f"Found {len(unique_image_names)} unique image names in dataset")
    
    # First pass: collect all captions
    for idx in range(len(dataset)):
        try:
            item = dataset[idx]
            image_name = dataset.samples[idx][0]  # Get image name from samples
            
            if 'raw_caption' not in item:
                print(f"Warning: Item {idx} does not have 'raw_caption' key. Available keys: {item.keys()}")
                continue
                
            caption = item['raw_caption']
            image_to_captions[image_name].append(caption)
        except Exception as e:
            print(f"Error processing item {idx} for evaluation data: {str(e)}")
    
    # Print some statistics
    image_count = len(image_to_captions)
    total_captions = sum(len(captions) for captions in image_to_captions.values())
    avg_captions = total_captions / image_count if image_count > 0 else 0
    
    print(f"Prepared evaluation data for {image_count} unique images")
    print(f"Total reference captions: {total_captions}")
    print(f"Average captions per image: {avg_captions:.2f}")
    
    # Count images with different numbers of captions
    caption_counts = {}
    for image_name, captions in image_to_captions.items():
        count = len(captions)
        caption_counts[count] = caption_counts.get(count, 0) + 1
    
    print("Distribution of captions per image:")
    for count, num_images in sorted(caption_counts.items()):
        print(f"  {count} caption(s): {num_images} image(s)")
    
    # Report on images without captions
    images_without_captions = unique_image_names - set(image_to_captions.keys())
    print(f"Found {len(images_without_captions)} images without any captions")
    if len(images_without_captions) > 0:
        print(f"Examples of images without captions: {list(images_without_captions)[:5]}")
    
    return image_to_captions


# Updated function to load a saved model for inference
def load_model_for_inference(checkpoint_path, device='cpu'):
    """Load model from checkpoint for inference."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint['vocab']
    config = checkpoint['config']
    
    # Initialize model with the same parameters
    model = LightweightCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, config
    
# New function for plotting metrics
def plot_metrics(train_losses, evaluation_metrics, batch_losses=None, output_dir='plots'):
    """
    Plot training losses and evaluation metrics.
    
    Args:
        train_losses: List of training losses per epoch
        evaluation_metrics: Dictionary of evaluation metrics per epoch
        batch_losses: List of lists containing batch losses for each epoch
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Matplotlib style for better looking plots
    plt.style.use('ggplot')
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='blue')
    plt.title('Training Loss per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_loss.png', dpi=300)
    plt.close()
    
    # Plot batch losses if provided (for the last epoch)
    if batch_losses and len(batch_losses) > 0:
        plt.figure(figsize=(12, 6))
        # Plot the most recent epoch's batch losses
        latest_batch_losses = batch_losses[-1]
        plt.plot(range(1, len(latest_batch_losses) + 1), latest_batch_losses, marker='.', linestyle='-', color='green')
        plt.title(f'Batch Losses for Epoch {len(train_losses)}', fontsize=16)
        plt.xlabel('Batch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/latest_batch_losses.png', dpi=300)
        
        # Plot a heatmap of all batch losses across epochs
        if len(batch_losses) > 1:  # Only if we have multiple epochs
            plt.figure(figsize=(12, 8))
            # Pad shorter batches lists to have the same length
            max_batches = max(len(batches) for batches in batch_losses)
            padded_batch_losses = [
                losses + [np.nan] * (max_batches - len(losses)) 
                for losses in batch_losses
            ]
            
            # Create heatmap
            plt.imshow(padded_batch_losses, aspect='auto', cmap='viridis')
            plt.colorbar(label='Loss')
            plt.title('Batch Losses Across Epochs', fontsize=16)
            plt.xlabel('Batch', fontsize=14)
            plt.ylabel('Epoch', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/batch_losses_heatmap.png', dpi=300)
        
        plt.close()
    
    # Plot evaluation metrics
    if evaluation_metrics:
        # Extract metrics
        epochs = sorted(evaluation_metrics.keys())
        metrics_dict = {}
        
        # Initialize metric dictionaries
        for epoch_idx in epochs:
            for metric_name in evaluation_metrics[epoch_idx].keys():
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(evaluation_metrics[epoch_idx][metric_name])
        
        # Plot each metric separately
        for metric_name, values in metrics_dict.items():
            plt.figure(figsize=(10, 6))
            # Ensure we're using the same number of epochs as we have values
            epoch_values = epochs[:len(values)]  # Ensure epochs and values have same length
            plt.plot(epoch_values, values, marker='o', linestyle='-', color='purple')
            plt.title(f'{metric_name.upper()} Score per Epoch', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel(f'{metric_name.upper()} Score', fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{metric_name}_score.png', dpi=300)
            plt.close()
        
        # Plot all BLEU scores together
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'green', 'orange', 'red']
        for i, metric_name in enumerate(['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']):
            if metric_name in metrics_dict:
                # Ensure we're using the same number of epochs as we have values
                epoch_values = epochs[:len(metrics_dict[metric_name])]
                plt.plot(epoch_values, metrics_dict[metric_name], marker='o', linestyle='-', 
                         label=f'{metric_name.upper()}', color=colors[i % len(colors)])
        
        plt.title('BLEU Scores per Epoch', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_bleu_scores.png', dpi=300)
        plt.close()
        
        # Create a comprehensive dashboard plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training Loss
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='blue')
        plt.title('Training Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True)
        
        # Plot 2: BLEU-1 and BLEU-2
        plt.subplot(2, 2, 2)
        if 'bleu-1' in metrics_dict and 'bleu-2' in metrics_dict:
            # Ensure we're using the same epochs for both metrics
            bleu1_epochs = epochs[:len(metrics_dict['bleu-1'])]
            bleu2_epochs = epochs[:len(metrics_dict['bleu-2'])]
            plt.plot(bleu1_epochs, metrics_dict['bleu-1'], marker='o', linestyle='-', label='BLEU-1', color='green')
            plt.plot(bleu2_epochs, metrics_dict['bleu-2'], marker='s', linestyle='--', label='BLEU-2', color='purple')
            plt.title('BLEU-1 and BLEU-2 Scores', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True)
        
        # Plot 3: BLEU-3 and BLEU-4
        plt.subplot(2, 2, 3)
        if 'bleu-3' in metrics_dict and 'bleu-4' in metrics_dict:
            # Ensure we're using the same epochs for both metrics
            bleu3_epochs = epochs[:len(metrics_dict['bleu-3'])]
            bleu4_epochs = epochs[:len(metrics_dict['bleu-4'])]
            plt.plot(bleu3_epochs, metrics_dict['bleu-3'], marker='o', linestyle='-', label='BLEU-3', color='orange')
            plt.plot(bleu4_epochs, metrics_dict['bleu-4'], marker='s', linestyle='--', label='BLEU-4', color='red')
            plt.title('BLEU-3 and BLEU-4 Scores', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True)
        
        # Plot 4: Most recent batch losses
        plt.subplot(2, 2, 4)
        if batch_losses and len(batch_losses) > 0:
            latest_batch_losses = batch_losses[-1]
            plt.plot(range(1, len(latest_batch_losses) + 1), latest_batch_losses, marker='.', 
                     linestyle='-', color='green')
            plt.title(f'Batch Losses (Epoch {len(train_losses)})', fontsize=14)
            plt.xlabel('Batch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_dashboard.png', dpi=300)
        plt.close()
    
    print(f"Plots saved to {output_dir} directory")

def evaluate_model_bleu_only(model, eval_loader, vocab, device, max_length=30):
    """
    Simplified evaluation function similar to the Keras implementation.
    Only computes BLEU scores for direct comparison.
    
    Args:
        model: The trained captioning model
        eval_loader: DataLoader for evaluation data with multiple references
        vocab: Vocabulary dictionary
        device: The device to run the model on
        max_length: Maximum caption length
        
    Returns:
        Dictionary of BLEU scores
    """
    model.eval()
    id_to_word = {v: k for k, v in vocab.items()}
    
    # Lists to store references and hypotheses
    all_references = []
    all_hypotheses = []
    
    # Skip special tokens during evaluation
    special_tokens = ['<start>', '<end>', '<pad>', '<unk>', 'startseq', 'endseq']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            try:
                # Extract images
                images = batch['image'].to(device)
                
                # Extract reference captions
                if 'captions' in batch:
                    # First print debug info about the batch structure
                    if batch_idx == 0:
                        print(f"Debug - batch['captions'] type: {type(batch['captions'])}")
                        print(f"Debug - batch['captions'] length: {len(batch['captions'])}")
                        if isinstance(batch['captions'], list):
                            print(f"Debug - first item type: {type(batch['captions'][0])}")
                            print(f"Debug - first item: {batch['captions'][0][:50]}...")  # Show first 50 chars
                    
                    # For the DataLoader from UniqueImageDataset, the captions are already
                    # structured as a list of lists of captions (one list per image)
                    captions_batch = batch['captions']
                elif 'all_captions' in batch:
                    captions_batch = batch['all_captions']
                elif 'caption' in batch:
                    captions = batch['caption']
                    captions_batch = [[" ".join([id_to_word[idx.item()] for idx in cap if idx.item() != 0])] 
                                   for cap in captions]
                else:
                    raise KeyError(f"Batch does not contain expected caption keys. Available keys: {batch.keys()}")
                
                # Process each image in the batch
                for i in range(len(images)):
                    # Get the captions for this image
                    image_captions = captions_batch[i] if i < len(captions_batch) else []
                    
                    # Skip if no captions available
                    if not image_captions:
                        print(f"Warning: No captions available for image {i} in batch {batch_idx}")
                        continue
                    
                    # Ensure image_captions is a list of strings
                    if isinstance(image_captions, str):
                        image_captions = [image_captions]
                    
                    # Generate a caption for the image
                    try:
                        # Generate caption using beam search
                        token_ids = generate_caption_beam(model, images[i:i+1], vocab, device, max_length, beam_size=3)
                        
                        # Convert token IDs to words
                        generated_caption = []
                        for idx in token_ids:
                            word = id_to_word.get(idx, None)
                            if word and word not in special_tokens:
                                generated_caption.append(word)
                        
                        # Skip if generated caption is empty
                        if not generated_caption:
                            continue
                            
                        # Process reference captions
                        references = []
                        for cap in image_captions:
                            # Skip if not a valid string
                            if not isinstance(cap, str) or not cap:
                                continue
                                
                            # Convert to list of words, removing special tokens
                            words = [w for w in cap.split() if w not in special_tokens]
                            if words:  # Only add non-empty references
                                references.append(words)
                        
                        # Add to lists for evaluation - format correctly for corpus_bleu
                        if references:
                            all_references.append(references)  # Removed the extra list nesting
                            all_hypotheses.append(generated_caption)
                    except Exception as e:
                        print(f"Error processing image {i} in batch {batch_idx}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue
    
    # Calculate BLEU scores using corpus_bleu similar to the Keras implementation
    from nltk.translate.bleu_score import corpus_bleu
    
    # Ensure we have at least some data to evaluate
    if not all_references or not all_hypotheses:
        print("Warning: No valid references or hypotheses collected. Cannot calculate BLEU scores.")
        return {
            'bleu-1': 0.0,
            'bleu-2': 0.0,
            'bleu-3': 0.0,
            'bleu-4': 0.0
        }
    
    # Print some statistics for debugging
    print(f"Evaluation complete. Collected {len(all_hypotheses)} generated captions with references.")
    print(f"Format check - references first item: {all_references[0][:1]}")
    print(f"Format check - hypothesis first item: {all_hypotheses[0][:10]}")
    
    try:
        bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1.0, 0, 0, 0))
        bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.3, 0.3, 0.3, 0))
        bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        
        # Return scores as percentages to match the Keras output
        return {
            'bleu-1': bleu1 * 100,
            'bleu-2': bleu2 * 100,
            'bleu-3': bleu3 * 100,
            'bleu-4': bleu4 * 100
        }
    except Exception as e:
        print(f"Error calculating BLEU scores: {str(e)}")
        print(f"Sample reference structure: {type(all_references[0])} with {len(all_references[0])} items")
        print(f"Sample reference first item: {all_references[0][0][:5]}")
        print(f"Sample hypothesis: {all_hypotheses[0]}")
        
        # Let's try with a simpler approach - recreate in the exact format needed
        try:
            print("Attempting with restructured data...")
            # Reformat data to exactly match what corpus_bleu expects
            list_of_references = [[ref] for ref in all_references]
            bleu1 = corpus_bleu(list_of_references, all_hypotheses, weights=(1.0, 0, 0, 0))
            bleu2 = corpus_bleu(list_of_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
            bleu3 = corpus_bleu(list_of_references, all_hypotheses, weights=(0.3, 0.3, 0.3, 0))
            bleu4 = corpus_bleu(list_of_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
            
            return {
                'bleu-1': bleu1 * 100,
                'bleu-2': bleu2 * 100,
                'bleu-3': bleu3 * 100,
                'bleu-4': bleu4 * 100
            }
        except Exception as e2:
            print(f"Second attempt failed: {str(e2)}")
            return {
                'bleu-1': 0.0,
                'bleu-2': 0.0,
                'bleu-3': 0.0,
                'bleu-4': 0.0
            }

# Modified main function to use all data for evaluation with plotting
def main():
    """Main training function with proper evaluation using the attention-based model."""
    try:
        # Configuration
        config = {
            'image_dir': '/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/',
            'captions_file': '/kaggle/input/multi-lingual-flickr8k/sanskrit_caption_p2.txt',
            'batch_size': 32,
            'embed_dim': 256,
            'hidden_dim': 512,
            'num_epochs': 80,
            'learning_rate': 1e-3,
            'train_val_split': 0.6,  # 80% train, 20% validation
            'plots_dir': 'training_plots',  # Directory to save plots
            'checkpoint_dir': 'checkpoints'  # Directory to save model checkpoints
        }
        
        # Create directories
        os.makedirs(config['plots_dir'], exist_ok=True)
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Create vocabulary
        vocab = create_vocabulary(config['captions_file'])
        
        # Create dataset for training
        full_dataset = SimpleDataset(
            config['image_dir'],
            config['captions_file'],
            vocab
        )
        
        # Split dataset for training and validation
        train_size = int(config['train_val_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create train and validation dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # Prepare evaluation dataset with multiple references per image
        print("Creating image-to-captions mapping for evaluation...")
        image_to_captions = prepare_evaluation_data(full_dataset)
        
        # Determine appropriate batch size for evaluation
        num_unique_images = len(image_to_captions)
        print(f"Number of unique images with captions: {num_unique_images}")
        
        # If we have very few unique images, use a smaller batch size
        eval_batch_size = min(8, max(1, num_unique_images // 2))
        print(f"Using evaluation batch size of {eval_batch_size}")
        
        # Define a custom collate function to handle lists of captions
        def eval_collate_fn(batch):
            # Extract all items
            images = [item['image'] for item in batch]
            captions = [item['captions'] for item in batch]
            image_names = [item['image_name'] for item in batch]
            
            # Stack images into a tensor
            images = torch.stack(images)
            
            return {
                'image': images,
                'captions': captions,
                'image_name': image_names
            }
        
        # Create evaluation dataset with unique images
        eval_dataset = UniqueImageDataset(full_dataset, image_to_captions)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=eval_batch_size,  # Adjusted batch size based on available data
            shuffle=False,
            num_workers=2,
            collate_fn=eval_collate_fn  # Use custom collate function
        )
        
        # Initialize model and training components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = LightweightCaptioningModel(
            vocab_size=len(vocab),
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Initialize lists to store metrics for plotting
        train_losses = []
        all_batch_losses = []
        evaluation_metrics = {}
        
        # Check if we have enough data for evaluation
        if num_unique_images < 5:
            print("WARNING: Very few unique images with captions available for evaluation.")
            print("BLEU scores may not be representative of model performance.")
            
        # Training loop with periodic evaluation
        best_bleu4 = 0
        for epoch in range(config['num_epochs']):
            # Training
            train_loss, batch_losses = train_epoch(
                model, train_loader, criterion, optimizer,
                device, epoch, config['num_epochs']
            )
            
            # Store losses for plotting
            train_losses.append(train_loss)
            all_batch_losses.append(batch_losses)
            
            # Single comprehensive evaluation using the BLEU-only method
            print(f"\nEvaluating model using BLEU metrics...")
            metrics = evaluate_model_bleu_only(
                model, 
                eval_loader,  # Using the eval_loader which has multiple references
                vocab,
                device
            )
            
            # Update learning rate based on BLEU-4 score
            scheduler.step(metrics['bleu-4'])
            
            # Store metrics for plotting
            evaluation_metrics[epoch + 1] = metrics
            
            # Log progress
            print(f'\nEpoch {epoch + 1} Summary:')
            print(f'Training Loss: {train_loss:.4f}')
            for metric, value in metrics.items():
                print(f'{metric}: {value:.4f}')
            
            # Plot metrics after each epoch
            plot_metrics(train_losses, evaluation_metrics, all_batch_losses, config['plots_dir'])
            
            # Save best model
            if metrics['bleu-4'] > best_bleu4:
                best_bleu4 = metrics['bleu-4']
                checkpoint_path = os.path.join(
                    config['checkpoint_dir'], 
                    f'best_sanskrit_captioner_bleu{best_bleu4:.2f}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab': vocab,
                    'config': config,
                    'metrics': metrics,
                    'train_losses': train_losses,
                    'evaluation_metrics': evaluation_metrics
                }, checkpoint_path)
                print(f'Saved new best model with BLEU-4: {best_bleu4:.4f}')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    config['checkpoint_dir'], 
                    f'checkpoint_epoch{epoch+1}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab': vocab,
                    'config': config,
                    'metrics': metrics,
                    'train_losses': train_losses,
                    'evaluation_metrics': evaluation_metrics
                }, checkpoint_path)
                
        # Final evaluation
        print("\nPerforming final evaluation...")
        final_metrics = evaluate_model_bleu_only(
            model, 
            eval_loader, 
            vocab,
            device
        )
        
        # Add final metrics to evaluation metrics
        evaluation_metrics[config['num_epochs']] = final_metrics
        
        # Generate final plots
        plot_metrics(train_losses, evaluation_metrics, all_batch_losses, config['plots_dir'])
        
        print("\nFinal Evaluation Results:")
        for metric, value in final_metrics.items():
            print(f'{metric}: {value:.4f}')
        
        # Save the final model
        torch.save({
            'epoch': config['num_epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': vocab,
            'config': config,
            'metrics': final_metrics,
            'train_losses': train_losses,
            'evaluation_metrics': evaluation_metrics
        }, 'final_sanskrit_captioner.pth')
        
        print("Training and evaluation completed successfully")
        
    except Exception as e:
        print(f"Error in training/evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()