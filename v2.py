import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from typing import Optional
import random
from googletrans import Translator

class ImageCaptioningDataset(Dataset):
    """
    Enhanced dataset class with data augmentation for both images and captions
    """
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
        self.data = pd.read_csv(captions_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.translator = Translator()
        
        # Enhanced image augmentation pipeline
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),  # Larger size for random cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Non-augmented transform for validation
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def augment_caption(self, caption: str) -> str:
        """
        Augment caption using back-translation
        """
        try:
            # Translate to English and back to Gujarati for paraphrasing
            english = self.translator.translate(caption, src='gu', dest='en').text
            augmented = self.translator.translate(english, src='en', dest='gu').text
            return augmented
        except:
            return caption

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        image_path = self.image_dir / row['image_path']
        caption = row['gujarati_caption']

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.augment:
            image = self.transform(image)
            # Randomly augment caption during training
            if random.random() < 0.3:
                caption = self.augment_caption(caption)
        else:
            image = self.eval_transform(image)

        caption_encoding = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': caption_encoding['input_ids'].squeeze(),
            'attention_mask': caption_encoding['attention_mask'].squeeze(),
            'caption': caption
        }

class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder with multi-head attention
    """
    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            dropout=0.1,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerDecoder(
            self.transformer_layer,
            num_layers=6
        )
        
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Transform target tokens to match transformer dimensions
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        # Apply transformer decoder
        output = self.transformer(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to vocabulary size
        output = self.output_projection(output)
        return output.transpose(0, 1)

class ImageEncoder(nn.Module):
    """
    Enhanced image encoder using pre-trained ResNet with feature projection
    """
    def __init__(self, encoded_dim: int = 2048):
        super().__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Keep spatial dimensions
        self.resnet = nn.Sequential(*modules)
        
        # Add adaptive pooling and projection
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.projection = nn.Linear(2048, encoded_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.flatten(2).transpose(1, 2)
        return self.projection(features)

class GujaratiImageCaptioning(nn.Module):
    """
    Enhanced image captioning model with transformer decoder
    """
    def __init__(
        self,
        config: XLMRobertaConfig,
        encoded_dim: int = 2048
    ):
        super().__init__()
        
        self.encoder = ImageEncoder(encoded_dim)
        self.roberta = XLMRobertaModel(config)
        self.decoder = TransformerDecoder(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Encode images
        image_features = self.encoder(images)
        
        # Get RoBERTa embeddings
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = roberta_outputs.last_hidden_state
        
        # Create transformer masks
        tgt_mask = self.generate_square_subsequent_mask(hidden_states.size(1))
        tgt_mask = tgt_mask.to(hidden_states.device)
        
        # Apply transformer decoder
        output = self.decoder(
            hidden_states,
            image_features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~attention_mask.bool()
        )
        
        return output
        
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

def beam_search_with_length_norm(
    model: nn.Module,
    image_features: torch.Tensor,
    tokenizer: XLMRobertaTokenizer,
    beam_width: int = 5,
    max_length: int = 128,
    temperature: float = 1.0,
    alpha: float = 0.7,  # Length normalization parameter
    device: torch.device = torch.device('cpu')
) -> List[str]:
    """
    Enhanced beam search with length normalization
    """
    def length_normalize(sequence_length: int, alpha: float) -> float:
        """
        Apply length normalization to score
        """
        return ((5 + sequence_length) ** alpha) / (6 ** alpha)
    
    # Initialize beam with start tokens
    beam = [(torch.tensor([[tokenizer.bos_token_id]]).to(device), 0)]
    
    for _ in range(max_length):
        candidates = []
        
        for sequence, score in beam:
            if sequence[0, -1].item() == tokenizer.eos_token_id:
                # Apply length normalization to completed sequences
                normalized_score = score / length_normalize(sequence.size(1), alpha)
                candidates.append((sequence, normalized_score))
                continue
                
            # Get model predictions
            attention_mask = torch.ones_like(sequence)
            outputs = model.decoder(
                model.roberta(sequence, attention_mask).last_hidden_state,
                image_features,
                tgt_mask=model.generate_square_subsequent_mask(sequence.size(1)).to(device)
            )
            logits = outputs[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Get top-k next tokens
            values, indices = probs[0].topk(beam_width)
            
            for value, token in zip(values, indices):
                new_sequence = torch.cat([sequence, token.unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score - torch.log(value).item()
                # Apply length normalization to partial sequences
                normalized_score = new_score / length_normalize(new_sequence.size(1), alpha)
                candidates.append((new_sequence, normalized_score))
        
        # Select top-k candidates
        candidates.sort(key=lambda x: x[1])
        beam = candidates[:beam_width]
        
        # Stop if all beams ended with EOS
        if all(b[0][0, -1].item() == tokenizer.eos_token_id for b in beam):
            break
    
    # Decode captions
    captions = []
    for sequence, _ in beam:
        caption = tokenizer.decode(sequence.squeeze().tolist(), skip_special_tokens=True)
        captions.append(caption)
    
    return captions

def calculate_metrics(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics
    """
    # Initialize Rouge scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate BLEU score
    bleu = corpus_bleu([[ref] for ref in references], hypotheses)
    
    # Calculate METEOR score
    meteor = np.mean([meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)])
    
    # Calculate ROUGE scores
    rouge_scores = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0
    }
    for ref, hyp in zip(references, hypotheses):
        scores = rouge_scorer_instance.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    
    # Average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= len(references)
    
    return {
        'bleu': bleu,
        'meteor': meteor,
        **rouge_scores
    }

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    tokenizer: XLMRobertaTokenizer,
    scheduler=None,
    patience: int = 3,
    grad_clip: float = 1.0
) -> Dict[str, List[float]]:
    """
    Enhanced training loop with mixed precision and gradient clipping
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': []
    }
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Mixed precision training
            with autocast():
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1)
                )
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1)
                )
                val_loss += loss.item()
                
                # Generate captions for evaluation
                image_features = model.encoder(images)
                generated_captions = []
                for features in image_features:
                    captions = beam_search_with_length_norm(
                        model,
                        features.unsqueeze(0),
                        tokenizer,
                        device=device,
                        alpha=0.7  # Length normalization parameter
                    )
                    generated_captions.append(captions[0])  # Take top beam
                
                # Prepare references and hypotheses
                references.extend([caption.split() for caption in batch['caption']])
                hypotheses.extend([caption.split() for caption in generated_captions])
        
        # Calculate all evaluation metrics
        metrics = calculate_metrics(references, hypotheses)
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['metrics'].append(metrics)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {history["train_loss"][-1]:.4f}')
        print(f'Val Loss: {history["val_loss"][-1]:.4f}')
        print('Metrics:')
        for metric_name, value in metrics.items():
            print(f'  {metric_name}: {value:.4f}')
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Early stopping check
        current_val_loss = val_loss / len(val_loader)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'history': history
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return history

def main():
    """
    Main function to set up and train the model
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize tokenizer and config
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    config = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
    
    # Create datasets with augmentation for training
    train_dataset = ImageCaptioningDataset(
        image_dir='path/to/images',
        captions_file='path/to/captions.csv',
        tokenizer=tokenizer,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = ImageCaptioningDataset(
        image_dir='path/to/images',
        captions_file='path/to/captions.csv',
        tokenizer=tokenizer,
        augment=False  # Disable augmentation for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = GujaratiImageCaptioning(config=config)
    
    # Set up training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-5,
        epochs=10,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% warmup
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
        tokenizer=tokenizer,
        scheduler=scheduler,
        patience=3,
        grad_clip=1.0
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, 'final_model.pth')
    
    return history

if __name__ == "__main__":
    main()