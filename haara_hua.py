import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration
import pandas as pd
from PIL import Image
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import os
from tqdm import tqdm

# Download required NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')

class FlickrGujaratiDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Load and process captions
        self.data = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    caption = parts[1]
                    self.data.append((img_name, caption))
                    
        # Remove duplicate images while keeping all captions
        self.image_to_captions = {}
        for img_name, caption in self.data:
            if img_name not in self.image_to_captions:
                self.image_to_captions[img_name] = []
            self.image_to_captions[img_name].append(caption)
            
        # Create a list of unique images with their first caption
        self.unique_data = [(img_name, captions[0]) 
                           for img_name, captions in self.image_to_captions.items()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Tokenize caption
        encoded_caption = self.tokenizer(caption, 
                                       padding='max_length',
                                       max_length=128,
                                       truncation=True,
                                       return_tensors='pt')
        
        return {
            'image': image,
            'input_ids': encoded_caption['input_ids'].squeeze(),
            'attention_mask': encoded_caption['attention_mask'].squeeze()
        }

class ImageCaptioningModel(nn.Module):
    def __init__(self, mbart_model):
        super().__init__()
        
        # Get mBART's hidden size
        self.mbart_hidden_size = mbart_model.config.hidden_size  # Usually 1024 for mBART
        
        # Image encoder (ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Project image features to match mBART hidden size
        self.feature_projection = nn.Linear(2048, self.mbart_hidden_size)  # ResNet outputs 2048 features
        
        # Additional layers to process image features
        self.image_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.mbart_hidden_size,
            nhead=8,
            dim_feedforward=2048
        )
        
        # mBART model
        self.mbart = mbart_model
        
        # Fusion layer to combine image and text features
        self.fusion_layer = nn.Linear(self.mbart_hidden_size * 2, self.mbart_hidden_size)
        
    def forward(self, images, input_ids, attention_mask):
        # 1. Process image features (batch_size x 2048)
        image_features = self.image_encoder(images)
        image_features = image_features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        
        # 2. Project to mBART hidden size (batch_size x mbart_hidden_size)
        image_features = self.feature_projection(image_features)
        
        # 3. Prepare image features for transformer (batch_size x 1 x mbart_hidden_size)
        image_features = image_features.unsqueeze(1)
        image_features = self.image_encoder_layer(image_features)
        
        # 4. Get mBART encoder outputs (batch_size x seq_len x mbart_hidden_size)
        encoder_outputs = self.mbart.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 5. Combine features
        encoder_hidden_states = encoder_outputs.last_hidden_state
        expanded_image_features = image_features.expand(-1, encoder_hidden_states.size(1), -1)
        
        # Concatenate along feature dimension
        # Shape: (batch_size x seq_len x (mbart_hidden_size * 2))
        combined_features = torch.cat([encoder_hidden_states, expanded_image_features], dim=-1)
        
        # 6. Fuse features (batch_size x seq_len x mbart_hidden_size)
        fused_features = self.fusion_layer(combined_features)
        
        # 7. Generate output through mBART
        decoder_outputs = self.mbart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=[fused_features],
            return_dict=True
        )
        
        return decoder_outputs.logits

def train_model(model, train_loader, val_loader, tokenizer, num_epochs, device, 
                accumulation_steps=4, mixed_precision=True):
    """
    Train the image captioning model with improved monitoring.
    
    Args:
        model: The image captioning model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        tokenizer: MBart tokenizer
        num_epochs: Number of training epochs (default is 10 in your main function)
        device: Training device (cuda/cpu)
        accumulation_steps: Number of steps for gradient accumulation
        mixed_precision: Whether to use mixed precision training
    """
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler() if mixed_precision else None
    
    print(f"Starting training for {num_epochs} epochs")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader)):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.amp.autocast(device_type='cuda', enabled=mixed_precision):
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), 
                               input_ids.view(-1))
                loss = loss / accumulation_steps
            
            if mixed_precision:
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), 
                               input_ids.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= 4:
                print("Early stopping triggered")
                break
        
        # Calculate metrics on validation set if it's the final epoch
        if epoch == num_epochs - 1:
            bleu1, bleu4, meteor, rouge = evaluate_model(model, val_loader, device, tokenizer)
            print(f"Final Metrics - BLEU-1: {bleu1:.2f}, BLEU-4: {bleu4:.2f}, "
                  f"METEOR: {meteor:.2f}, ROUGE: {rouge:.2f}")
 

    return model            

def create_train_val_dataloaders(dataset, batch_size, val_split=0.1):
    """
    Create train and validation dataloaders from a single dataset.
    Uses unique images for splitting to prevent data leakage.
    """
    # Calculate lengths for split
    total_length = len(dataset.unique_data)
    val_length = int(total_length * val_split)
    train_length = total_length - val_length
    
    # Create train/val splits based on unique images
    train_indices = set()
    val_indices = set()
    
    # Randomly split unique images
    unique_indices = torch.randperm(total_length).tolist()
    train_unique_indices = unique_indices[val_length:]
    val_unique_indices = unique_indices[:val_length]
    
    # Map unique image indices to all corresponding caption indices
    for idx, (img_name, _) in enumerate(dataset.data):
        img_unique_idx = next(i for i, (unique_img, _) 
                            in enumerate(dataset.unique_data) 
                            if unique_img == img_name)
        if img_unique_idx in train_unique_indices:
            train_indices.add(idx)
        else:
            val_indices.add(idx)
    
    # Create subset random samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(list(train_indices))
    val_sampler = torch.utils.data.SubsetRandomSampler(list(val_indices))
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def evaluate_model(model, val_loader, device, tokenizer):
    """
    Evaluate the model using various metrics.
    
    Parameters:
        model: The image captioning model
        val_loader: DataLoader for validation data
        device: Device to evaluate on (cuda/cpu)
        tokenizer: The mBART tokenizer
    """
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['image'].to(device)
            
            # Generate captions
            generated_ids = model.mbart.generate(
                encoder_hidden_states=model.feature_projection(
                    model.image_encoder(images).squeeze(-1).squeeze(-1)
                ).unsqueeze(1),
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode generated captions
            generated_captions = tokenizer.batch_decode(generated_ids, 
                                                      skip_special_tokens=True)
            actual_captions = tokenizer.batch_decode(batch['input_ids'], 
                                                   skip_special_tokens=True)
            
            # Prepare for metric calculation
            references.extend([[cap.split()] for cap in actual_captions])
            hypotheses.extend([cap.split() for cap in generated_captions])
    
    # Calculate metrics
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate METEOR and ROUGE
    meteor_scores = []
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        meteor_scores.append(meteor_score(ref, hyp))
        rouge_scores.append(
            rouge_scorer_obj.score(' '.join(ref[0]), ' '.join(hyp))['rouge1'].fmeasure
        )
    
    meteor_final = sum(meteor_scores) / len(meteor_scores)
    rouge_final = sum(rouge_scores) / len(rouge_scores)
    
    return bleu1 * 100, bleu4 * 100, meteor_final * 100, rouge_final * 100

def main():
    """
    Main function to set up and run the training pipeline.
    """
    # Initialize tokenizer and model
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", 
                                              src_lang="gu_IN", 
                                              tgt_lang="gu_IN")
    mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = FlickrGujaratiDataset(
        r"flickr8k\Flickr_Data\Flickr_Data\Images",
        "gujarati_captions.txt",
        tokenizer,
        transform
    )
    
    # Create train and validation dataloaders
    train_loader, val_loader = create_train_val_dataloaders(dataset, batch_size=16)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptioningModel(mbart_model).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, tokenizer, num_epochs=10, device=device)

if __name__ == "__main__":
    main()