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
from tqdm.auto import tqdm
import gc
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlickrGujaratiDataset(Dataset):
    def __init__(self, image_dir, captions_file, tokenizer, transform=None, max_length=64):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and validate data
        self.data = []
        missing_images = 0
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    image_path = os.path.join(image_dir, img_name)
                    if os.path.exists(image_path):
                        self.data.append((img_name, parts[1]))
                    else:
                        missing_images += 1
        
        print(f"Dataset initialized with {len(self.data)} valid image-caption pairs")
        if missing_images > 0:
            print(f"{missing_images} images were not found and will be skipped")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        image_path = os.path.join(self.image_dir, img_name)
        
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                if self.transform:
                    image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
        
        encoded_caption = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoded_caption['input_ids'].squeeze(),
            'attention_mask': encoded_caption['attention_mask'].squeeze()
        }

class CustomBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        batch_indices = []
        while len(batch_indices) < self.batch_size and self.current_idx < len(self.dataset):
            batch_indices.append(self.indices[self.current_idx])
            self.current_idx += 1
        
        return batch_indices
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch])
    }

class EfficientImageCaptioningModel(nn.Module):
    def __init__(self, mbart_model, dropout_rate=0.1):
        super().__init__()
        
        # Image encoder
        self.image_encoder = models.efficientnet_b0(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        
        # Get mBART's hidden size
        self.mbart_hidden_size = mbart_model.config.hidden_size
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),  # Changed from ReLU to GELU
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.mbart_hidden_size)
        )
        
        # Fusion layer with improved architecture
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.mbart_hidden_size * 2, self.mbart_hidden_size),
            nn.LayerNorm(self.mbart_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.mbart_hidden_size, self.mbart_hidden_size),
            nn.LayerNorm(self.mbart_hidden_size)
        )
        
        self.mbart = mbart_model
        
    def forward(self, images, input_ids, attention_mask):
        # Process image features
        with torch.cuda.amp.autocast():
            image_features = self.image_encoder(images)
            image_features = image_features.mean(dim=[2, 3])
            image_features = self.feature_projection(image_features)
        
        # Get mBART encoder outputs
        encoder_outputs = self.mbart.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Feature fusion
        encoder_hidden_states = encoder_outputs.last_hidden_state
        image_features = image_features.unsqueeze(1).expand(-1, encoder_hidden_states.size(1), -1)
        combined_features = torch.cat([encoder_hidden_states, image_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        # Generate output
        outputs = self.mbart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=[fused_features],
            return_dict=True
        )
        
        return outputs.logits

def train_model(model, train_loader, val_loader, tokenizer, num_epochs, device, 
                accumulation_steps=4, mixed_precision=True, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    optimizer = optim.AdamW([
        {'params': model.image_encoder.parameters(), 'lr': 1e-5},
        {'params': model.feature_projection.parameters(), 'lr': 2e-5},
        {'params': model.fusion_layer.parameters(), 'lr': 2e-5},
        {'params': model.mbart.parameters(), 'lr': 2e-5}
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler() if mixed_precision else None
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        # Training loop with simple progress bar
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            if batch is None:
                continue
                
            images = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            with autocast(enabled=mixed_precision):
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                loss = loss / accumulation_steps
            
            if mixed_precision:
                scaler.scale(loss).backward()
                if (batch_count + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_count + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps
            batch_count += 1
            
            if batch_count % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        avg_train_loss = total_train_loss / batch_count
        print(f"Epoch {epoch+1} training completed. Average loss: {avg_train_loss:.4f}")
        
        # Validation loop
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                if batch is None:
                    continue
                    
                images = batch['image'].to(device, non_blocking=True)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                
                with autocast(enabled=mixed_precision):
                    outputs = model(images, input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                
                total_val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = total_val_loss / val_batch_count
        print(f"Validation completed. Average validation loss: {avg_val_loss:.4f}")
        current_epoch = epoch + 1  # Convert to 1-based indexing for clarity

        # Save checkpoint if validation loss improves or every 5 epochs
        if current_epoch == 3 or current_epoch % 5 == 0:
            best_val_loss = min(avg_val_loss, best_val_loss)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

def main():
    # Set environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cudnn.benchmark = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25",
                                              src_lang="gu_IN",
                                              tgt_lang="gu_IN")
    
    # Load mBART model
    mbart_config = MBartConfig.from_pretrained("facebook/mbart-large-cc25")
    mbart_config.gradient_checkpointing = True
    mbart_model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-cc25",
        config=mbart_config
    )
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = FlickrGujaratiDataset(
        "/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/",
        "/kaggle/input/guj-captions/gujarati_captions.txt",
        tokenizer,
        transform,
        max_length=64
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=CustomBatchSampler(train_dataset, batch_size=8, shuffle=True),
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=CustomBatchSampler(val_dataset, batch_size=8, shuffle=False),
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = EfficientImageCaptioningModel(mbart_model).to(device)
    model.mbart.gradient_checkpointing_enable()
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        num_epochs=5,
        device=device,
        checkpoint_dir='model_checkpoints'
    )

if __name__ == "__main__":
    main()