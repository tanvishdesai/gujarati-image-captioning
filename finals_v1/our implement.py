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
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import json

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
        corrupted_images = 0
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    image_path = os.path.join(image_dir, img_name)
                    
                    if os.path.exists(image_path):
                        try:
                            with Image.open(image_path) as img:
                                img.verify()
                                self.data.append((img_name, parts[1]))
                        except Exception as e:
                            corrupted_images += 1
                            continue
                    else:
                        missing_images += 1
        
        print(f"Dataset initialized with {len(self.data)} valid image-caption pairs")
        print(f"Skipped {missing_images} missing images")
        print(f"Skipped {corrupted_images} corrupted images")
    
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
            raise RuntimeError(f"Error loading verified image {image_path}: {str(e)}")
        
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
        
        self.image_encoder = models.efficientnet_b0(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        
        self.mbart_hidden_size = mbart_model.config.hidden_size
        
        self.feature_projection = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.mbart_hidden_size)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.mbart_hidden_size * 2, self.mbart_hidden_size),
            nn.LayerNorm(self.mbart_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.mbart_hidden_size, self.mbart_hidden_size),
            nn.LayerNorm(self.mbart_hidden_size)
        )
        
        self.mbart = mbart_model
        
    def forward(self, images, input_ids=None, attention_mask=None):
        if input_ids is None:
            with torch.no_grad():
                image_features = self.image_encoder(images)
                image_features = image_features.mean(dim=[2, 3])
                image_features = self.feature_projection(image_features)
                
                encoder_outputs = {
                    'last_hidden_state': image_features.unsqueeze(1),
                    'attentions': None
                }
                
                return encoder_outputs
        
        image_features = self.image_encoder(images)
        image_features = image_features.mean(dim=[2, 3])
        image_features = self.feature_projection(image_features)
        
        encoder_outputs = self.mbart.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state
        image_features = image_features.unsqueeze(1).expand(-1, encoder_hidden_states.size(1), -1)
        combined_features = torch.cat([encoder_hidden_states, image_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        outputs = self.mbart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=[fused_features],
            return_dict=True
        )
        
        return outputs.logits

def generate_caption(model, image, tokenizer, device, max_length=64):
    model.eval()
    with torch.no_grad():
        try:
            model = model.half()
            image = image.to(device).half()
            
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            encoder_outputs = model(image)
            last_hidden_state = encoder_outputs['last_hidden_state'].half()
            
            dummy_input_ids = torch.ones((1, 1), dtype=torch.long, device=device)
            
            from transformers.modeling_outputs import BaseModelOutput
            proper_encoder_outputs = BaseModelOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=None,
                attentions=None
            )
            
            outputs = model.mbart.generate(
                input_ids=dummy_input_ids,
                max_length=max_length,
                num_beams=2,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
                encoder_outputs=proper_encoder_outputs,
                return_dict_in_generate=False
            )
            
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            del encoder_outputs, proper_encoder_outputs
            torch.cuda.empty_cache()
            
            return caption
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"WARNING: OOM error. Current GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                return ""
            raise e

def tokenize_gujarati(text):
    punctuation = ".,!?।॥''""()[]{}:;-"
    for p in punctuation:
        text = text.replace(p, f" {p} ")
    tokens = [token.strip() for token in text.split() if token.strip()]
    return tokens

def evaluate_model(model, test_loader, tokenizer, device, output_file='evaluation_results.json'):
    model.eval()
    smooth = SmoothingFunction()
    rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results_buffer = {
        'generated': [],
        'reference': []
    }
    
    buffer_size = 50
    total_processed = 0
    accumulated_scores = None
    
    def process_buffer(buffer):
        if not buffer['generated']:
            return {}
        
        scores = {
            'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0,
            'meteor': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0
        }
        
        refs = [[tokenize_gujarati(r)] for r in buffer['reference']]
        hyps = [tokenize_gujarati(h) for h in buffer['generated']]
        
        # Calculate BLEU scores
        weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
        for i, w in enumerate(weights, 1):
            scores[f'bleu{i}'] = corpus_bleu(refs, hyps, weights=w, smoothing_function=smooth.method1)
        
        # Calculate METEOR and ROUGE scores
        for ref, hyp in zip(buffer['reference'], buffer['generated']):
            ref_tokens = tokenize_gujarati(ref)
            hyp_tokens = tokenize_gujarati(hyp)
            
            scores['meteor'] += meteor_score([ref_tokens], hyp_tokens)
            
            rouge_scores = rouge_calculator.score(ref, hyp)
            scores['rouge1'] += rouge_scores['rouge1'].fmeasure
            scores['rouge2'] += rouge_scores['rouge2'].fmeasure
            scores['rougeL'] += rouge_scores['rougeL'].fmeasure
        
        n = len(buffer['generated'])
        return {k: v/n for k, v in scores.items()}
    
    print("Generating captions and evaluating in chunks...")
    try:
        for batch in tqdm(test_loader):
            if batch is None:
                continue
                
            for i in range(len(batch['image'])):
                generated_caption = generate_caption(model, batch['image'][i], tokenizer, device)
                
                if generated_caption:
                    reference_caption = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                    results_buffer['generated'].append(generated_caption)
                    results_buffer['reference'].append(reference_caption)
                    
                    if len(results_buffer['generated']) >= buffer_size:
                        chunk_scores = process_buffer(results_buffer)
                        if accumulated_scores is None:
                            accumulated_scores = chunk_scores
                        else:
                            for k in accumulated_scores:
                                accumulated_scores[k] = (accumulated_scores[k] * total_processed + 
                                                    chunk_scores[k] * buffer_size) / (total_processed + buffer_size)
                        
                        total_processed += buffer_size
                        results_buffer = {'generated': [], 'reference': []}
                        gc.collect()
                        torch.cuda.empty_cache()
            
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if not accumulated_scores:
            raise e
    
    # Process any remaining samples in the buffer
    if results_buffer['generated']:
        final_scores = process_buffer(results_buffer)
        if accumulated_scores is None:
            accumulated_scores = final_scores
        else:
            remaining = len(results_buffer['generated'])
            for k in accumulated_scores:
                accumulated_scores[k] = (accumulated_scores[k] * total_processed + 
                                    final_scores[k] * remaining) / (total_processed + remaining)
    
    # Save and print results
    if accumulated_scores:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(accumulated_scores, f, indent=4)
        
        print(f"\nProcessed {total_processed} images successfully")
        print("\nEvaluation Results:")
        for metric, value in accumulated_scores.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("No scores were accumulated during evaluation")
    
    return accumulated_scores
def train_and_evaluate_model(model, train_loader, val_loader, test_loader, tokenizer, num_epochs, device, 
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
        
        # # Save checkpoint
        # checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        #     'val_loss': avg_val_loss,
        #     'best_val_loss': best_val_loss,
        # }, checkpoint_path)
        # print(f"Saved checkpoint to {checkpoint_path}")
        
        # Perform evaluation after the final epoch
        if epoch == num_epochs - 1:
            print("\nTraining completed. Starting evaluation...")
            model = model.half()  # Convert to half precision for evaluation
            evaluation_results = evaluate_model(
                model=model,
                test_loader=test_loader,
                tokenizer=tokenizer,
                device=device,
                output_file=os.path.join(checkpoint_dir, 'final_evaluation_results.json')
            )
            return evaluation_results

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
    
    # Split dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size]
    )
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for evaluation
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    # Initialize model
    model = EfficientImageCaptioningModel(mbart_model).to(device)
    model.mbart.gradient_checkpointing_enable()
    
    # Train and evaluate model
    evaluation_results = train_and_evaluate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        num_epochs=5,
        device=device,
        checkpoint_dir='model_checkpoints'
    )
    
    print("\nFinal Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()