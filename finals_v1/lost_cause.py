import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import os
from transformers.modeling_outputs import BaseModelOutput
from tqdm.auto import tqdm
import gc
import logging
import math
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict

@dataclass
class ModelConfig:
    batch_size: int = 16
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    image_encoder_lr: float = 2e-5
    num_epochs: int = 1
    accumulation_steps: int = 4
    max_length: int = 64
    warmup_ratio: float = 0.2
    dropout_rate: float = 0.3
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4
    mixed_precision: bool = True
    weight_decay: float = 0.02
    gradient_clip_val: float = 1.0
    num_beams: int = 4
    checkpoint_dir: str = 'checkpoints'
    early_stopping_patience: int = 8
    min_learning_rate: float = 1e-8
    cycles: int = 6
    fp16_training: bool = True  # Enable mixed precision

    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    num_attention_heads: int = 8
    fpn_channels: Tuple[int, ...] = (256, 512, 1024)
    num_encoder_layers: int = 3
    use_multi_scale: bool = True
    scale_factors: Tuple[float, ...] = (1.0, 0.5, 0.25)
    hidden_size: int = 1024
    intermediate_size: int = 4096

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()



class MultiScaleAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        attention_output = self.output(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        return attention_output

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # EfficientNet-B0 outputs 1280 channels
        self.in_channels = 1280
        self.fpn_channels = config.fpn_channels  # [256, 512, 1024]
        
        # Lateral connections from the 1280-channel input to FPN channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(self.in_channels, c, kernel_size=1)
            for c in self.fpn_channels
        ])
        
        # Top-down pathway with consistent channels
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.GELU()
            )
            for c in self.fpn_channels
        ])
        
        # Create adaptation convs going from larger to smaller channels
        # For channels [256, 512, 1024], we need:
        # 1024 -> 512 and 512 -> 256
        self.adapt_convs = nn.ModuleList([
            nn.Conv2d(self.fpn_channels[i], self.fpn_channels[i-1], kernel_size=1)
            for i in range(len(self.fpn_channels)-1, 0, -1)
        ])
        
        self.dropout = nn.Dropout2d(config.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Validate input dimensions and channels
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")
        
        # Bottom-up pathway: Create lateral connections
        laterals = []
        for conv in self.lateral_convs:
            lateral = conv(x)  # Convert from 1280 to respective FPN channels
            laterals.append(lateral)
        
        # Top-down pathway
        results = [laterals[-1]]  # Start with the deepest layer (1024 channels)
        
        for idx in range(len(self.fpn_channels) - 1):
            # Get the current feature map
            prev_features = results[-1]
            lateral = laterals[-(idx + 2)]  # Get corresponding lateral connection
            
            # Upsample previous features
            target_size = lateral.shape[-2:]
            upsampled = F.interpolate(prev_features, size=target_size, mode='nearest')
            
            # Adapt channels from larger to smaller (e.g., 1024->512 or 512->256)
            adapted = self.adapt_convs[idx](upsampled)
            
            # Merge features
            merged = lateral + adapted
            
            # Apply FPN convolution
            output = self.fpn_convs[-(idx + 2)](merged)
            output = self.dropout(output)
            
            results.append(output)
        
        # Return in order from smallest to largest channels (256, 512, 1024)
        return results[::-1]

class EnhancedImageCaptioningModel(nn.Module):
    def __init__(self, mbart_model, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained EfficientNet-B0
        self.image_encoder = models.efficientnet_b0(pretrained=True)
        self.image_encoder = self.image_encoder.features
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(config)
        
        # Multi-scale attention modules - one for each scale
        self.scale_attentions = nn.ModuleList([
            MultiScaleAttention(config)
            for _ in range(len(config.fpn_channels))
        ])
        
        # Feature projections for each FPN level
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout_rate)
            )
            for c in config.fpn_channels
        ])
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            MultiScaleAttention(config)
            for _ in range(config.num_encoder_layers)
        ])
        
        # Calculate total hidden size based on number of FPN channels
        total_hidden_size = config.hidden_size * len(config.fpn_channels)
        
        # Fusion layer with proper input dimension
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        self.mbart = mbart_model
        
    def forward(self, images: torch.Tensor, input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> Union[dict, torch.Tensor]:
        # Image encoding
        features = self.image_encoder(images)
        # print(f"Features shape after encoder: {features.shape}")
        
        # Multi-scale feature extraction through FPN
        fpn_features = self.fpn(features)
        
        # Process each scale
        processed_features = []
        for idx, (features, attention, projection) in enumerate(zip(fpn_features, self.scale_attentions, self.feature_projections)):
            # Get shape information
            B, C, H, W = features.shape
            
            # Reshape features to sequence
            features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Project features to hidden size
            features = projection(features)  # [B, H*W, hidden_size]
            
            # Apply attention
            features = attention(features)  # [B, H*W, hidden_size]
            
            # Global average pooling over spatial dimensions
            features = features.mean(dim=1)  # [B, hidden_size]
            
            processed_features.append(features)
        
        # Combine features
        combined_features = torch.cat(processed_features, dim=1)  # [B, total_hidden_size]
        fused_features = self.fusion_layer(combined_features)  # [B, hidden_size]
        
        # Expand fused features to sequence length
        fused_features = fused_features.unsqueeze(1)  # [B, 1, hidden_size]
        
        if input_ids is None:
            return {
                'last_hidden_state': fused_features,
                'attentions': None
            }
        
        # Get text features
        encoder_outputs = self.mbart.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Apply cross-attention
        hidden_states = encoder_hidden_states
        for cross_attn in self.cross_attention:
            hidden_states = cross_attn(hidden_states)
        
        # Generate caption
        outputs = self.mbart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=[hidden_states],
            return_dict=True
        )
        
        return outputs.logits    

class MixupAugmentation:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.alpha <= 0:
            return batch
            
        lambda_param = np.random.beta(self.alpha, self.alpha)
        batch_size = batch['image'].size(0)
        shuffle_idx = torch.randperm(batch_size)
        
        mixed_images = lambda_param * batch['image'] + (1 - lambda_param) * batch['image'][shuffle_idx]
        mixed_input_ids = batch['input_ids'].clone()
        mixed_attention_mask = batch['attention_mask'].clone()
        
        # Randomly select captions from either original or shuffled batch
        mask = torch.rand(batch_size) < lambda_param
        mixed_input_ids[mask] = batch['input_ids'][shuffle_idx][mask]
        mixed_attention_mask[mask] = batch['attention_mask'][shuffle_idx][mask]
        
        return {
            'image': mixed_images,
            'input_ids': mixed_input_ids,
            'attention_mask': mixed_attention_mask,
            'image_id': batch['image_id'],
            'all_captions': batch['all_captions']
        }

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(output, dim=-1)
        
        # Create smoothed targets
        vocab_size = output.size(-1)
        smoothed_targets = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))
        smoothed_targets.scatter_(-1, target.unsqueeze(-1), 1 - self.smoothing)
        
        # Mask padding tokens
        padding_mask = (target != self.ignore_index).float()
        loss = -torch.sum(log_probs * smoothed_targets, dim=-1)
        loss = torch.sum(loss * padding_mask) / torch.sum(padding_mask)
        
        return loss

from torch.utils.data.dataloader import default_collate

class EnhancedFlickrGujaratiDataset(Dataset):
    def __init__(self, image_dir: str, captions_file: str, tokenizer, transform=None, max_length: int = 64):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_captions = defaultdict(list)
        self.valid_images = []
        
        print("Validating images and loading captions...")
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    caption = parts[1]
                    
                    image_path = os.path.join(image_dir, img_name)
                    if os.path.exists(image_path):
                        try:
                            if img_name not in self.image_captions:
                                with Image.open(image_path) as img:
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    if self.transform:
                                        test_tensor = self.transform(img)
                                        if test_tensor.shape != (3, 224, 224):
                                            continue
                            self.image_captions[img_name].append(caption)
                            if img_name not in self.valid_images:
                                self.valid_images.append(img_name)
                        except Exception as e:
                            continue
                            
        print(f"Dataset initialized with {len(self.valid_images)} valid images")

    def __len__(self) -> int:
        return len(self.valid_images)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, List[str]]]:
        try:
            img_name = self.valid_images[idx]
            captions = self.image_captions[img_name]
            image_path = os.path.join(self.image_dir, img_name)
            
            # Select a random caption
            caption = np.random.choice(captions)
            
            # Load and process image
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                if self.transform:
                    image = self.transform(image)
            
            # Encode caption
            encoded_caption = self.tokenizer(
                caption,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            # Ensure tensors have the correct shape
            input_ids = encoded_caption['input_ids'].squeeze(0)
            attention_mask = encoded_caption['attention_mask'].squeeze(0)
            
            # Verify tensor shapes
            if input_ids.shape[0] != self.max_length or attention_mask.shape[0] != self.max_length:
                # If shapes are incorrect, pad or truncate
                if input_ids.shape[0] < self.max_length:
                    pad_length = self.max_length - input_ids.shape[0]
                    input_ids = F.pad(input_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
                    attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
                else:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
            
            return {
                'image': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image_id': img_name,
                'all_captions': captions
            }
        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")
            # Return the first valid item instead
            return self.__getitem__(0)

def custom_collate_fn(batch):
    """Custom collate function to ensure all tensors in the batch have the same size."""
    if len(batch) == 0:
        return {}
    
    error_count = 0
    valid_samples = []
    
    for sample in batch:
        try:
            # Verify sample structure and tensor sizes
            if all(k in sample for k in ['image', 'input_ids', 'attention_mask', 'image_id', 'all_captions']):
                if (sample['image'].shape == (3, 224, 224) and
                    sample['input_ids'].shape == sample['attention_mask'].shape):
                    valid_samples.append(sample)
                else:
                    error_count += 1
            else:
                error_count += 1
        except:
            error_count += 1
    
    if len(valid_samples) == 0:
        return None
    
    # Group the valid samples by key
    collated_batch = {}
    for key in valid_samples[0].keys():
        if key in ['image', 'input_ids', 'attention_mask']:
            collated_batch[key] = torch.stack([s[key] for s in valid_samples])
        else:
            collated_batch[key] = [s[key] for s in valid_samples]
    
    return collated_batch

def create_data_loaders(config: ModelConfig, tokenizer):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    full_dataset = EnhancedFlickrGujaratiDataset(
        image_dir="/kaggle/input/flickr8k/Flickr_Data/Flickr_Data/Images/",
        captions_file="/kaggle/input/guj-captions/gujarati_captions.txt",
        tokenizer=tokenizer,
        transform=transform,
        max_length=config.max_length
    )

    # Calculate splits
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

class ImageCaptioningTrainer:
    def __init__(self, model: nn.Module, config: ModelConfig, tokenizer, device: torch.device):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.scaler = GradScaler() if config.mixed_precision else None
        self.optimizer = self._create_optimizer()
        self.mixup = MixupAugmentation(config.mixup_alpha)
        self.criterion = LabelSmoothingLoss(config.label_smoothing, self.tokenizer.pad_token_id)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def _create_optimizer(self) -> optim.AdamW:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        
    def _should_stop_early(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict()
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                return True
        return False
        
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Optional[float]:
        if batch is None:
            return None
            
        # Apply mixup augmentation
        batch = self.mixup(batch)
        
        # Move batch to device
        images = batch['image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.mixed_precision):
            outputs = self.model(images, input_ids, attention_mask)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            loss = loss / self.config.accumulation_steps
        
        # Backward pass with gradient scaling
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation and optimizer step
        if (self.batch_idx + 1) % self.config.accumulation_steps == 0:
            if self.config.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            clear_gpu_memory()

        return loss.item() * self.config.accumulation_steps
        
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                if batch is None:
                    continue
                
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(images, input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                
                total_val_loss += loss.item()
                val_batch_count += 1
                
                if val_batch_count % 50 == 0:
                    torch.cuda.empty_cache()
        
        return total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        
    def generate_caption(self, image: torch.Tensor) -> str:
        self.model.eval()
        with torch.no_grad():
            try:
                image = image.to(self.device)
                
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                
                encoder_outputs = self.model(image)
                last_hidden_state = encoder_outputs['last_hidden_state']
                
                dummy_input_ids = torch.ones((1, 1), dtype=torch.long, device=self.device)
                proper_encoder_outputs = BaseModelOutput(
                    last_hidden_state=last_hidden_state,
                    hidden_states=None,
                    attentions=None
                )
                
                outputs = self.model.mbart.generate(
                    input_ids=dummy_input_ids,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True,
                    encoder_outputs=proper_encoder_outputs,
                    return_dict_in_generate=False
                )
                
                caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return caption
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    return ""
                raise e

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        smooth = SmoothingFunction()
        rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        image_references = defaultdict(list)
        image_hypotheses = {}
        
        print("Generating captions for evaluation...")
        for batch in tqdm(test_loader):
            if batch is None:
                continue
            
            image_id = batch['image_id'][0]
            all_captions = batch['all_captions'][0]
            generated_caption = self.generate_caption(batch['image'])
            
            if generated_caption:
                image_references[image_id].extend(all_captions)
                image_hypotheses[image_id] = generated_caption
        
        references = []
        hypotheses = []
        
        for image_id in image_hypotheses.keys():
            references.append([self._tokenize_gujarati(ref) for ref in image_references[image_id]])
            hypotheses.append(self._tokenize_gujarati(image_hypotheses[image_id]))
        
        # Calculate metrics
        bleu_scores = {}
        weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
        for i, w in enumerate(weights, 1):
            bleu_scores[f'bleu{i}'] = corpus_bleu(references, hypotheses, 
                                                weights=w, 
                                                smoothing_function=smooth.method1)
        
        meteor_scores = []
        rouge_scores = defaultdict(list)
        
        for refs, hyp in zip(references, hypotheses):
            meteor_scores.append(max(meteor_score([ref], hyp) for ref in refs))
            
            best_rouge_scores = None
            best_rouge_total = -1
            
            for ref in refs:
                ref_text = ' '.join(ref)
                hyp_text = ' '.join(hyp)
                rouge_result = rouge_calculator.score(ref_text, hyp_text)
                rouge_total = sum(score.fmeasure for score in rouge_result.values())
                
                if rouge_total > best_rouge_total:
                    best_rouge_scores = rouge_result
                    best_rouge_total = rouge_total
            
            for metric, score in best_rouge_scores.items():
                rouge_scores[metric].append(score.fmeasure)
        
        results = {
            **bleu_scores,
            'meteor': np.mean(meteor_scores),
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
        
        return results
    
    def _tokenize_gujarati(self, text: str) -> List[str]:
        """Tokenize Gujarati text."""
        punctuation = ".,!?।॥''""()[]{}:;-"
        
        for p in punctuation:
            text = text.replace(p, f" {p} ")
        
        return [token.strip() for token in text.split() if token.strip()]
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> Dict[str, float]:
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        best_bleu4 = 0
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_train_loss = 0
            batch_count = 0
            
            progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{self.config.num_epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                self.batch_idx = batch_idx
                loss = self._training_step(batch)
                
                if loss is not None:
                    total_train_loss += loss
                    batch_count += 1
                    progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                if batch_count % 100 == 0:
                    torch.cuda.empty_cache()
            
            avg_train_loss = total_train_loss / batch_count if batch_count > 0 else float('inf')
            avg_val_loss = self._validate(val_loader, self.criterion)
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            if (epoch + 1) % 1 == 0:
                eval_results = self.evaluate(test_loader)
                print("\nBLEU Scores:")
                print(f"  BLEU-1: {eval_results['bleu1']:.4f}")
                print(f"  BLEU-2: {eval_results['bleu2']:.4f}")
                print(f"  BLEU-3: {eval_results['bleu3']:.4f}")
                print(f"  BLEU-4: {eval_results['bleu4']:.4f}")
                print("\nOther Metrics:")
                print(f"  METEOR: {eval_results['meteor']:.4f}")
                print(f"  ROUGE-1: {eval_results['rouge1']:.4f}")
                print(f"  ROUGE-2: {eval_results['rouge2']:.4f}")
                print(f"  ROUGE-L: {eval_results['rougeL']:.4f}")
                current_bleu4 = eval_results['bleu4']

                if current_bleu4 > best_bleu4:
                    best_bleu4 = current_bleu4
                    
            if self._should_stop_early(avg_val_loss):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                self.model.load_state_dict(self.best_model_state)
                break
        
        # Final evaluation
        print("\nTraining completed. Running final evaluation...")
        final_results = self.evaluate(test_loader)
        
        return final_results




def main():
    # Set environment variables and config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(level=logging.INFO)
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # Empty cache before starting
        clear_gpu_memory()
    # Initialize tokenizer
    tokenizer = MBartTokenizer.from_pretrained(
        "facebook/mbart-large-cc25",
        src_lang="gu_IN",
        tgt_lang="gu_IN"
    )
    
    # Initialize model configuration
    mbart_config = MBartConfig.from_pretrained("facebook/mbart-large-cc25")
    mbart_config.gradient_checkpointing = True
    mbart_config.dropout = config.dropout_rate
    mbart_config.use_cache = False  # Disable KV cache during training

    # Initialize model
    mbart_model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-cc25",
        config=mbart_config
    )
    
    # Create model and move to device
    model = EnhancedImageCaptioningModel(mbart_model, config).to(device)
    
    # Create data loaders with tokenizer
    train_loader, val_loader, test_loader = create_data_loaders(config, tokenizer)
    
    # Create trainer and start training
    trainer = ImageCaptioningTrainer(model, config, tokenizer, device)
    results = trainer.train(train_loader, val_loader, test_loader)
    
    return results

if __name__ == "__main__":
    main()