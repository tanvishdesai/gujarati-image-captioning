import torch
import torch.nn as nn
from PIL import Image
import logging
from pathlib import Path
import argparse
from transformers import MT5Tokenizer, BlipProcessor, BlipForConditionalGeneration,MT5ForConditionalGeneration

# Define the model class (same as in your training code)
class GujaratiCaptioningModelV8(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize BLIP with stable dtype
        self.blip = BlipForConditionalGeneration.from_pretrained(
            config.blip_model,
            torch_dtype=torch.float32
        )
        self.blip_processor = BlipProcessor.from_pretrained(config.blip_model)
        
        # Freeze BLIP parameters
        for param in self.blip.parameters():
            param.requires_grad = False
        
        # Initialize mT5 with stable dtype
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(
            config.mt5_model,
            torch_dtype=torch.float32
        )
        
        # Get hidden sizes
        blip_hidden_size = self.blip.config.vision_config.hidden_size
        mt5_hidden_size = self.mt5.config.hidden_size
        
        # Feature processing with enhanced normalization
        self.feature_projection = nn.Sequential(
            nn.Linear(blip_hidden_size, mt5_hidden_size),
            nn.LayerNorm(mt5_hidden_size, eps=1e-6),  # Increased eps
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Modified cross-attention with scaling and normalization
        self.query_norm = nn.LayerNorm(mt5_hidden_size, eps=1e-6)
        self.key_norm = nn.LayerNorm(mt5_hidden_size, eps=1e-6)
        self.value_norm = nn.LayerNorm(mt5_hidden_size, eps=1e-6)
        self.output_norm = nn.LayerNorm(mt5_hidden_size, eps=1e-6)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=mt5_hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Initialize weights with careful scaling
        self._init_weights()
        
        # Temperature scaling for attention
        self.attention_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def _init_weights(self):
        """Initialize weights with careful scaling."""
        def _init_layer(module):
            if isinstance(module, nn.Linear):
                # Xavier initialization with smaller bounds
                torch.nn.init.xavier_uniform_(
                    module.weight,
                    gain=0.1
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        self.feature_projection.apply(_init_layer)
        
        # Initialize attention weights with smaller values
        with torch.no_grad():
            for param in self.cross_attention.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_normal_(param, gain=0.1)
    
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        """Forward pass with enhanced numerical stability."""
        try:
            # Ensure inputs are float32
            images = images.to(dtype=torch.float32)
            
            # Get BLIP features with gradient checkpointing
            with torch.no_grad():
                blip_features = self.blip.vision_model(
                    pixel_values=images
                ).last_hidden_state
            
            # Project and normalize features
            projected_features = self.feature_projection(blip_features)
            
            # Get mT5 encoder outputs
            encoder_outputs = self.mt5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Normalize inputs to cross-attention
            query = self.query_norm(encoder_outputs.last_hidden_state)
            key = self.key_norm(projected_features)
            value = self.value_norm(projected_features)
            
            # Apply cross-attention with scaled dot product
            enhanced_features, _ = self._scaled_dot_product_attention(query, key, value)
            
            # Final output normalization
            enhanced_features = self.output_norm(enhanced_features)
            
            # Generate outputs
            outputs = self.mt5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=(enhanced_features,),
                labels=labels,
                return_dict=True
            )
            
            return outputs
        
        except Exception as e:
            logging.error(f"Forward pass error: {str(e)}")
            raise

    @torch.no_grad()
    def generate_caption(self, image, tokenizer, max_length=64, num_beams=4):
        """Generate caption with error checking."""
        try:
            self.eval()
            
            # Ensure input is float32
            image = image.to(dtype=torch.float32)
            
            # Process image with BLIP
            blip_features = self.blip.vision_model(
                pixel_values=image.unsqueeze(0)
            ).last_hidden_state
            
            # Check for valid features
            if not torch.isfinite(blip_features).all():
                raise ValueError("Non-finite values in BLIP features during generation")
            
            # Project features
            projected_features = self.feature_projection(blip_features)
            
            if not torch.isfinite(projected_features).all():
                raise ValueError("Non-finite values after feature projection during generation")
            
            # Generate caption
            input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(image.device)
            
            outputs = self.mt5.generate(
                input_ids=input_ids,
                encoder_outputs=torch.nn.utils.rnn.pad_sequence(
                    [projected_features], 
                    batch_first=True
                ),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=0.8
            )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return ""

class ImageCaptionInference:
    def __init__(self, model_path):
        # Load the saved model and configurations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load configurations
        self.config = self.checkpoint['config']
        
        # Initialize tokenizer and processor
        self.tokenizer = MT5Tokenizer.from_pretrained(self.config['mt5_model'])
        self.processor = BlipProcessor.from_pretrained(self.config['blip_model'])
        
        # Initialize model
        self.model = GujaratiCaptioningModelV8(self.config).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
    
    def generate_caption(self, image_path, max_length=64):
        """Generate caption for a single image."""
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image = self.processor(images=image, return_tensors="pt")["pixel_values"][0].to(self.device)
            
            # Generate caption
            caption = self.model.generate_caption(
                image=image,
                tokenizer=self.tokenizer,
                max_length=max_length
            )
            
            return caption
            
        except Exception as e:
            logging.error(f"Error generating caption: {str(e)}")
            return "Error generating caption"

def main():
    # Set default paths
    default_image_path = r"flickr8k\Flickr_Data\Flickr_Data\Images\19212715_20476497a3.jpg"
    default_model_path = r"C:\Users\DELL\Downloads\best_model_epoch_1.pth"
    
    try:
        # Initialize captioning system
        captioner = ImageCaptionInference(default_model_path)
        
        # Generate and print caption
        caption = captioner.generate_caption(default_image_path)
        print(f"\nImage Path: {default_image_path}")
        print(f"Generated Caption: {caption}")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    main()