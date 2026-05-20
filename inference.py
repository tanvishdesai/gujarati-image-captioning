import math
import torch
from PIL import Image
import logging
from transformers import AutoTokenizer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os
from transformers import AutoTokenizer, ViTModel

import torch.nn.functional as F

import torch
import torch.nn as nn
# Constants (should match your training code)
HIDDEN_SIZE = 768
MAX_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the image preprocessing pipeline

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, ff_size, max_len, dropout=0.1):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, hidden_size))
        
        # Add input projection for memory
        self.memory_proj = nn.Linear(hidden_size, hidden_size)
        
        # Pre-norm architecture
        self.pre_norm = nn.LayerNorm(hidden_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        self.hidden_size = hidden_size
        self.max_len = max_len
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.pos_enc, mean=0, std=0.02)
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Project and normalize memory (image features)
        memory = self.memory_proj(memory)
        
        # Embed and add positional encoding
        tgt = self.embed(tgt) * math.sqrt(self.hidden_size)
        tgt = tgt + self.pos_enc[:, :tgt.size(1), :]
        
        # Generate attention mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        
        # Pre-norm
        tgt = self.pre_norm(tgt)
        
        # Decoder forward pass
# Ensure proper mask type
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.bool()
        
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )     
        output = self.final_norm(output)
        return self.fc_out(output)
    
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class ViTCaptioningModel(nn.Module):
    def __init__(self, vit_model, decoder):
        super().__init__()
        self.vit = vit_model
        self.decoder = decoder
        
        # Project ViT features to decoder dimension
        self.proj = nn.Sequential(
            nn.Linear(vit_model.config.hidden_size, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(0.1)
        )
        
        self.img_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.vit.gradient_checkpointing_enable()
    
    def forward(self, images, captions, attention_mask=None):
        # Extract and project image features
        with torch.cuda.amp.autocast():
            vit_output = self.vit(pixel_values=images).last_hidden_state
            image_features = self.proj(vit_output)
            image_features = self.img_norm(image_features)
        
        # Generate output
# Convert attention_mask to proper boolean type
        if attention_mask is not None:
            tgt_key_padding_mask = attention_mask.bool()
        else:
            tgt_key_padding_mask = None
        
        outputs = self.decoder(
            captions,
            image_features,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return outputs
    
    @torch.no_grad()
    def generate_with_beam(self, images, beam_size=5, max_length=MAX_LENGTH):
        """Fixed beam search generation"""
        batch_size = images.size(0)
        device = images.device
        
        # Get image features
        vit_output = self.vit(pixel_values=images).last_hidden_state
        image_features = self.proj(vit_output)
        image_features = self.img_norm(image_features)
        
        # Initialize return tensor
        generated_sequences = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        
        # Process each image in batch separately
        for idx in range(batch_size):
            # Get single image features and expand for beam size
            img_feat = image_features[idx:idx+1].expand(beam_size, -1, -1)
            
            # Initialize sequence with start token
            curr_sequences = torch.full((beam_size, 1), 
                                     self.decoder.embed.num_embeddings-1,
                                     dtype=torch.long, 
                                     device=device)
            
            sequence_scores = torch.zeros(beam_size, device=device)
            complete_sequences = []
            complete_scores = []
            
            for step in range(max_length - 1):
                # Get predictions
                outputs = self.decoder(curr_sequences, img_feat)
                token_probs = F.log_softmax(outputs[:, -1, :], dim=-1)
                
                # Get top k scores and tokens
                vocab_size = token_probs.size(-1)
                scores = token_probs + sequence_scores.unsqueeze(1)
                scores = scores.view(-1)
                top_scores, top_indices = scores.topk(beam_size, dim=0)
                
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # Update sequences
                new_sequences = torch.cat([
                    curr_sequences[beam_indices],
                    token_indices.unsqueeze(1)
                ], dim=1)
                
                # Handle completed sequences
                for seq_idx in range(beam_size):
                    if token_indices[seq_idx] == self.decoder.embed.num_embeddings-2:  # End token
                        complete_sequences.append(new_sequences[seq_idx].clone())
                        complete_scores.append(top_scores[seq_idx].clone())
                
                # Update current sequences and scores
                incomplete_indices = token_indices != self.decoder.embed.num_embeddings-2
                if not incomplete_indices.any():
                    break
                
                curr_sequences = new_sequences[incomplete_indices]
                sequence_scores = top_scores[incomplete_indices]
                
                if curr_sequences.size(0) < beam_size:
                    pad_size = beam_size - curr_sequences.size(0)
                    curr_sequences = torch.cat([curr_sequences, curr_sequences[:pad_size]], dim=0)
                    sequence_scores = torch.cat([sequence_scores, sequence_scores[:pad_size]], dim=0)
            
            # Select best sequence
            if complete_sequences:
                complete_sequences = torch.stack(complete_sequences)
                complete_scores = torch.stack(complete_scores)
                _, best_idx = complete_scores.max(dim=0)
                best_sequence = complete_sequences[best_idx]
            else:
                best_sequence = curr_sequences[0]
            
            # Pad if necessary
            if best_sequence.size(0) < max_length:
                padding = torch.full((max_length - best_sequence.size(0),),
                                  self.decoder.embed.num_embeddings-2,
                                  dtype=torch.long,
                                  device=device)
                best_sequence = torch.cat([best_sequence, padding])
            
            generated_sequences[idx] = best_sequence[:max_length]
        
        return generated_sequences
    
# Define the image preprocessing pipeline
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ImageCaptionInference:
    def __init__(self, model_path):
        """
        Initialize the image captioning inference system with a pre-trained model
        
        Args:
            model_path: Path to the saved model checkpoint
        """
        self.transform = Compose([
            Resize((224, 224)),     # Resize images to ViT's expected input size
            ToTensor(),             # Convert PIL Image to tensor
            Normalize(              # Normalize with standard ImageNet statistics
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
            )
        ])
        self.device = DEVICE
        print(f"Loading model from {model_path}")
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize tokenizer with larger vocabulary size to match the checkpoint
        print("Initializing tokenizer with expanded vocabulary...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/indic-bert",
            use_fast=True,
            strip_accents=False,
            # Set model_max_length to match your MAX_LENGTH constant
            model_max_length=MAX_LENGTH,
            # Pad the vocabulary to match the checkpoint size
            vocab_size=203879
        )
        
        # Add special tokens for Gujarati
        special_tokens_dict = {
            'pad_token': '[PAD]',
            'bos_token': '<શરૂ>',
            'eos_token': '<અંત>',
            'unk_token': '[UNK]'
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Initialize ViT model
        print("Initializing ViT model...")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
        
        # Initialize decoder with the checkpoint's vocabulary size
        print(f"Initializing decoder with vocabulary size: 203879")
        self.decoder = TransformerDecoder(
            vocab_size=203879,  # Match the checkpoint's vocabulary size
            hidden_size=HIDDEN_SIZE,
            num_layers=8,
            num_heads=12,
            ff_size=3072,
            max_len=MAX_LENGTH
        )
        
        # Initialize the full model
        self.model = ViTCaptioningModel(self.vit_model, self.decoder).to(self.device)
        
        # Load model weights
        print("Loading model weights...")
        try:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            print("Model loaded successfully!")
        except RuntimeError as e:
            print(f"Error loading model weights: {str(e)}")
            raise
    
    def generate_caption(self, image_path):
        """
        Generate a caption for the given image
        
        Args:
            image_path: Path to the input image
            
        Returns:
            str: Generated caption
        """
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate caption
            self.model.eval()
            with torch.no_grad():
                generated_ids = self.model.generate_with_beam(
                    image,
                    beam_size=5,
                    max_length=MAX_LENGTH
                )
                
            # Decode caption
            caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            raise

def main():
    # Set default paths
    default_image_path = r"flickr8k\Flickr_Data\Flickr_Data\Images\44856031_0d82c2c7d1.jpg"
    default_model_path = r"C:\Users\DELL\Downloads\gujarati_caption_model_final.pt"  # Update with your model path
    
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