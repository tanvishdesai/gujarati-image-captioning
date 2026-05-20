import json
import torch
import torch.nn as nn
import torchvision.models as models

# Load the fixed notebook to extract model architecture
with open('train-italian-fixed.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Define the model class from the notebook
class LightweightCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        # Use ResNet18 instead of MobileNetV2 for better efficiency
        resnet = models.resnet18(pretrained=False)
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
            num_layers=2,
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
        
        # Initialize hidden state with image features - FIXED LINE
        hidden = image_features.unsqueeze(0).repeat(2, 1, 1)  # Shape: [2, batch_size, hidden_dim]
        
        # Decode
        output, _ = self.decoder(embedded, hidden)
        output = self.output(output)
        
        return output

# Test model with dummy data
def test_model():
    print("Testing model with dummy data...")
    
    # Create a model with a small vocab size
    model = LightweightCaptioningModel(vocab_size=1000, embed_dim=256, hidden_dim=512)
    
    # Create dummy data
    batch_size = 16
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_captions = torch.randint(0, 1000, (batch_size, 20))
    
    # Forward pass
    try:
        outputs = model(dummy_images, dummy_captions)
        print(f"Success! Output shape: {outputs.shape}")
        print("The model now works correctly.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_model() 