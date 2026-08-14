import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class TrOCR(nn.Module):
    """
    معماری ساده‌شده TrOCR:
    Vision Transformer (Encoder) + Transformer Decoder
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6):
        super(TrOCR, self).__init__()
        
        # Encoder: Vision Transformer
        self.encoder = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.encoder_proj = nn.Linear(1000, d_model)
        
        # Decoder: Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Embedding & Output
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 256, d_model))
    
    def forward(self, images, targets):
        # Encode
        encoder_out = self.encoder(images)
        encoder_out = self.encoder_proj(encoder_out).unsqueeze(1)
        
        # Decode
        tgt_emb = self.embedding(targets) + self.positional_encoding[:, :targets.size(1), :]
        decoded = self.decoder(tgt_emb, encoder_out)
        output = self.fc_out(decoded)
        
        return output
