# model.py

import torch
import torch.nn as nn
import math
import numpy as np


# ===== Model Components =====

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model
    """
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, embed_dim]
        """
        seq_len = x.size(1)
        
        # Handle case where sequence is longer than max_len
        if seq_len > self.max_len:
            print(f"Warning: Sequence length {seq_len} is longer than maximum positional encoding length {self.max_len}.")
            print(f"Truncating or using chunked processing is recommended for very long sequences.")
            
            # Apply positional encoding to the maximum length we have
            x[:self.max_len] = x[:self.max_len] + self.pe
            
            # For the rest, just repeat the pattern (not ideal but better than crashing)
            # This means positions beyond max_len will start repeating their encoding pattern
            for pos in range(self.max_len, seq_len, self.max_len):
                end_pos = min(pos + self.max_len, seq_len)
                x[:, pos:end_pos, :] += self.pe[:, :end_pos - pos,:]
        else:
            # Standard case: sequence length is within max_len
            x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)

class PianoTransformer(nn.Module):
    def __init__(self, n_cqt_bins=88, hidden_dim=256, num_heads=4, num_layers=3, 
                 dropout=0.1, max_len=1000):
        super(PianoTransformer, self).__init__()
        
        input_dim = n_cqt_bins # CQT bins
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True  # Use batch_first=True for better GPU performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers for onset, offset, and velocity
        self.note_layer = nn.Linear(hidden_dim, 88)  # Only note presence
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize the parameters of the model with bias initialization for output layers
        to ensure some predictions are active from the start
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize output layer biases to small positive values
        # This gives a small positive bias to the logits, which helps with the initial predictions
        # beacause the notes presences per frame are sparse (mostly 0) we need higher constant. 
        # changed it from 0.1 to 1.0. chatGPT say we should try also 2.0, 3.0 ...
        nn.init.constant_(self.note_layer.bias, 1.0)
    
    def forward(self, src, src_mask=None):
        """
        Forward pass
        
        Args:
            src: Source tensor of shape [batch_size, seq_len, input_dim]
            src_mask: Mask for source tensor
            
        Returns:
            note_presence: Note presence logits [batch_size, seq_len, 88]
            velocity: Velocity logits [batch_size, seq_len, 88]
        """
        # With batch_first=True, we don't need to transpose the input
        
        # Embed input
        src = self.input_norm(self.embedding(src) * math.sqrt(self.hidden_dim))
        
        # Add positional encoding - need to adapt this for batch_first=True
        # The positional encoding expects [seq_len, batch_size, hidden_dim]
        # So we transpose, add positional encoding, and transpose back
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Apply output layers to get logits
        note_presence = self.note_layer(output)
        
        # Return logits (without sigmoid) for loss function
        return note_presence
