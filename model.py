import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Splits image into patches and projects them to the embedding dimension.
    """
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches is equivalent to a Conv2d
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x) # (batch, embed_dim, grid_h, grid_w)
        x = x.flatten(2) # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2) # (batch, n_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Lightweight Vision Transformer (ViT) for chest X-ray feature extraction.
    """
    def __init__(self, img_size=128, patch_size=16, in_channels=1, num_classes=14, embed_dim=64, depth=2, num_heads=2, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional encodings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=mlp_dim, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Last layer gradients hook for Grad-CAM
        self.gradients = None
        self.activations = None
        
    def save_gradients(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        # Hook activations and gradients at the last block
        for i, block in enumerate(self.blocks.layers):
            x = block(x)
            if i == len(self.blocks.layers) - 1:
                # Save activations and register hook for backward pass (Grad-CAM)
                self.activations = x
                if x.requires_grad:
                    x.register_hook(self.save_gradients)
                    
        x = self.norm(x)
        
        # Return CLS token representation
        return x[:, 0]


class ClinicalMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for clinical EHR feature representation.
    """
    def __init__(self, input_dim=10, embed_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class MultimodalAttentionFusionModel(nn.Module):
    """
    Joint Multimodal Neural Network with Vision Transformer, Clinical MLP,
    and a Cross-Modal Attention Fusion block.
    """
    def __init__(self, clinical_dim=10, img_size=128, num_classes=14, embed_dim=64, fusion_heads=2, fusion_depth=1):
        super().__init__()
        # Visual processing stream (ViT)
        self.vit = VisionTransformer(img_size=img_size, num_classes=num_classes, embed_dim=embed_dim)
        
        # Clinical processing stream: Instead of standard MLP on whole vector,
        # we project each individual feature to a private 16-dim token space,
        # creating a sequence of 10 tokens representing clinical facts.
        self.clinical_dim = clinical_dim
        self.clin_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, embed_dim)
            ) for _ in range(clinical_dim)
        ])
        
        # Fusion Multi-head Self-Attention
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=fusion_heads,
            dim_feedforward=embed_dim * 2, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=fusion_depth)
        
        # Classification Head (FC Network)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes) # sigmoid will be applied via BCEWithLogitsLoss or in prediction
        )
        
        # To store attention weights for clinical explainability
        self.attention_weights = None
        
    def forward(self, img, clinical):
        # 1. Extract visual features (shape: batch, embed_dim)
        vis_feat = self.vit(img) # Shape: (batch, embed_dim)
        
        # 2. Extract clinical features as tokens
        # clinical shape: (batch, clinical_dim)
        clin_tokens = []
        for i in range(self.clinical_dim):
            feat_val = clinical[:, i:i+1] # shape: (batch, 1)
            token = self.clin_projections[i](feat_val) # shape: (batch, embed_dim)
            clin_tokens.append(token.unsqueeze(1)) # shape: (batch, 1, embed_dim)
            
        clin_tokens = torch.cat(clin_tokens, dim=1) # shape: (batch, clinical_dim, embed_dim)
        
        # 3. Form fusion sequence: [Visual Token, Clin Token 1, Clin Token 2, ...]
        # Sequence length: 1 + 10 = 11 tokens.
        vis_token = vis_feat.unsqueeze(1) # shape: (batch, 1, embed_dim)
        fusion_seq = torch.cat([vis_token, clin_tokens], dim=1) # shape: (batch, 11, embed_dim)
        
        # 4. Multi-modal Attention Fusion
        # We hook into the self-attention weights inside the fusion transformer block
        fused_seq = self.fusion_transformer(fusion_seq) # shape: (batch, 11, embed_dim)
        
        # Calculate attention weights manually for visual presentation
        # We perform QK^T calculation for the sequence to show the user.
        # This mirrors what the transformer encoder layers do internally.
        # Shape of fusion_seq: (batch, 11, embed_dim)
        q = fusion_seq[:, 0:1] # Query: Visual token (shape: batch, 1, embed_dim)
        k = fusion_seq        # Key: All tokens (shape: batch, 11, embed_dim)
        
        # Calculate scores: (batch, 1, 11)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (fusion_seq.shape[-1] ** 0.5)
        self.attention_weights = F.softmax(attn_scores, dim=-1).squeeze(1) # shape: (batch, 11)
        
        # The visual token at index 0 now represents the fused multimodal representation
        fused_rep = fused_seq[:, 0] # shape: (batch, embed_dim)
        
        # 5. Output multi-label logits
        logits = self.classifier(fused_rep)
        return logits


class ImageOnlyModel(nn.Module):
    """
    Baseline model using only chest X-ray images.
    """
    def __init__(self, img_size=128, num_classes=14, embed_dim=64):
        super().__init__()
        self.vit = VisionTransformer(img_size=img_size, num_classes=num_classes, embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, img, clinical=None):
        feats = self.vit(img)
        return self.classifier(feats)


class ClinicalOnlyModel(nn.Module):
    """
    Baseline model using only clinical tabular records.
    """
    def __init__(self, clinical_dim=10, num_classes=14, embed_dim=64):
        super().__init__()
        self.mlp = ClinicalMLP(input_dim=clinical_dim, embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, img, clinical):
        feats = self.mlp(clinical)
        return self.classifier(feats)
