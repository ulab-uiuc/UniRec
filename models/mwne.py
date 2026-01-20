import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from typing import Optional, Tuple, Union

class MathematicallyAwareNormalizer(nn.Module):
    """
    Enhanced normalization that better preserves mathematical properties
    """
    def __init__(self, encoder, target_std=1.0, momentum=0.99, min_std=0.1):
        super().__init__()
        self.encoder = encoder
        self.target_std = target_std
        self.momentum = momentum
        self.min_std = min_std
        
        # Track running statistics
        self.register_buffer('running_mean', torch.zeros(encoder.embedding_dim))
        self.register_buffer('running_std', torch.ones(encoder.embedding_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        
        # Track mathematical property preservation
        self.register_buffer('additivity_error_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))
        
    def forward(self, numbers):
        embeddings = self.encoder(numbers)
        
        if self.training:
            # Update statistics more conservatively
            batch_std = embeddings.std(dim=0, keepdim=True)
            
            # Only update if the change is reasonable (prevents sudden jumps)
            if self.num_batches_tracked > 0:
                std_change = torch.abs(batch_std.squeeze() - self.running_std)
                max_allowed_change = self.running_std * 0.5  # Max 50% change
                
                # Apply momentum only if change is reasonable
                valid_update = std_change < max_allowed_change
                self.running_std = torch.where(
                    valid_update,
                    self.momentum * self.running_std + (1 - self.momentum) * batch_std.squeeze(),
                    self.running_std
                )
            else:
                self.running_std = batch_std.squeeze()
            
            self.num_batches_tracked += 1
            
            # Clamp std to reasonable bounds
            self.running_std = torch.clamp(self.running_std, min=self.min_std)
        
        # Apply scaling with stability
        scaling_factor = self.target_std / (self.running_std.unsqueeze(0) + 1e-8)
        
        # Clip scaling factor to prevent extreme values
        scaling_factor = torch.clamp(scaling_factor, min=0.1, max=10.0)
        
        scaled = embeddings * scaling_factor
        
        return scaled
    
    def validate_mathematical_properties(self, test_numbers):
        """Monitor property preservation during training"""
        with torch.no_grad():
            # Test additivity
            a, b = test_numbers[:2], test_numbers[2:4]
            embed_a = self.forward(a)
            embed_b = self.forward(b)
            embed_sum = self.forward(a + b)
            
            additivity_error = F.mse_loss(embed_a + embed_b, embed_sum).item()
            
            # Store in history
            idx = self.history_idx % 100
            self.additivity_error_history[idx] = additivity_error
            self.history_idx += 1
            
            # Return average error over recent history
            recent_error = self.additivity_error_history[:min(self.history_idx, 100)].mean().item()
            
            return {
                'current_error': additivity_error,
                'recent_avg_error': recent_error,
                'std_stability': self.running_std.std().item()
            }

class ImprovedMathematicalEncoder(nn.Module):
    """
    Redesigned encoder with stronger mathematical guarantees and simpler training
    """
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 num_frequencies: int = 16,
                 max_frequency: float = 100.0,
                 include_raw: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies
        self.max_frequency = max_frequency
        self.include_raw = include_raw
        self.device = device
        
        # Fixed logarithmic frequency spacing for better coverage
        frequencies = torch.logspace(-2, math.log10(max_frequency), num_frequencies)
        self.register_buffer('frequencies', frequencies)
        
        # Dimension allocation
        fourier_dim = 2 * num_frequencies  # cos + sin for each frequency
        raw_dim = 2 if include_raw else 0  # raw value + sign
        remaining_dim = embedding_dim - fourier_dim - raw_dim
        
        if remaining_dim < 0:
            raise ValueError(f"embedding_dim {embedding_dim} too small for {fourier_dim} + {raw_dim}")
        
        # Simple linear projections (no complex networks that could break properties)
        self.fourier_weight = nn.Parameter(torch.ones(fourier_dim))
        if remaining_dim > 0:
            self.extra_proj = nn.Linear(1, remaining_dim, bias=False)
        else:
            self.extra_proj = None
        
        # Learnable scaling for the raw component
        if include_raw:
            self.raw_scale = nn.Parameter(torch.tensor([1.0, 1.0]))
    
    def forward(self, numbers: torch.Tensor) -> torch.Tensor:
        """
        Encode numbers with strong mathematical property preservation
        """
        original_shape = numbers.shape
        x = numbers.view(-1, 1).float()
        
        components = []
        
        # 1. Fourier component (multi-scale periodic encoding)
        fourier_component = self.compute_fourier_component(x)
        components.append(fourier_component)
        
        # 2. Raw value component (preserves linear relationships)
        if self.include_raw:
            raw_component = self.compute_raw_component(x)
            components.append(raw_component)
        
        # 3. Additional learned component
        if self.extra_proj is not None:
            extra_component = self.extra_proj(x)
            components.append(extra_component)
        
        # Concatenate all components
        embedding = torch.cat(components, dim=-1)
        
        # Reshape back to original
        return embedding.view(*original_shape, -1)
    
    def compute_fourier_component(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weighted Fourier features"""
        # x: [batch_size, 1], frequencies: [num_frequencies]
        phases = x * self.frequencies.unsqueeze(0)  # [batch_size, num_frequencies]
        
        cos_features = torch.cos(phases)
        sin_features = torch.sin(phases)
        
        # Interleave cos and sin
        fourier_features = torch.stack([cos_features, sin_features], dim=-1)
        fourier_features = fourier_features.view(x.size(0), -1)  # [batch_size, 2*num_frequencies]
        
        # Apply learnable weights
        return fourier_features * self.fourier_weight.unsqueeze(0)
    
    def compute_raw_component(self, x: torch.Tensor) -> torch.Tensor:
        """Include raw value and sign information"""
        raw_value = x
        sign_value = torch.sign(x)
        
        raw_features = torch.cat([raw_value, sign_value], dim=-1)
        return raw_features * self.raw_scale.unsqueeze(0)


class NormalizedMathematicalEncoder(nn.Module):
    """
    Mathematical encoder with integrated normalization that preserves properties
    """
    def __init__(self, 
                 base_encoder: ImprovedMathematicalEncoder,
                 target_std: float = 1.0,
                 momentum: float = 0.99,
                 min_std: float = 0.1):
        super().__init__()
        self.base_encoder = base_encoder
        self.normalizer = MathematicallyAwareNormalizer(
            base_encoder, target_std, momentum, min_std
        )
        
    def forward(self, numbers: torch.Tensor) -> torch.Tensor:
        """Forward pass with mathematical property-preserving normalization"""
        return self.normalizer(numbers)
    
    def validate_properties(self, test_numbers: torch.Tensor) -> dict:
        """Validate mathematical properties are preserved"""
        return self.normalizer.validate_mathematical_properties(test_numbers)
    
    @property
    def embedding_dim(self):
        return self.base_encoder.embedding_dim
    
    @property
    def num_frequencies(self):
        return self.base_encoder.num_frequencies
    
    @property
    def max_frequency(self):
        return self.base_encoder.max_frequency
    
    @property
    def include_raw(self):
        return self.base_encoder.include_raw


class SimplifiedTrainer:
    """Simplified trainer focusing on the most critical mathematical properties"""
    
    def __init__(self, encoder: Union[ImprovedMathematicalEncoder, NormalizedMathematicalEncoder], device: str = 'cuda'):
        self.encoder = encoder
        self.device = device
        
        # Check if we're using normalized encoder
        self.is_normalized = isinstance(encoder, NormalizedMathematicalEncoder)
        
        # Simple decoder for invertibility
        self.decoder = nn.Sequential(
            nn.Linear(encoder.embedding_dim, encoder.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(encoder.embedding_dim // 2, encoder.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(encoder.embedding_dim // 4, 1)
        ).to(device)
        
        # Loss weights (simplified)
        self.additivity_weight = 10.0
        self.invertibility_weight = 1.0
        self.distance_weight = 1.0
        
        # Training history
        self.history = []
    
    def compute_loss(self, numbers: torch.Tensor) -> dict:
        """Compute simplified but effective loss"""
        batch_size = len(numbers)
        losses = {}
        total_loss = 0.0
        
        # 1. Additivity Loss (most important)
        if batch_size >= 2:
            add_loss = self._compute_additivity_loss(numbers)
            losses['additivity'] = add_loss
            total_loss += self.additivity_weight * add_loss
        
        # 2. Invertibility Loss
        inv_loss = self._compute_invertibility_loss(numbers)
        losses['invertibility'] = inv_loss
        total_loss += self.invertibility_weight * inv_loss
        
        # 3. Distance Loss (simplified)
        if batch_size >= 3:
            dist_loss = self._compute_distance_loss(numbers)
            losses['distance'] = dist_loss
            total_loss += self.distance_weight * dist_loss
        
        # 4. Property validation for normalized encoder
        if self.is_normalized and batch_size >= 4:
            prop_metrics = self.encoder.validate_properties(numbers)
            losses['property_error'] = prop_metrics['current_error']
            # Add small penalty for property degradation
            if prop_metrics['current_error'] > 0.1:
                total_loss += 0.1 * prop_metrics['current_error']
        
        losses['total'] = total_loss
        return losses
    
    def _compute_additivity_loss(self, numbers: torch.Tensor) -> torch.Tensor:
        """Strong additivity constraint"""
        batch_size = len(numbers)
        
        # Sample pairs more systematically
        n_pairs = min(batch_size // 2, 32)
        
        # Strategy 1: Consecutive pairs
        if batch_size >= 4:
            a1 = numbers[:n_pairs]
            b1 = numbers[n_pairs:2*n_pairs]
            loss1 = self._additivity_loss_for_pairs(a1, b1)
        else:
            loss1 = torch.tensor(0.0, device=numbers.device)
        
        # Strategy 2: Random pairs with replacement
        if batch_size >= 2:
            idx_a = torch.randint(0, batch_size, (n_pairs,), device=numbers.device)
            idx_b = torch.randint(0, batch_size, (n_pairs,), device=numbers.device)
            a2 = numbers[idx_a]
            b2 = numbers[idx_b]
            loss2 = self._additivity_loss_for_pairs(a2, b2)
        else:
            loss2 = torch.tensor(0.0, device=numbers.device)
        
        return (loss1 + loss2) / 2
    
    def _additivity_loss_for_pairs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Core additivity loss computation"""
        # Ensure no normalization interferes
        embed_a = self.encoder(a)
        embed_b = self.encoder(b)
        embed_sum_true = self.encoder(a + b)
        embed_sum_pred = embed_a + embed_b
        
        return F.mse_loss(embed_sum_pred, embed_sum_true)
    
    def _compute_invertibility_loss(self, numbers: torch.Tensor) -> torch.Tensor:
        """Invertibility through decoder"""
        embeddings = self.encoder(numbers)
        decoded = self.decoder(embeddings).squeeze(-1)
        
        # Use relative error to handle different scales
        abs_numbers = torch.abs(numbers.flatten())
        relative_error = torch.abs(decoded - numbers.flatten()) / (abs_numbers + 1e-6)
        
        return relative_error.mean()
    
    def _compute_distance_loss(self, numbers: torch.Tensor) -> torch.Tensor:
        """Simplified distance preservation"""
        embeddings = self.encoder(numbers)
        
        # Sample triplets (a, b, c) and ensure d(a,b) < d(a,c) implies d(E(a),E(b)) < d(E(a),E(c))
        n = len(numbers)
        if n < 3:
            return torch.tensor(0.0, device=numbers.device)
        
        # Random triplets
        n_triplets = min(n, 10)
        losses = []
        
        for _ in range(n_triplets):
            idx = torch.randperm(n)[:3]
            a, b, c = numbers[idx]
            ea, eb, ec = embeddings[idx]
            
            # Numerical distances
            d_ab = torch.abs(a - b)
            d_ac = torch.abs(a - c)
            
            # Embedding distances
            ed_ab = torch.norm(ea - eb)
            ed_ac = torch.norm(ea - ec)
            
            # Ranking loss: if d_ab < d_ac, then ed_ab should be < ed_ac
            if d_ab < d_ac:
                ranking_loss = F.relu(ed_ab - ed_ac + 0.1)  # margin of 0.1
                losses.append(ranking_loss)
            elif d_ac < d_ab:
                ranking_loss = F.relu(ed_ac - ed_ab + 0.1)
                losses.append(ranking_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=numbers.device)
    
    def train_epoch(self, num_batches: int = 50, batch_size: int = 64, lr: float = 1e-3) -> dict:
        """Train for one epoch with focus on stability"""
        # Combined optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-6)
        
        epoch_losses = []
        
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            
            # Generate balanced number batch
            numbers = self._generate_training_batch(batch_size)
            
            # Compute loss
            losses = self.compute_loss(numbers)
            total_loss = losses['total']
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
            optimizer.step()
            
            # Log
            batch_losses = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
            epoch_losses.append(batch_losses)
            
            if batch_idx % 20 == 0:
                log_msg = f"  Batch {batch_idx}: Loss = {batch_losses['total']:.4f}, "
                log_msg += f"Add = {batch_losses.get('additivity', 0):.4f}, "
                log_msg += f"Inv = {batch_losses.get('invertibility', 0):.4f}"
                
                if self.is_normalized and 'property_error' in batch_losses:
                    log_msg += f", Prop = {batch_losses['property_error']:.4f}"
                
                print(log_msg)
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
        
        self.history.append(avg_losses)
        return avg_losses
    
    def _generate_training_batch(self, batch_size: int) -> torch.Tensor:
        """Generate more controlled training batches"""
        numbers = []
        
        # Ensure variety but avoid extreme values that cause numerical issues
        n_each = batch_size // 5
        
        # Small positive numbers
        numbers.extend(torch.rand(n_each) * 10)
        
        # Medium numbers
        numbers.extend(torch.rand(n_each) * 100 + 10)
        
        # Small negative numbers
        numbers.extend(-torch.rand(n_each) * 10)
        
        # Decimals
        numbers.extend(torch.rand(n_each) * 1)
        
        # Mixed
        remaining = batch_size - len(numbers)
        numbers.extend(torch.randn(remaining) * 5)
        
        return torch.tensor(numbers, dtype=torch.float32).to(self.device)
    
    def evaluate(self, test_numbers: torch.Tensor) -> dict:
        """Evaluate mathematical properties"""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            results = {}
            
            # Test additivity with specific examples
            test_pairs = [
                (1.0, 2.0), (3.0, 5.0), (0.1, 0.9), 
                (-2.0, 4.0), (10.0, 0.01)
            ]
            
            additivity_errors = []
            for a_val, b_val in test_pairs:
                a = torch.tensor([a_val], device=self.device)
                b = torch.tensor([b_val], device=self.device)
                
                embed_a = self.encoder(a)
                embed_b = self.encoder(b)
                embed_sum_true = self.encoder(a + b)
                embed_sum_pred = embed_a + embed_b
                
                error = F.mse_loss(embed_sum_pred, embed_sum_true).item()
                additivity_errors.append(error)
            
            results['additivity_mse'] = np.mean(additivity_errors)
            results['additivity_max'] = max(additivity_errors)
            
            # Test invertibility
            embeddings = self.encoder(test_numbers)
            decoded = self.decoder(embeddings).squeeze(-1)
            
            inv_errors = torch.abs(decoded - test_numbers.flatten())
            results['invertibility_mean'] = inv_errors.mean().item()
            results['invertibility_max'] = inv_errors.max().item()
            
            # Test distance preservation with correlation
            if len(test_numbers) >= 5:
                embed_all = self.encoder(test_numbers)
                num_dists = torch.pdist(test_numbers.view(-1, 1))
                embed_dists = torch.pdist(embed_all.view(-1, embed_all.size(-1)))
                
                if len(num_dists) > 1:
                    # Spearman rank correlation
                    num_ranks = torch.argsort(torch.argsort(num_dists)).float()
                    embed_ranks = torch.argsort(torch.argsort(embed_dists)).float()
                    correlation = torch.corrcoef(torch.stack([num_ranks, embed_ranks]))[0,1].item()
                    results['distance_rank_correlation'] = correlation
            
            # Additional metrics for normalized encoder
            if self.is_normalized:
                prop_metrics = self.encoder.validate_properties(test_numbers)
                results['property_preservation_error'] = prop_metrics['current_error']
                results['property_preservation_avg'] = prop_metrics['recent_avg_error']
                results['std_stability'] = prop_metrics['std_stability']
        
        self.encoder.train()
        self.decoder.train()
        return results


class TimestampEncoder(nn.Module):
    """
    Encodes timestamps by decomposing them into cyclical and secular components.
    This approach allows the model to understand both the long-term trend and
    periodic patterns like time of day, week, and year.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Features: 1 secular (trend) + 4 cyclical (sin/cos pairs)
        # Total input features: 1 (scaled timestamp) + 2*4 = 9
        input_dim = 9
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: A tensor of Unix timestamps in seconds.
        """
        # Ensure input is float and has a feature dimension
        x = timestamps.float().view(-1, 1)
        
        components = []
        
        # 1. Secular component (long-term trend)
        # Scaled by seconds in a year for numerical stability
        seconds_in_year = 365.25 * 24 * 60 * 60
        secular_comp = x / seconds_in_year
        components.append(secular_comp)
        
        # 2. Cyclical components
        seconds_in_day = 24 * 60 * 60
        seconds_in_week = 7 * seconds_in_day
        
        # Time of day phase
        day_phase = (x % seconds_in_day) / seconds_in_day
        components.append(torch.sin(2 * math.pi * day_phase))
        components.append(torch.cos(2 * math.pi * day_phase))
        
        # Day of week phase (Unix epoch 1970-01-01 was a Thursday)
        week_phase = ((x / seconds_in_day) + 4) / 7
        components.append(torch.sin(2 * math.pi * week_phase))
        components.append(torch.cos(2 * math.pi * week_phase))
        
        # Day of year phase
        year_phase = (x % seconds_in_year) / seconds_in_year
        components.append(torch.sin(2 * math.pi * year_phase))
        components.append(torch.cos(2 * math.pi * year_phase))
        
        # Month of year (approximated)
        # A more precise implementation would use a calendar-aware library.
        month_phase = year_phase * 12
        components.append(torch.sin(2 * math.pi * month_phase))
        components.append(torch.cos(2 * math.pi * month_phase))
        
        feature_vector = torch.cat(components, dim=-1)
        return self.projection(feature_vector)


class GeoCoordinateEncoder(nn.Module):
    """
    Encodes latitude/longitude coordinates by first projecting them onto a 3D 
    unit sphere. This correctly represents the geometry of coordinates on Earth
    and avoids issues with distance distortion found in raw lat/lon values,
    especially near the poles.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Input is 3D cartesian coordinates (x, y, z)
        self.projection = nn.Sequential(
            nn.Linear(3, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coordinates: A tensor of shape [batch_size, 2] with (latitude, longitude) 
                         pairs in degrees.
        """
        if coordinates.dim() != 2 or coordinates.shape[1] != 2:
            raise ValueError("Input coordinates must be of shape [batch_size, 2]")
            
        lat, lon = coordinates[:, 0], coordinates[:, 1]
        
        # Convert degrees to radians for trigonometric functions
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
        
        # Convert spherical coordinates to 3D Cartesian coordinates
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        
        # Stack to form a [batch_size, 3] tensor
        cartesian_coords = torch.stack([x, y, z], dim=-1)
        
        return self.projection(cartesian_coords)


def load_trained_encoder(model_path: str, device: str = 'cuda', use_normalization: bool = True):
    """Load a previously trained mathematical encoder with optional normalization"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the saved model with weights_only=False for PyTorch 2.6+ compatibility
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Standard loading failed, trying with weights_only=False: {e}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract configuration
    config = checkpoint['encoder_config']
    
    # Create base encoder with the same configuration
    base_encoder = ImprovedMathematicalEncoder(
        embedding_dim=config['embedding_dim'],
        num_frequencies=config['num_frequencies'],
        max_frequency=config['max_frequency'],
        include_raw=config['include_raw'],
        device=device
    ).to(device)
    
    # Load the trained weights
    base_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    # Create normalized encoder if requested
    if use_normalization:
        encoder = NormalizedMathematicalEncoder(base_encoder, target_std=1.0)
        encoder.eval()
        print(f"✓ Loaded normalized encoder from: {model_path}")
    else:
        encoder = base_encoder
        encoder.eval()
        print(f"✓ Loaded raw encoder from: {model_path}")
    
    print(f"  - Embedding dimension: {config['embedding_dim']}")
    print(f"  - Number of frequencies: {config['num_frequencies']}")
    print(f"  - Max frequency: {config['max_frequency']}")
    print(f"  - Include raw: {config['include_raw']}")
    print(f"  - Normalization: {use_normalization}")
    
    # Print final metrics if available
    if 'final_metrics' in checkpoint:
        metrics = checkpoint['final_metrics']
        print(f"  - Final additivity MSE: {metrics['additivity_mse']:.8f}")
        print(f"  - Final invertibility mean: {metrics['invertibility_mean']:.8f}")
        print(f"  - Final distance correlation: {metrics['distance_rank_correlation']:.8f}")
    
    return encoder


def train_improved_encoder(use_normalization: bool = True):
    """Train the improved encoder with better stability and optional normalization"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create base encoder with better design
    base_encoder = ImprovedMathematicalEncoder(
        embedding_dim=1024,
        num_frequencies=20,
        max_frequency=50.0,
        include_raw=True,
        device=device
    ).to(device)
    
    # Create normalized encoder if requested
    if use_normalization:
        encoder = NormalizedMathematicalEncoder(base_encoder, target_std=1.0)
        print("Training normalized mathematical encoder...")
    else:
        encoder = base_encoder
        print("Training raw mathematical encoder...")
    
    # Create trainer
    trainer = SimplifiedTrainer(encoder, device)
    
    print(f"Architecture: {encoder.embedding_dim}D embeddings, {encoder.num_frequencies} frequencies")
    print(f"Normalization: {use_normalization}")
    
    # Training loop with better monitoring
    num_epochs = 30
    best_additivity = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Adjust learning rate
        lr = 1e-3 if epoch < 15 else 1e-4 if epoch < 25 else 5e-4
        
        # Train epoch
        losses = trainer.train_epoch(num_batches=80, batch_size=128, lr=lr)
        
        print(f"  Avg Losses - Total: {losses['total']:.4f}, "
              f"Additivity: {losses.get('additivity', 0):.4f}, "
              f"Invertibility: {losses.get('invertibility', 0):.4f}")
        
        if use_normalization and 'property_error' in losses:
            print(f"  Property Preservation Error: {losses['property_error']:.6f}")
        
        # Evaluate every few epochs
        if (epoch + 1) % 5 == 0:
            test_numbers = trainer._generate_training_batch(100)
            results = trainer.evaluate(test_numbers)
            
            print(f"  Evaluation:")
            print(f"    Additivity MSE: {results.get('additivity_mse', 'N/A'):.6f}")
            print(f"    Additivity Max: {results.get('additivity_max', 'N/A'):.6f}")
            print(f"    Invertibility Mean: {results.get('invertibility_mean', 'N/A'):.6f}")
            print(f"    Distance Correlation: {results.get('distance_rank_correlation', 'N/A'):.6f}")
            
            if use_normalization:
                print(f"    Property Error: {results.get('property_preservation_error', 'N/A'):.6f}")
                print(f"    STD Stability: {results.get('std_stability', 'N/A'):.6f}")
            
            # Track best
            current_additivity = results.get('additivity_mse', float('inf'))
            if current_additivity < best_additivity:
                best_additivity = current_additivity
                print(f"    ✓ Best additivity so far: {best_additivity:.6f}")
    
    # Final comprehensive test
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Test with specific mathematical examples
    test_cases = torch.tensor([
        0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0,
        -1.0, -2.0, -5.0, -10.0,
        0.01, 0.001, 100.0, 1000.0, 123.456, 111.111, 123456.789, 111111.111
    ]).to(device)
    
    results = trainer.evaluate(test_cases)
    
    print("Mathematical Property Performance:")
    for key, value in results.items():
        print(f"  {key}: {value:.8f}")
    
    # Manual additivity tests
    print("\nManual Additivity Tests:")
    test_pairs = [(1.0, 2.0), (5.0, 3.0), (0.1, 0.9), (-2.0, 4.0)]
    
    for a_val, b_val in test_pairs:
        a = torch.tensor([a_val], device=device)
        b = torch.tensor([b_val], device=device)
        sum_val = a_val + b_val
        
        embed_a = encoder(a)
        embed_b = encoder(b)
        embed_sum_true = encoder(torch.tensor([sum_val], device=device))
        embed_sum_pred = embed_a + embed_b
        
        error = F.mse_loss(embed_sum_pred, embed_sum_true).item()
        print(f"  E({a_val}) + E({b_val}) ≈ E({sum_val}): MSE = {error:.8f}")
    
    # Save the trained model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs("number_encoders", exist_ok=True)
    
    # Save encoder weights
    model_suffix = "_normalized" if use_normalization else "_raw"
    encoder_path = f"number_encoders/mathematical_encoder_1024d{model_suffix}.pth"
    
    # Save base encoder state dict for compatibility
    base_state_dict = base_encoder.state_dict()
    
    torch.save({
        'encoder_state_dict': base_state_dict,
        'encoder_config': {
            'embedding_dim': encoder.embedding_dim,
            'num_frequencies': encoder.num_frequencies,
            'max_frequency': encoder.max_frequency,
            'include_raw': encoder.include_raw
        },
        'normalization_config': {
            'use_normalization': use_normalization,
            'target_std': 1.0 if use_normalization else None,
            'momentum': 0.99 if use_normalization else None,
            'min_std': 0.1 if use_normalization else None
        },
        'training_results': results,
        'final_metrics': {
            'additivity_mse': results.get('additivity_mse', 0),
            'additivity_max': results.get('additivity_max', 0),
            'invertibility_mean': results.get('invertibility_mean', 0),
            'invertibility_max': results.get('invertibility_max', 0),
            'distance_rank_correlation': results.get('distance_rank_correlation', 0)
        }
    }, encoder_path)
    
    print(f"✓ Encoder saved to: {encoder_path}")
    print(f"  - Embedding dimension: {encoder.embedding_dim}")
    print(f"  - Number of frequencies: {encoder.num_frequencies}")
    print(f"  - Max frequency: {encoder.max_frequency}")
    print(f"  - Include raw: {encoder.include_raw}")
    print(f"  - Normalization: {use_normalization}")
    print(f"  - Final additivity MSE: {results.get('additivity_mse', 0):.8f}")
    
    return encoder, trainer




if __name__ == "__main__":
    # Train with normalization by default (better for multi-modal alignment)
    print("Training normalized mathematical encoder...")
    trained_encoder, trainer = train_improved_encoder(use_normalization=True)
    