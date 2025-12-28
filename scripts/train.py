"""
File: scripts/train.py

Training script for Stable Diffusion / DDPM.
Implements complete training loop with:
- Data loading
- Loss computation (noise prediction MSE)
- Gradient accumulation
- Checkpointing
- Validation
- Logging
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

# Import modules (adjust paths as needed)
from src.models.diffusion import Diffusion
from src.models.encoder import Encoder
from src.models.clip import CLIP
from src.schedulers.ddpm_scheduler import DDPMScheduler
from .tokenizer import Tokenizer
from .utils import get_time_embedding


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Stable Diffusion model")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Model
    parser.add_argument("--latent_channels", type=int, default=4, help="Latent channels")
    parser.add_argument("--image_size", type=int, default=512, help="Image resolution")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    
    # Diffusion
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    # Logging
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    # Compute
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision (fp16)")
    
    return parser.parse_args()


class DiffusionTrainer:
    """
    Trainer for diffusion models.
    
    Implements DDPM training:
    1. Sample timestep t ~ Uniform(0, T-1)
    2. Add noise: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    3. Predict noise: ε_θ(x_t, c, t)
    4. Loss: MSE(ε_θ, ε)
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize models
        self.setup_models()
        
        # Initialize scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_timesteps,
            beta_schedule=args.beta_schedule,
            device=self.device
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs,
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Resume from checkpoint if specified
        if args.resume_from:
            self.load_checkpoint(args.resume_from)
    
    def setup_models(self):
        """Initialize models."""
        # Main diffusion model (U-Net)
        self.model = Diffusion().to(self.device)
        
        # VAE encoder (frozen during training)
        self.vae_encoder = Encoder().to(self.device)
        self.vae_encoder.eval()
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        
        # CLIP text encoder (frozen during training)
        self.text_encoder = CLIP().to(self.device)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Tokenizer
        self.tokenizer = Tokenizer()
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_loss(self, batch):
        """
        Compute diffusion training loss.
        
        Args:
            batch: Dict with 'images' and 'captions'
            
        Returns:
            Loss value
        """
        images = batch['images'].to(self.device)  # (B, 3, H, W)
        captions = batch['captions']
        
        batch_size = images.shape[0]
        
        # Encode images to latents
        with torch.no_grad():
            noise_enc = torch.randn(batch_size, 4, 64, 64, device=self.device)
            latents = self.vae_encoder(images, noise_enc)  # (B, 4, 64, 64)
        
        # Encode text prompts
        with torch.no_grad():
            tokens = self.tokenizer.encode_batch(captions)
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            text_embeddings = self.text_encoder(tokens)  # (B, 77, 768)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.args.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Add noise to latents (forward diffusion)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get timestep embeddings
        time_emb = torch.stack([
            get_time_embedding(t.item(), dtype=latents.dtype)
            for t in timesteps
        ]).to(self.device)
        
        # Predict noise
        noise_pred = self.model(noisy_latents, text_embeddings, time_emb)
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise, reduction='mean')
        
        return loss
    
    def train_step(self, batch):
        """Single training step."""
        # Compute loss
        if self.scaler:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(batch)
            
            # Backward with gradient scaling
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.args.grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # Standard training
            loss = self.compute_loss(batch)
            loss = loss / self.args.grad_accum_steps
            loss.backward()
            
            if (self.global_step + 1) % self.args.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return loss.item() * self.args.grad_accum_steps
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss = self.train_step(batch)
            epoch_loss += loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })
            
            # Logging
            if self.global_step % self.args.log_every == 0:
                self.log_metrics({
                    'train/loss': loss,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.epoch,
                    'train/step': self.global_step
                })
            
            # Save checkpoint
            if self.global_step % self.args.save_every == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        return epoch_loss / len(dataloader)
    
    def train(self, train_dataloader):
        """Main training loop."""
        print(f"Starting training for {self.args.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Gradient accumulation steps: {self.args.grad_accum_steps}")
        
        for epoch in range(self.epoch, self.args.num_epochs):
            self.epoch = epoch
            
            # Train one epoch
            avg_loss = self.train_epoch(train_dataloader)
            
            print(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        print("Training complete!")
        self.save_checkpoint("final_model.pt")
    
    def save_checkpoint(self, filename=None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint_path = os.path.join(self.args.output_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'args': vars(self.args)
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
    
    def log_metrics(self, metrics):
        """Log training metrics."""
        if self.args.use_wandb:
            try:
                wandb.log(metrics, step=self.global_step)
            except ImportError:
                pass

class DummyDataset(Dataset):
    """
    Create a dummy dataset
    """
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def __len__(self):
        return 1000
    
    def __getitem__(self):
        return {
            'images': torch.randn(3, self.image_size, self.image_size),
            'captions': "A beautiful landscape"
        }


def create_dummy_dataloader(args):
    """
    Create dummy dataloader for testing.
    Replace with actual dataset implementation.
    """
    
    dataset = DummyDataset(args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return dataloader


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize W&B if requested
    if args.use_wandb:
        try:
            wandb.init(project="stable-diffusion", config=vars(args))
        except ImportError:
            print("W&B not installed, skipping logging")
            args.use_wandb = False
    
    # Create dataloader
    # TODO: Replace with actual dataset
    train_dataloader = create_dummy_dataloader(args)
    
    # Initialize trainer
    trainer = DiffusionTrainer(args)
    
    # Train
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()