"""
File: scripts/inference.py
Unit Test: tests/test_scripts.py::TestInference

Inference pipeline for Stable Diffusion.
Supports text-to-image and image-to-image generation with classifier-free guidance.
"""
import os
from typing import List, Optional, Union, Dict
import torch
from PIL import Image
from tqdm import tqdm

# Assume these are defined elsewhere in the project
# from tokenizer import Tokenizer
# from model_loader import load_clip, load_encoder, load_decoder, load_diffusion
from src.sampling.k_lms import KLMSSampler
from src.sampling.k_euler import KEulerSampler
from src.sampling.k_euler_ancestral import KEulerAncestralSampler
from scripts.utils import rescale, move_channel, get_time_embedding, prepare_image_tensor


def generate(
    prompts: Union[str, List[str]],
    unconditional_prompts: Optional[Union[str, List[str]]] = None,
    input_images: Optional[List[Union[str, Image.Image]]] = None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    sampler: str = "k_lms",
    n_inference_steps: int = 50,
    models: Optional[Dict] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    idle_device: Optional[str] = None,
    output_dir: str = "outputs",
    save_images: bool = True
) -> List[Image.Image]:
    """
    Generate images using Stable Diffusion.
    
    Supports two modes:
    1. Text-to-Image: Generate from text prompts only
    2. Image-to-Image: Transform input images guided by text
    
    Args:
        prompts: Text prompt(s) to guide generation.
            Can be single string or list of strings.
        
        unconditional_prompts: Negative prompts for CFG.
            Defaults to [""] * len(prompts).
            Ignored when do_cfg=False.
        
        input_images: Starting images for img2img.
            List of PIL Images or file paths.
            If None, performs text-to-image generation.
        
        strength: Denoising strength for img2img (0-1).
            1.0 = maximum transformation (almost text-to-image)
            0.0 = minimal transformation (nearly original image)
            Only used when input_images is provided.
        
        do_cfg: Enable classifier-free guidance.
            True = use both conditional and unconditional predictions
            False = only use conditional prediction
        
        cfg_scale: Guidance scale (1.0-20.0).
            Higher values = stronger adherence to prompt
            Lower values = more creative freedom
            Typical range: 7-15
            Ignored when do_cfg=False.
        
        height: Generated image height in pixels.
            Must be multiple of 8.
            Ignored when input_images provided.
        
        width: Generated image width in pixels.
            Must be multiple of 8.
            Ignored when input_images provided.
        
        sampler: Sampling algorithm.
            Options: "k_euler", "k_euler_ancestral", "k_lms"
            k_lms = best quality (default)
            k_euler = fastest
            k_euler_ancestral = good balance
        
        n_inference_steps: Number of denoising steps.
            More steps = higher quality but slower.
            Typical range: 20-100
        
        models: Pre-loaded models dict.
            Keys: "clip", "encoder", "decoder", "diffusion"
            If None, models loaded dynamically (slower).
        
        seed: Random seed for reproducibility.
            If None, uses random seed.
        
        device: Device for inference ("cuda" or "cpu").
            If None, auto-detects CUDA availability.
        
        idle_device: Device to move idle models.
            Useful for memory management.
            If None, keeps models on main device.
        
        output_dir: Directory to save generated images.
            Only used if save_images=True.
        
        save_images: Whether to save images to disk.
    
    Returns:
        List of PIL Images
    
    Raises:
        ValueError: If input validation fails
    
    Example:
        >>> # Text-to-image
        >>> images = generate(
        ...     prompts="A beautiful sunset over mountains",
        ...     height=512,
        ...     width=512,
        ...     n_inference_steps=50
        ... )
        
        >>> # Image-to-image
        >>> images = generate(
        ...     prompts="Turn this into a watercolor painting",
        ...     input_images=["photo.jpg"],
        ...     strength=0.7
        ... )
    """
    
    # === Input Validation ===
    
    # Convert single prompt to list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError("Prompts must be a non-empty list, tuple, or string")
    
    # Handle unconditional prompts
    if unconditional_prompts is None:
        unconditional_prompts = [""] * len(prompts)
    elif isinstance(unconditional_prompts, str):
        unconditional_prompts = [unconditional_prompts] * len(prompts)
    
    if not isinstance(unconditional_prompts, (list, tuple)):
        raise ValueError("Unconditional prompts must be list, tuple, or string")
    
    # Validate input images
    if input_images is not None:
        if not isinstance(input_images, (list, tuple)):
            raise ValueError("input_images must be a list or tuple")
        if len(prompts) != len(input_images):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match "
                f"number of input images ({len(input_images)})"
            )
    
    # Validate strength
    if not 0 <= strength <= 1:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    
    # Validate dimensions
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"height and width must be multiples of 8, got {height}x{width}")
    
    # === Device Setup ===
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Function to move models to idle device
    if idle_device:
        to_idle = lambda x: x.to(idle_device)
    else:
        to_idle = lambda x: x
    
    # === Random Seed ===
    
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
        print(f"Using seed: {seed}")
    
    # === Models (placeholder - you'll need to import your model_loader) ===
    
    if models is None:
        models = {}
    
    # Load CLIP for text encoding
    # clip = models.get('clip') or load_clip(device)
    # clip.to(device)
    # NOTE: Replace above with your actual model loading
    
    with torch.no_grad():  # Inference mode
        
        # === Text Encoding ===
        
        # tokenizer = Tokenizer()
        # 
        # if do_cfg:
        #     # Encode both conditional and unconditional prompts
        #     conditional_tokens = tokenizer.encode_batch(prompts)
        #     conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device=device)
        #     conditional_context = clip(conditional_tokens)
        #     
        #     unconditional_tokens = tokenizer.encode_batch(unconditional_prompts)
        #     unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device=device)
        #     unconditional_context = clip(unconditional_tokens)
        #     
        #     # Concatenate for batched processing
        #     context = torch.cat([conditional_context, unconditional_context])
        # else:
        #     # Only conditional
        #     tokens = tokenizer.encode_batch(prompts)
        #     tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        #     context = clip(tokens)
        # 
        # to_idle(clip)
        # del tokenizer, clip
        
        # PLACEHOLDER: Create dummy context for demonstration
        context = torch.randn(len(prompts) * (2 if do_cfg else 1), 77, 768, device=device)
        dtype = torch.float32
        
        # === Sampler Setup ===
        
        if sampler == "k_euler":
            sampler_obj = KEulerSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler_ancestral":
            sampler_obj = KEulerAncestralSampler(
                n_inference_steps=n_inference_steps,
                generator=generator
            )
        elif sampler == "k_lms":
            sampler_obj = KLMSSampler(n_inference_steps=n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        # === Latent Initialization ===
        
        noise_shape = (len(prompts), 4, height // 8, width // 8)
        
        if input_images:
            # Image-to-Image: Encode input images
            
            # encoder = models.get('encoder') or load_encoder(device)
            # encoder.to(device)
            
            # Process input images
            processed_images = []
            for img in input_images:
                img_tensor = prepare_image_tensor(img, width, height, dtype, device)
                processed_images.append(img_tensor)
            
            images_tensor = torch.stack(processed_images)  # (B, C, H, W)
            
            # Encode to latent space
            # encoder_noise = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            # latents = encoder(images_tensor, encoder_noise)
            
            # PLACEHOLDER
            latents = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            
            # Add noise for denoising
            latents_noise = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            sampler_obj.set_strength(strength=strength)
            latents = latents + latents_noise * sampler_obj.initial_scale
            
            # to_idle(encoder)
            # del encoder, processed_images, images_tensor, latents_noise
        
        else:
            # Text-to-Image: Start from pure noise
            latents = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            latents *= sampler_obj.initial_scale
        
        # === Denoising Loop ===
        
        # diffusion = models.get("diffusion") or load_diffusion(device)
        # diffusion.to(device)
        
        print(f"Generating {len(prompts)} image(s) with {n_inference_steps} steps...")
        
        timesteps = tqdm(sampler_obj.timesteps, desc="Denoising")
        for i, timestep in enumerate(timesteps):
            # Get timestep embedding
            time_embedding = get_time_embedding(int(timestep), dtype=dtype).to(device)
            time_embedding = time_embedding.unsqueeze(0).repeat(len(prompts), 1)
            
            # Scale input for sampler
            input_latents = latents * sampler_obj.get_input_scale()
            
            # Classifier-free guidance: duplicate latents
            if do_cfg:
                input_latents = input_latents.repeat(2, 1, 1, 1)
            
            # Predict noise
            # output = diffusion(input_latents, context, time_embedding)
            
            # PLACEHOLDER
            output = torch.randn_like(input_latents)
            
            # Apply classifier-free guidance
            if do_cfg:
                output_cond, output_uncond = output.chunk(2)
                output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # Denoise one step
            latents = sampler_obj.step(latents, output)
        
        # to_idle(diffusion)
        # del diffusion
        
        # === Decode Latents ===
        
        # decoder = models.get("decoder") or load_decoder(device)
        # decoder.to(device)
        # images = decoder(latents)
        # to_idle(decoder)
        # del decoder
        
        # PLACEHOLDER
        images = torch.randn(len(prompts), 3, height, width, device=device)
        
        # === Post-processing ===
        
        # Rescale to [0, 255]
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        
        # Convert to (B, H, W, C) for PIL
        images = move_channel(images, to="last")
        
        # To numpy
        images = images.cpu().to(torch.uint8).numpy()
        
        # Convert to PIL Images
        pil_images = [Image.fromarray(img) for img in images]
        
        # Save if requested
        if save_images:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(pil_images):
                filename = f"generated_{seed}_{i:02d}.png" if seed else f"generated_{i:02d}.png"
                img.save(os.path.join(output_dir, filename))
            print(f"Saved images to {output_dir}/")
        
        return pil_images


if __name__ == "__main__":
    # Example usage
    images = generate(
        prompts="A beautiful mountain landscape at sunset",
        height=512,
        width=512,
        n_inference_steps=50,
        seed=42
    )
    print(f"Generated {len(images)} image(s)")