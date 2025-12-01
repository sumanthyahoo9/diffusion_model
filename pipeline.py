"""
This script gives the main pipeline to run the training and the inference processes.
"""
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import util
import model_loader
from tokenizer import Tokenizer
from k_lms import KLMSSampler
from k_euler import KEulerSampler
from k_euler_ancestral import KEulerAncestralSampler


def generate(
		prompts,
		unconditional_prompts=None,
		input_images=None,
		strength=0.8,
		do_cfg=True,
		cfg_scale=7.5,
		height=512,
		width=512,
		sampler="k_lms",
		n_inference_steps=50,
		models=None,
		seed=None,
		device=None,
		idle_device=None
	):
	"""
	Function called to run the pipeline for image generation
	Param prompts: The prompts to guide the image generation
	Param unconditional_prompts: **Optional**, defaults to [""] * len(prompts). 
		Ignored when not using guidance, i.e. do_cfg = False
	Param input_images: Input images to serve as the starting point for image generation
	Param strength: Conceptually, indicates the level to which the input_images should be transformed and numerically,
		it indicates the level of the noise to be added and is a value between 0 and 1.
		The number of de-noising steps depends on the amount of noise added initially and when the STRENGTH IS 1, added
		noise is MAXIMUM and the de-noising process runs for the FULL NUMBER of iterations specified
	Param do_cfg: Enable classifier-free guidance, default is True
	Param cfg_scale: Guidance scale of classifier-free guidance. Ignored when disabled, i.e. do_cfg = False. 
		Higher guidance scale encourages to generate images that are closely linked to the text 'prompt', usually at the 
		EXPENSE of LOWER IMAGE QUALITY.
	Param height: Height of the generated image in pixels. Ignored when input images are given.
	Param width: Width of the generated image in pixels. Ignored when input images are given.
	Param sampler: A sampler used to DENOISE the ENCODED image latents. 
		Has to be one of "k_lms" and "k_euler"
	Param n_inference_steps: The number of de-noising steps and more de-noising steps leads to HIGHER QUALITY IMAGE
		at the EXPENSE of SLOWER INFERENCE.
	Param models: Preloaded models and no models are provided, they will be dynamically loaded
	Param seed: Seed to make deterministic generation
	Param device: Device on which the generation occurs
	Param idle_device: Device where the models not in use are moved to. 
	"""
	if models is None:
		models = {}
	with torch.no_grad():
		# Run in inference mode
		if not isinstance(prompts, (list, tuple)) or not prompts:
			raise ValueError("Prompts must be a non-empty list or tuple")
		if unconditional_prompts and not isinstance(unconditional_prompts, (list, tuple)):
			raise ValueError("Unconditional Prompts must be a non-empty list or a tuple")
		unconditional_prompts = unconditional_prompts or [""] * len(prompts)

		if input_images and not isinstance(unconditional_prompts, (list, tuple)):
			raise ValueError(f"{input_images} must be a non-empty list or tuple if provided")
		if input_images and len(prompts) != len(input_images):
			raise ValueError(f"Length of input_images {input_images} must be the same as the length of prompts {prompts}")
		if not 0 <= strength <= 1:
			raise ValueError(f"{strength} must be between 0 and 1")

		if height % 8 or width % 8:
			raise ValueError("Height and Width must be multiples of 8")

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if idle_device:
			to_idle = lambda x: x.to(idle_device)
		else:
			to_idle = lambda x: x

		generator = torch.Generator(device=device)
		if seed is None:
			generator.seed()
		else:
			generator.manual_seed(seed)

		tokenizer = Tokenizer()
		clip = models.get('clip') or model_loader.load_clip(device)
		clip.to(device)

		# Use the datatype (dtype) of the model weights as the default dtype
		dtype = clip.embedding.position_value.dtype
		if do_cfg:
			conditional_tokens = tokenizer.encode_batch(prompts)
			conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device=device)
			conditional_context = clip(conditional_tokens)
			unconditional_tokens = tokenizer.encode_batch(unconditional_prompts)
			unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device=device)
			unconditional_context = clip(unconditional_tokens)
			context = torch.cat([conditional_context, unconditional_context])
		else:
			tokens = tokenizer.encode_batch(prompts)
			tokens = torch.tensor(tokens, dtype=torch.long, device=device)
			context = clip(tokens)
		to_idle(clip)
		del tokenizer, clip

		if sampler == "k_euler":
			sampler = KEulerSampler(n_inference_steps=n_inference_steps)
		elif sampler == "k_euler_ancestral":
			sampler = KEulerAncestralSampler(n_inference_steps=n_inference_steps)
		elif sampler == "l_lms":
			sampler = KLMSSampler(n_inference_steps=n_inference_steps)

		else:
			raise ValueError("Unknown sampler value")

		noise_shape = (len(prompts), 4, height//8, width//8)

		if input_images:
			encoder = models.get('encoder') or model_loader.load_encoder(device)
			encoder.to(device)
			processed_input_images = []
			for input_image in input_images:
				if isinstance(input_image, str):
					input_image = Image.open(input_image)
				input_image = input_image.resize((width, height))
				input_image = np.array(input_image)
				input_image = torch.tensor(input_image, dtype=dtype)
				input_image = util.rescale(input_image, (0, 255), (-1, 1))
				processed_input_images.append(input_image)
			input_images_tensor = torch.stack(processed_input_images).to(device)
			input_images_tensor = util.move_channel(input_images_tensor, to="first")

			_, _, height, width = input_images_tensor.shape
			noise_shape = (len(prompts), height//8, width//8)

			# Noise for the encoder
			encoder_noise = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
			latents = encoder(input_images_tensor, encoder_noise)

			latents_noise = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
			sampler.set_strength(strength=strength)
			latents += latents_noise * sampler.initial_scale

			to_idle(encoder)
			del encoder, processed_input_images, input_images_tensor, latents_noise
		else:
			latents = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
			latents *= sampler.initial_scale

		diffusion = models.get("diffusion") or model_loader.load_diffusion(device)
		diffusion.to(device)

		time_steps = tqdm(sampler.timesteps)
		for i, timestep in enumerate(time_steps):
			time_embedding = util.get_time_embedding(timestep, dtype).to(device)
			input_latents = latents * sampler.get_input_scale()
			if do_cfg:
				input_latents = input_latents.repeat(2, 1, 1, 1)
			output = diffusion(input_latents, context, time_embedding)
			if do_cfg:
				output_cond, output_uncond = output.chunk(2) 
				output = cfg_scale * (output_cond - output_uncond) + output_uncond
			latents = sampler.step(latents, output)

		to_idle(diffusion)
		del diffusion

		decoder = models.get("decoder") or model_loader.load_decoder(device)
		decoder.to(device)
		images = decoder(latents)
		to_idle(decoder)  # Move the model not in use to the idle device
		del decoder  # Delete the Decoder

		images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
		images = util.move_channel(images, to="last")
		images = images.to('cpu', torch.uint8).numpy()

		return [Image.fromarray(image) for image in images]
