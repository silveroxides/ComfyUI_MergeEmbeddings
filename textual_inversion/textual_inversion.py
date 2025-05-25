import os
# from collections import namedtuple # For TextualInversionTemplate, to be removed
# from contextlib import closing # Used in train_embedding, to be removed

import torch
# import tqdm # Used in train_embedding, to be removed
# import html # Used in train_embedding, to be removed
# import datetime # Used in train_embedding, to be removed
# import csv # Used in write_loss, to be removed
import safetensors.torch

import numpy as np # Used in tensorboard_add_image, to be removed
from PIL import Image, PngImagePlugin # PngImagePlugin used in train_embedding for saving image with embedding. Keep PIL.Image for read_embedding_from_image

# Most of these 'modules' imports are A1111 specific.
# Keep 'devices' for now for create_embedding_from_data and load_from_file, but mark with TODO.
# Keep 'hashes' for create_embedding_from_data, but this was also marked for refactoring.
from modules import devices, sd_hijack, sd_models, errors, hashes # Removed: shared, images, sd_samplers, sd_hijack_checkpoint, cache
# import modules.textual_inversion.dataset # Used in train_embedding, to be removed
# from modules.textual_inversion.learn_schedule import LearnRateScheduler # Used in train_embedding, to be removed

from modules.textual_inversion.image_embedding import embedding_to_b64, embedding_from_b64, extract_image_data_embed # Removed: insert_image_data_embed, caption_image_overlay (used in training previews)
# from modules.textual_inversion.saving_settings import save_settings_to_file # Used in train_embedding, to be removed


# TextualInversionTemplate = namedtuple("TextualInversionTemplate", ["name", "path"]) # Training specific
# textual_inversion_templates = {} # Training specific

# def list_textual_inversion_templates(): # Training specific
#     textual_inversion_templates.clear()
# 
#     # shared.cmd_opts.textual_inversion_templates_dir is A1111 specific
#     for root, _, fns in os.walk(shared.cmd_opts.textual_inversion_templates_dir):
#         for fn in fns:
#             path = os.path.join(root, fn)
# 
#             textual_inversion_templates[fn] = TextualInversionTemplate(fn, path)
# 
#     return textual_inversion_templates


class Embedding:
    """
    Represents a textual inversion embedding.

    Attributes:
        vec (torch.Tensor): The embedding vector.
        name (str): The name of the embedding.
        step (int, optional): The training step at which the embedding was saved. Defaults to None.
        shape (int, optional): The dimensionality of the embedding vector. Defaults to None.
        vectors (int): The number of vectors in the embedding. Defaults to 0.
        cached_checksum (str, optional): The cached checksum of the embedding. Defaults to None.
        sd_checkpoint (str, optional): The hash of the Stable Diffusion checkpoint used for training. Defaults to None.
        sd_checkpoint_name (str, optional): The name of the Stable Diffusion checkpoint used for training. Defaults to None.
        optimizer_state_dict (dict, optional): The state dictionary of the optimizer used for training. Defaults to None.
        filename (str, optional): The filename of the embedding. Defaults to None.
        hash (str, optional): The full hash of the embedding file. Defaults to None.
        shorthash (str, optional): A short version of the embedding file hash. Defaults to None.
    """
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None # Dimensionality of the embedding vector
        self.vectors = 0 # Number of vectors in the embedding
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None
        self.hash = None
        self.shorthash = None

    def save(self, filename, save_optimizer_state=False):
        """
        Saves the embedding to a file.

        Args:
            filename (str): The path to save the embedding file.
            save_optimizer_state (bool, optional): Whether to save the optimizer state. Defaults to False.
        """
        embedding_data = {
            "string_to_token": {"*": 265}, # Placeholder token
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

        # Save optimizer state if requested and available
        if save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            torch.save(optimizer_saved_dict, f"{filename}.optim")

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum

    def set_hash(self, v):
        self.hash = v
        self.shorthash = self.hash[0:12]


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        """Checks if the directory has been modified since the last check."""
        if not os.path.isdir(self.path):
            return False

        mt = os.path.getmtime(self.path)
        if self.mtime is None or mt > self.mtime: # Check if mtime is newer
            return True

    def update(self):
        """Updates the mtime of the directory."""
        if not os.path.isdir(self.path):
            return

        self.mtime = os.path.getmtime(self.path)


class EmbeddingDatabase:
    """
    Manages and provides access to textual inversion embeddings.

    Attributes:
        ids_lookup (dict): A dictionary mapping token IDs to a list of (token_ids, embedding) tuples.
                           Used for efficient lookup of embeddings based on token IDs.
        word_embeddings (dict): A dictionary mapping embedding names to Embedding objects.
        skipped_embeddings (dict): A dictionary mapping names of skipped embeddings to Embedding objects.
                                   Embeddings are skipped if their shape doesn't match the expected shape.
        expected_shape (int): The expected dimensionality of the embedding vectors.
                              Set based on the loaded model. -1 if not yet determined.
        embedding_dirs (dict): A dictionary mapping directory paths to DirWithTextualInversionEmbeddings objects.
                               Tracks directories containing embedding files.
        previously_displayed_embeddings (tuple): A tuple containing two tuples:
                                                 (names of loaded embeddings, names of skipped embeddings).
                                                 Used to detect changes for printing loaded/skipped embeddings.
        image_embedding_cache (dict): A simple dictionary for caching embedding data read from images.
                                      Keys are file paths, values are dicts with 'data', 'name', 'mtime'.
    """
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {} # Loaded embeddings
        self.skipped_embeddings = {} # Embeddings that were skipped due to shape mismatch
        self.expected_shape = -1 # Expected shape of embedding vectors, derived from the model
        self.embedding_dirs = {} # Directories to scan for embeddings
        self.previously_displayed_embeddings = () # For logging changes in loaded embeddings
        self.image_embedding_cache = {} # Simple cache for image embedding data

    def add_embedding_dir(self, path):
        """Adds a directory to scan for embeddings."""
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        """Clears all registered embedding directories."""
        self.embedding_dirs.clear()

    def register_embedding(self, embedding, model):
        """
        Registers an embedding with the database.

        Args:
            embedding (Embedding): The embedding object to register.
            model: The Stable Diffusion model object (specifically its cond_stage_model for tokenization).
        """
        return self.register_embedding_by_name(embedding, model, embedding.name)

    def register_embedding_by_name(self, embedding, model, name):
        """
        Registers an embedding with the database using a specific name.

        Args:
            embedding (Embedding): The embedding object to register.
            model: The Stable Diffusion model object (specifically its cond_stage_model for tokenization).
            name (str): The name to register the embedding under.
        """
        # Tokenize the embedding name to get token IDs
        ids = model.cond_stage_model.tokenize([name])[0]
        first_id = ids[0] # Get the first token ID
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = [] # Initialize list if token ID not seen before

        # Remove any existing embedding with the same name for this token ID
        if name in self.word_embeddings:
            lookup = [x for x in self.ids_lookup[first_id] if x[1].name != name]
        else:
            lookup = self.ids_lookup[first_id]

        if embedding is not None:
            lookup.append((ids, embedding)) # Add the new embedding (token_ids, embedding_obj)

        # Sort by the length of token_ids in descending order to prioritize longer matches
        self.ids_lookup[first_id] = sorted(lookup, key=lambda x: len(x[0]), reverse=True)

        if embedding is None:
            # Unregister embedding if embedding is None
            if name in self.word_embeddings:
                del self.word_embeddings[name]
            if not self.ids_lookup[first_id]: # Remove entry if list becomes empty
                del self.ids_lookup[first_id]
            return None

        self.word_embeddings[name] = embedding
        return embedding

    def get_expected_shape(self, clip_model):
        """
        Determines the expected shape of embedding vectors based on the provided CLIP model.

        Args:
            clip_model: The CLIP model (or its text encoder part) which has an
                        `encode_embedding_init_text` method or equivalent.

        Returns:
            int: The dimensionality of the embedding vectors.
        """
        # The device should ideally be handled by the caller or ComfyUI's environment.
        # For now, we assume the clip_model is already on the correct device.
        # devices.torch_npu_set_device() # Removed A1111-specific device call
        vec = clip_model.encode_embedding_init_text(",", 1) # Get a dummy vector
        return vec.shape[1] # Return its dimensionality

    def read_embedding_from_image(self, path, name, cache_image_embeddings=True):
        """
        Reads embedding data from an image file.

        Args:
            path (str): Path to the image file.
            name (str): Default name for the embedding if not found in image.
            cache_image_embeddings (bool): Whether to cache the read embedding data.

        Returns:
            tuple: (embedding_data, name) or (None, None) if an error occurs.
        """
        try:
            ondisk_mtime = os.path.getmtime(path)
            if cache_image_embeddings and (cache_entry := self.image_embedding_cache.get(path)):
                if ondisk_mtime == cache_entry.get('mtime', 0):
                    # Use cached data if file modification time matches
                    return cache_entry.get('data', None), cache_entry.get('name', None)

            embed_image = Image.open(path)
            data = None
            if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                # Extract embedding from PNG info text
                data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                name = data.get('name', name) # Use name from data if available
            elif extracted_data := extract_image_data_embed(embed_image): # Check other extraction methods
                data = extracted_data
                name = data.get('name', name)

            if cache_image_embeddings and (data is not None or not self.image_embedding_cache.get(path)): # Cache if data found or not previously cached
                self.image_embedding_cache[path] = {'data': data, 'name': None if data is None else name, 'mtime': ondisk_mtime}
            return data, name
        except Exception as e:
            print(f"Error loading embedding from image {path}: {e}") # Use print instead of errors.report
        return None, None

    def load_from_file(self, path, filename, sd_model):
        """
        Loads an embedding from a file.

        Args:
            path (str): The full path to the embedding file.
            filename (str): The name of the embedding file.
            sd_model: The Stable Diffusion model, used for registering the embedding.
        """
        name, ext = os.path.splitext(filename)
        ext = ext.upper() # Convert extension to uppercase for comparison

        data = None
        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            _, second_ext = os.path.splitext(name)
            if second_ext.upper() == '.PREVIEW': # Skip preview files
                return

            # Try to read embedding data from image metadata
            data, name = self.read_embedding_from_image(path, name)
            if data is None:
                return
        elif ext in ['.BIN', '.PT']: # PyTorch binary files
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']: # Safetensors files
            data = safetensors.torch.load_file(path, device="cpu")
        else: # Unknown extension
            return

        if data is not None:
            # Create embedding object from loaded data
            # Assuming create_embedding_from_data is refactored to accept device
            embedding = create_embedding_from_data(data, name, filename=filename, filepath=path, device=devices.device) #TODO: pass device

            # Register if shape matches or if expected_shape is not yet set
            if self.expected_shape == -1 or self.expected_shape == embedding.shape:
                self.register_embedding(embedding, sd_model)
            else:
                self.skipped_embeddings[name] = embedding # Skip if shape mismatches
                print(f"Skipping embedding '{name}' due to shape mismatch. Expected {self.expected_shape}, got {embedding.shape}")
        else:
            print(f"Unable to load Textual Inversion embedding due to data issue: '{filename}'.")

    def load_from_dir(self, embdir, sd_model):
        """
        Loads all embeddings from a directory.

        Args:
            embdir (DirWithTextualInversionEmbeddings): The directory object to load from.
            sd_model: The Stable Diffusion model, passed to load_from_file.
        """
        if not os.path.isdir(embdir.path):
            return

        for root, _, fns in os.walk(embdir.path, followlinks=True):
            for fn in fns:
                try:
                    fullfn = os.path.join(root, fn)

                    if os.stat(fullfn).st_size == 0: # Skip empty files
                        continue

                    self.load_from_file(fullfn, fn, sd_model)
                except Exception as e:
                    print(f"Error loading embedding {fn}: {e}") # Use print instead of errors.report
                    continue

    def load_textual_inversion_embeddings(self, clip_model, sd_model, textual_inversion_print_at_load=True, force_reload=False):
        """
        Loads all textual inversion embeddings from the registered directories.

        Args:
            clip_model: The CLIP model (or text encoder) for `get_expected_shape`.
            sd_model: The Stable Diffusion model for `load_from_dir` and `register_embedding`.
            textual_inversion_print_at_load (bool): Whether to print loaded/skipped embeddings.
            force_reload (bool): If True, forces a reload even if directories haven't changed.
        """
        if not force_reload:
            need_reload = False
            for embdir in self.embedding_dirs.values():
                if embdir.has_changed(): # Check if any directory has been modified
                    need_reload = True
                    break
            if not need_reload:
                return

        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        self.expected_shape = self.get_expected_shape(clip_model) # Determine expected shape from clip_model

        for embdir in self.embedding_dirs.values():
            self.load_from_dir(embdir, sd_model) # Load embeddings from each directory
            embdir.update() # Update directory mtime

        # Sort word_embeddings alphabetically by name
        sorted_word_embeddings = {e.name: e for e in sorted(self.word_embeddings.values(), key=lambda e: e.name.lower())}
        self.word_embeddings.clear()
        self.word_embeddings.update(sorted_word_embeddings)

        if textual_inversion_print_at_load:
            displayed_embeddings = (tuple(self.word_embeddings.keys()), tuple(self.skipped_embeddings.keys()))
            if self.previously_displayed_embeddings != displayed_embeddings: # Print if there are changes
                self.previously_displayed_embeddings = displayed_embeddings
                print(f"Textual inversion embeddings loaded({len(self.word_embeddings)}): {', '.join(self.word_embeddings.keys())}")
                if self.skipped_embeddings:
                    print(f"Textual inversion embeddings skipped({len(self.skipped_embeddings)}): {', '.join(self.skipped_embeddings.keys())}")

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None


def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
def create_embedding(name: str, num_vectors_per_token: int, overwrite_old: bool, init_text: str = '*',
                     clip_model=None, device=None, embeddings_dir: str = None, save_optimizer_state: bool = False):
    """
    Creates a new embedding, initializes it, and saves it to a file.

    Args:
        name (str): The name for the new embedding.
        num_vectors_per_token (int): The number of vectors per token for the embedding.
        overwrite_old (bool): If True, overwrite an existing embedding file with the same name.
        init_text (str, optional): Text used to initialize the embedding vectors. Defaults to '*'.
        clip_model: The CLIP model (e.g., cond_stage_model) to use for initialization.
                    Required if init_text is provided.
        device (str or torch.device, optional): The device to create tensors on. If None, will attempt
                                               to use a default device (e.g., 'cpu' or from context).
        embeddings_dir (str, optional): The directory to save the embedding in. If None,
                                        a default path might be assumed by the caller or an error raised.
        save_optimizer_state (bool, optional): Whether to save an optimizer state. Defaults to False.
                                               Typically False for newly created embeddings not yet trained.

    Returns:
        str: The full path to the saved embedding file, or None on failure.

    Raises:
        ValueError: If required arguments like `clip_model` (for init_text), `device`,
                    or `embeddings_dir` are missing.
    """
    if init_text and not clip_model:
        raise ValueError("clip_model is required when init_text is provided.")
    if not device:
        # Fallback or raise error if a global device isn't desired/available
        # For now, let's assume 'cpu' if not provided, but ideally it should be explicit.
        # device = torch.device("cpu") # Or raise ValueError("Device must be specified.")
        raise ValueError("Device must be specified for tensor creation.")
    if not embeddings_dir:
        raise ValueError("Embeddings directory must be specified.")

    # Initialization logic using the passed clip_model
    # Assuming clip_model has `encode_embedding_init_text` method
    # The autocast context might be needed depending on the model and device (e.g., for MPS, half-precision)
    # For now, let's assume the caller handles autocast if necessary, or the model works without it.
    # with (devices.autocast() if devices else nullcontext()): # A1111 specific context
    
    embedded = clip_model.encode_embedding_init_text(init_text or '*', num_vectors_per_token)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=device)

    if init_text:
        for i in range(num_vectors_per_token):
            vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    # Sanitize name
    name = "".join(x for x in name if (x.isalnum() or x in "._- "))
    
    filepath = os.path.join(embeddings_dir, f"{name}.pt")
    if not overwrite_old and os.path.exists(filepath):
        raise FileExistsError(f"File {filepath} already exists and overwrite_old is False.")

    embedding = Embedding(vec, name)
    embedding.step = 0 # New embedding
    # Use the Embedding class's save method.
    # save_optimizer_state is passed, typically False for a new embedding.
    embedding.save(filepath, save_optimizer_state=save_optimizer_state)

    return filepath


def create_embedding_from_data(data, name, filename='unknown embedding file', filepath=None, device='cpu', file_hash=None):
    """
    Creates an Embedding object from loaded data.

    Args:
        data (dict): The loaded embedding data.
        name (str): The name of the embedding.
        filename (str, optional): The original filename. Defaults to 'unknown embedding file'.
        filepath (str, optional): The full path to the embedding file. Defaults to None.
        device (str or torch.device, optional): The device to move the embedding tensors to. Defaults to 'cpu'.
        file_hash (str, optional): Precomputed SHA256 hash of the file. If None, hash will not be set.

    Returns:
        Embedding: The created Embedding object.

    Raises:
        Exception: If the data format is not recognized.
    """
    vec = None
    shape = -1
    vectors = 0

    if 'string_to_param' in data:  # Standard textual inversion embeddings
        param_dict = data['string_to_param']
        # Fix for older PyTorch versions loading files saved from newer versions
        param_dict = getattr(param_dict, '_parameters', param_dict)
        if len(param_dict) != 1:
            raise ValueError(f"Embedding file '{filename}' has multiple terms in it. Only single-term embeddings are supported.")
        emb = next(iter(param_dict.items()))[1]
        vec = emb.detach().to(device, dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    elif isinstance(data, dict) and 'clip_g' in data and 'clip_l' in data:  # SDXL embedding
        # For SDXL, vec is a dict of tensors
        vec = {k: v.detach().to(device, dtype=torch.float32) for k, v in data.items() if isinstance(v, torch.Tensor)}
        # Shape and vectors might need to be handled differently for SDXL,
        # this is a simplified representation.
        if 'clip_g' in vec and 'clip_l' in vec:
            shape = vec['clip_g'].shape[-1] + vec['clip_l'].shape[-1] # Combined shape
            vectors = vec['clip_g'].shape[0] # Assuming vectors are consistent across parts
        else:
            raise ValueError(f"SDXL embedding file '{filename}' is missing 'clip_g' or 'clip_l' tensors.")
    elif isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), torch.Tensor):  # Diffuser concepts
        # Handles embeddings that are simple dicts with one tensor e.g. {'<concept>': tensor}
        emb = next(iter(data.values()))
        if len(emb.shape) == 1: # Ensure tensor is at least 2D
            emb = emb.unsqueeze(0)
        vec = emb.detach().to(device, dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    else:
        raise Exception(f"Couldn't identify '{filename}' as a recognized textual inversion embedding format (standard, SDXL, or diffuser concept).")

    embedding = Embedding(vec, name)
    embedding.step = data.get('step', None)
    embedding.sd_checkpoint = data.get('sd_checkpoint', None) # Checkpoint hash if available
    embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None) # Checkpoint name if available
    embedding.vectors = vectors
    embedding.shape = shape

    if filepath:
        embedding.filename = filepath
        if file_hash: # Set hash if provided
            embedding.set_hash(file_hash)
        # Original code used: hashes.sha256(filepath, "textual_inversion/" + name) or ''
        # This dependency on modules.hashes should be handled by the caller providing file_hash.

    return embedding


def write_loss(log_directory, filename, step, epoch_len, values):
    if shared.opts.training_write_csv_every == 0:
        return

    if step % shared.opts.training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        if write_csv_header:
            csv_writer.writeheader()

        epoch = (step - 1) // epoch_len
        epoch_step = (step - 1) % epoch_len

        csv_writer.writerow({
            "step": step,
            "epoch": epoch,
            "epoch_step": epoch_step,
            **values,
        })

def tensorboard_setup(log_directory):
    # This function seems okay for now but depends on shared.opts.training_enable_tensorboard
    # and shared.opts.training_tensorboard_flush_every.
    # If this function is part of the refactoring scope, these options should be passed as arguments.
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(os.path.join(log_directory, "tensorboard"), exist_ok=True)
    return SummaryWriter(
            log_dir=os.path.join(log_directory, "tensorboard"),
            flush_secs=shared.opts.training_tensorboard_flush_every) # A1111-specific option

def tensorboard_add(tensorboard_writer, loss, global_step, step, learn_rate, epoch_num):
    # This function itself is fine, but its usage might be tied to A1111's training loop.
    tensorboard_add_scaler(tensorboard_writer, "Loss/train", loss, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Loss/train/epoch-{epoch_num}", loss, step)
    tensorboard_add_scaler(tensorboard_writer, "Learn rate/train", learn_rate, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Learn rate/train/epoch-{epoch_num}", learn_rate, step)

def tensorboard_add_scaler(tensorboard_writer, tag, value, step):
    tensorboard_writer.add_scalar(tag=tag,
        scalar_value=value, global_step=step)

def tensorboard_add_image(tensorboard_writer, tag, pil_image, step):
    # Convert a pil image to a torch tensor
    img_tensor = torch.as_tensor(np.array(pil_image, copy=True))
    img_tensor = img_tensor.view(pil_image.size[1], pil_image.size[0],
        len(pil_image.getbands()))
    img_tensor = img_tensor.permute((2, 0, 1))

    tensorboard_writer.add_image(tag, img_tensor, global_step=step)

def validate_train_inputs(model_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_model_every, create_image_every, log_directory, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert isinstance(batch_size, int), "Batch size must be integer"
    assert batch_size > 0, "Batch size must be positive"
    assert isinstance(gradient_step, int), "Gradient accumulation step must be integer"
    assert gradient_step > 0, "Gradient accumulation step must be positive"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_filename, "Prompt template file not selected"
    assert template_file, f"Prompt template file {template_filename} not found"
    assert os.path.isfile(template_file.path), f"Prompt template file {template_filename} doesn't exist"
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0, "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0, "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0, "Create image must be positive or 0"
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"


def train_embedding(id_task, embedding_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width, training_height, varsize, steps, clip_grad_mode, clip_grad_value, shuffle_tags, tag_drop_out, latent_sampling_method, use_weight, create_image_every, save_embedding_every, template_filename, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_name, preview_cfg_scale, preview_seed, preview_width, preview_height):
    from modules import processing

    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    template_file = textual_inversion_templates.get(template_filename, None)
    validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_embedding_every, create_image_every, log_directory, name="embedding")
    template_file = template_file.path # Path from TextualInversionTemplate

    # shared.state calls are UI specific and should be removed for backend refactoring
    # shared.state.job = "train-embedding"
    # shared.state.textinfo = "Initializing textual inversion training..."
    # shared.state.job_count = steps

    # shared.cmd_opts.embeddings_dir is A1111-specific
    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    # shared.opts.unload_models_when_training is A1111-specific
    unload = shared.opts.unload_models_when_training

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    checkpoint = sd_models.select_checkpoint()

    initial_step = embedding.step or 0
    if initial_step >= steps:
        shared.state.textinfo = "Model has already been trained beyond specified max steps"
        return embedding, filename

    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else \
        torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else \
        None
    if clip_grad:
        clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    tensorboard_writer = None
    # shared.opts.training_enable_tensorboard is A1111-specific
    if shared.opts.training_enable_tensorboard:
        try:
            tensorboard_writer = tensorboard_setup(log_directory)
        except ImportError:
            # errors.report is A1111-specific, replace with print or logging
            print("Error initializing tensorboard: Tensorboard not installed or setup failed.")
            # errors.report("Error initializing tensorboard", exc_info=True)

    # shared.opts.pin_memory is A1111-specific
    pin_memory = shared.opts.pin_memory

    # modules.textual_inversion.dataset.PersonalizedBase and its arguments like shared.opts.training_image_repeats_per_epoch,
    # shared.sd_model, devices.device are A1111-specific.
    # This whole dataset and dataloader setup is tightly coupled with A1111.
    ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=template_file, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize, use_weight=use_weight)

    # shared.opts.save_training_settings_to_txt is A1111-specific
    if shared.opts.save_training_settings_to_txt:
        save_settings_to_file(log_directory, {**dict(model_name=checkpoint.model_name, model_hash=checkpoint.shorthash, num_of_dataset_images=len(ds), num_vectors_per_token=len(embedding.vec)), **locals()})

    latent_sampling_method = ds.latent_sampling_method

    dl = modules.textual_inversion.dataset.PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)

    if unload: # unload is from shared.opts
        # shared.parallel_processing_allowed is A1111-specific
        shared.parallel_processing_allowed = False
        # shared.sd_model and devices.cpu are A1111-specific
        shared.sd_model.first_stage_model.to(devices.cpu)

    embedding.vec.requires_grad = True
    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate, weight_decay=0.0)
    # shared.opts.save_optimizer_state is A1111-specific
    if shared.opts.save_optimizer_state:
        optimizer_state_dict = None
        if os.path.exists(f"{filename}.optim"): # filename is from shared.cmd_opts.embeddings_dir
            optimizer_saved_dict = torch.load(f"{filename}.optim", map_location='cpu')
            if embedding.checksum() == optimizer_saved_dict.get('hash', None):
                optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)

        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            print("Loaded existing optimizer from checkpoint")
        else:
            print("No saved optimizer exists in checkpoint")

    scaler = torch.cuda.amp.GradScaler()

    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # n steps = batch_size * gradient_step * n image processed
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0 #internal

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False

    is_training_inpainting_model = shared.sd_model.model.conditioning_key in {'hybrid', 'concat'}
    img_c = None

    pbar = tqdm.tqdm(total=steps - initial_step)
    try:
        sd_hijack_checkpoint.add()

        for _ in range((steps-initial_step) * gradient_step):
            if scheduler.finished:
                break
            if shared.state.interrupted:
                break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                scheduler.apply(optimizer, embedding.step)
                if scheduler.finished:
                    break
                if shared.state.interrupted:
                    break

                if clip_grad:
                    clip_grad_sched.step(embedding.step)

                with devices.autocast():
                    x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
                    if use_weight:
                        w = batch.weight.to(devices.device, non_blocking=pin_memory)
                    c = shared.sd_model.cond_stage_model(batch.cond_text)

                    if is_training_inpainting_model:
                        if img_c is None:
                            img_c = processing.txt2img_image_conditioning(shared.sd_model, c, training_width, training_height)

                        cond = {"c_concat": [img_c], "c_crossattn": [c]}
                    else:
                        cond = c

                    if use_weight:
                        loss = shared.sd_model.weighted_forward(x, cond, w)[0] / gradient_step
                        del w
                    else:
                        loss = shared.sd_model.forward(x, cond)[0] / gradient_step
                    del x

                    _loss_step += loss.item()
                scaler.scale(loss).backward()

                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue

                if clip_grad:
                    clip_grad(embedding.vec, clip_grad_sched.learn_rate)

                scaler.step(optimizer)
                scaler.update()
                embedding.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0

                steps_done = embedding.step + 1

                epoch_num = embedding.step // steps_per_epoch
                epoch_step = embedding.step % steps_per_epoch

                description = f"Training textual inversion [Epoch {epoch_num}: {epoch_step+1}/{steps_per_epoch}] loss: {loss_step:.7f}"
                pbar.set_description(description)
                if embedding_dir is not None and steps_done % save_embedding_every == 0:
                    # Before saving, change name to match current checkpoint.
                    embedding_name_every = f'{embedding_name}-{steps_done}'
                    last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
                # save_embedding now uses the save_optimizer_state argument from shared.opts
                save_embedding(embedding, optimizer, checkpoint, embedding_name_every, last_saved_file, remove_cached_checksum=True, save_optimizer_state=shared.opts.save_optimizer_state)
                    embedding_yet_to_be_embedded = True

                write_loss(log_directory, "textual_inversion_loss.csv", embedding.step, steps_per_epoch, {
                    "loss": f"{loss_step:.7f}",
                    "learn_rate": scheduler.learn_rate
                })

            if images_dir is not None and steps_done % create_image_every == 0: # create_image_every is an argument
                    forced_filename = f'{embedding_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)

                # shared.sd_model and devices.device are A1111-specific
                    shared.sd_model.first_stage_model.to(devices.device)

                # processing.StableDiffusionProcessingTxt2Img and processing.process_images are A1111-specific
                # This whole image generation block is A1111-specific.
                    p = processing.StableDiffusionProcessingTxt2Img(
                    sd_model=shared.sd_model, # A1111-specific
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                        do_not_reload_embeddings=True,
                    )

                if preview_from_txt2img: # preview_from_txt2img is an argument
                        p.prompt = preview_prompt
                        p.negative_prompt = preview_negative_prompt
                        p.steps = preview_steps
                    # sd_samplers is A1111-specific
                        p.sampler_name = sd_samplers.samplers_map[preview_sampler_name.lower()]
                        p.cfg_scale = preview_cfg_scale
                        p.seed = preview_seed
                        p.width = preview_width
                        p.height = preview_height
                    else:
                        p.prompt = batch.cond_text[0]
                        p.steps = 20
                        p.width = training_width
                        p.height = training_height

                    preview_text = p.prompt

                with closing(p): # p is A1111-specific
                    processed = processing.process_images(p) # A1111-specific
                        image = processed.images[0] if len(processed.images) > 0 else None

                if unload: # unload is from shared.opts
                    # shared.sd_model and devices.cpu are A1111-specific
                        shared.sd_model.first_stage_model.to(devices.cpu)

                    if image is not None:
                    # shared.state.assign_current_image is UI specific
                    # shared.state.assign_current_image(image)

                    # images.save_image and shared.opts.samples_format are A1111-specific
                        last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
                        last_saved_image += f", prompt: {preview_text}"

                    # shared.opts.training_tensorboard_save_images is A1111-specific
                        if tensorboard_writer and shared.opts.training_tensorboard_save_images:
                            tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image, embedding.step)

                if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded: # save_image_with_stored_embedding is an argument

                        last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

                        info = PngImagePlugin.PngInfo()
                        data = torch.load(last_saved_file)
                        info.add_text("sd-ti-embedding", embedding_to_b64(data))

                        title = f"<{data.get('name', '???')}>"

                        try:
                            vectorSize = list(data['string_to_param'].values())[0].shape[0]
                        except Exception:
                            vectorSize = '?'

                        checkpoint = sd_models.select_checkpoint()
                        footer_left = checkpoint.model_name
                        footer_mid = f'[{checkpoint.shorthash}]'
                        footer_right = f'{vectorSize}v {steps_done}s'

                        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                        captioned_image = insert_image_data_embed(captioned_image, data)

                        captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                        embedding_yet_to_be_embedded = False

                    # images.save_image and shared.opts.samples_format are A1111-specific
                    last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False) # A1111-specific
                    last_saved_image += f", prompt: {preview_text}"

                # shared.state.job_no and shared.state.textinfo are UI specific
                # shared.state.job_no = embedding.step
                # shared.state.textinfo = f"""
# <p>
# Loss: {loss_step:.7f}<br/>
# Step: {steps_done}<br/>
# Last prompt: {html.escape(batch.cond_text[0])}<br/>
# Last saved embedding: {html.escape(last_saved_file)}<br/>
# Last saved image: {html.escape(last_saved_image)}<br/>
# </p>
# """
        # shared.cmd_opts.embeddings_dir is A1111-specific
        filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
        # save_embedding now uses the save_optimizer_state argument from shared.opts
        save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True, save_optimizer_state=shared.opts.save_optimizer_state)
    except Exception as e: # Catch specific errors if possible
        # errors.report is A1111-specific, replace with print or logging
        print(f"Error training embedding: {e}")
        # errors.report("Error training embedding", exc_info=True)
    finally:
        pbar.leave = False
        pbar.close()
        # shared.sd_model, devices.device, shared.parallel_processing_allowed are A1111-specific
        shared.sd_model.first_stage_model.to(devices.device)
        shared.parallel_processing_allowed = old_parallel_processing_allowed
        # sd_hijack_checkpoint is A1111-specific
        sd_hijack_checkpoint.remove()

    return embedding, filename


def save_embedding(embedding: Embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True, save_optimizer_state=False):
    """
    Saves the embedding and its optimizer state (optional).

    Args:
        embedding (Embedding): The embedding object to save.
        optimizer: The optimizer used for training this embedding.
        checkpoint: The model checkpoint associated with this training.
        embedding_name (str): The name to save the embedding as.
        filename (str): The filename to save the embedding to.
        remove_cached_checksum (bool, optional): Whether to clear the cached checksum before saving. Defaults to True.
        save_optimizer_state (bool, optional): Whether to save the optimizer state. Defaults to False.
    """
    old_embedding_name = embedding.name
    old_sd_checkpoint = getattr(embedding, "sd_checkpoint", None)
    old_sd_checkpoint_name = getattr(embedding, "sd_checkpoint_name", None)
    old_cached_checksum = getattr(embedding, "cached_checksum", None)
    try:
        embedding.sd_checkpoint = checkpoint.shorthash # Assumes checkpoint has shorthash
        embedding.sd_checkpoint_name = checkpoint.model_name # Assumes checkpoint has model_name
        if remove_cached_checksum:
            embedding.cached_checksum = None
        embedding.name = embedding_name
        if save_optimizer_state: # Check flag before assigning
            embedding.optimizer_state_dict = optimizer.state_dict()
        else:
            embedding.optimizer_state_dict = None # Ensure it's None if not saving

        # Call the Embedding class's save method, passing the save_optimizer_state flag
        embedding.save(filename, save_optimizer_state=save_optimizer_state)
    except Exception as e:
        # Restore previous state on error
        embedding.name = old_embedding_name
        embedding.sd_checkpoint = old_sd_checkpoint
        embedding.sd_checkpoint_name = old_sd_checkpoint_name
        embedding.cached_checksum = old_cached_checksum
        # It might be good to log the error here or re-raise it with more context
        raise RuntimeError(f"Failed to save embedding {embedding_name} to {filename}: {e}") from e