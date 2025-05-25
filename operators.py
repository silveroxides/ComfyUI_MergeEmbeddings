import torch # Added for type hinting and direct torch usage
import traceback # Keep for error handling
# import open_clip.tokenizer # Potentially remove if tokenizer functionality is fully passed in
from modules import shared, devices # These are A1111 specific, aim to remove direct use in refactored functions
from helpers import get_model_clips, str_to_escape, add_temp_embedding, em_regexp # These are helpers, some may need refactoring or their deps passed
# merge_parser is not defined here, assumed to be external or part of a class structure.


def tokens_to_text(clip_tokenizer_model, tokens):
    """
    Converts a list of token IDs back into text using the provided CLIP tokenizer.

    It uses adapter classes `VanillaClip` or `OpenClip` to interface with different
    CLIP tokenizer implementations, expecting them to provide `vocab()` and `byte_decoder()` methods.

    Args:
        clip_tokenizer_model: A CLIP model object or a tokenizer object that has
                              `get_vocab()` (or `encoder` for OpenClip style) and
                              `byte_decoder()` methods, or can be wrapped by
                              VanillaClip/OpenClip adapters.
        tokens (list of int): A list of token IDs.

    Returns:
        function: An inner function `_tokens_to_text` that performs the conversion.
                  This inner function itself returns a list of tuples, where each
                  tuple is (list_of_token_ids, decoded_string_segment).
                  Returns None if an error occurs during setup.
    """
    try:
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer
        class VanillaClip:
            """Adapter for vanilla CLIP tokenizers (e.g., from OpenAI)."""
            def __init__(self, clip):
                self.clip = clip
            def vocab(self):
                return self.clip.tokenizer.get_vocab()
            def byte_decoder(self):
                return self.clip.tokenizer.byte_decoder

        class OpenClip:
            """Adapter for OpenCLIP tokenizers."""
            def __init__(self, clip):
                self.clip = clip
                # Assumes open_clip.tokenizer._tokenizer is the actual tokenizer instance
                self.tokenizer = open_clip.tokenizer._tokenizer
            def vocab(self):
                return self.tokenizer.encoder # OpenCLIP uses 'encoder' for vocab
            def byte_decoder(self):
                return self.tokenizer.byte_decoder

        # Determine the type of the passed model/tokenizer
        # If it's not already one of our known wrapped types, try to determine if it's OpenCLIP or Vanilla
        final_clip_tokenizer = None
        if hasattr(clip_tokenizer_model, 'tokenizer') and hasattr(clip_tokenizer_model.tokenizer, 'get_vocab'):
            # Likely a Vanilla CLIP model (e.g. wrapped SD1.5 CLIPModel)
            final_clip_tokenizer = VanillaClip(clip_tokenizer_model)
        elif hasattr(clip_tokenizer_model, 'tokenizer') and hasattr(clip_tokenizer_model.tokenizer, 'encoder'):
             # Potentially an OpenCLIP model if it has .tokenizer.encoder
            try:
                # This is a bit of a heuristic. If open_clip is available and the structure matches.
                import open_clip.tokenizer
                if isinstance(clip_tokenizer_model.tokenizer, open_clip.tokenizer.SimpleTokenizer):
                     final_clip_tokenizer = OpenClip(clip_tokenizer_model)
                else: # Fallback if it's not the SimpleTokenizer, but has encoder/byte_decoder directly
                    if hasattr(clip_tokenizer_model.tokenizer, 'byte_decoder'):
                        final_clip_tokenizer = clip_tokenizer_model.tokenizer # Assume it's a raw tokenizer
                    else:
                        raise ValueError("Tokenizer structure not recognized as OpenClip or VanillaClip compatible.")
            except ImportError:
                 # If open_clip is not importable, but it has .encoder, assume it's a raw tokenizer with that structure
                if hasattr(clip_tokenizer_model, 'encoder') and hasattr(clip_tokenizer_model, 'byte_decoder'):
                    final_clip_tokenizer = clip_tokenizer_model # Assume it's a raw tokenizer (OpenClip style)
                else:
                    raise ValueError("open_clip.tokenizer not found, and model structure not directly compatible.")

        elif hasattr(clip_tokenizer_model, 'get_vocab') and hasattr(clip_tokenizer_model, 'byte_decoder'):
            # It's already a tokenizer object (Vanilla Style)
            final_clip_tokenizer = clip_tokenizer_model
        elif hasattr(clip_tokenizer_model, 'encoder') and hasattr(clip_tokenizer_model, 'byte_decoder'):
            # It's already a tokenizer object (OpenClip Style)
            final_clip_tokenizer = clip_tokenizer_model
        else:
            # Try to infer from the type name if it's a raw A1111 style CLIP wrapper
            # This part is more heuristic and A1111 dependent.
            _clip_to_wrap = clip_tokenizer_model
            if hasattr(_clip_to_wrap, 'embedders'): # Handle A1111 SDXL structure
                _clip_to_wrap = _clip_to_wrap.embedders[0]
            if hasattr(_clip_to_wrap, 'wrapped'): # Handle A1111 general wrapper
                _clip_to_wrap = _clip_to_wrap.wrapped

            typename = type(_clip_to_wrap).__name__.split('.')[-1]
            if typename == 'FrozenOpenCLIPEmbedder':
                import open_clip.tokenizer # Needed for OpenClip adapter
                final_clip_tokenizer = OpenClip(_clip_to_wrap)
            elif typename == 'FrozenCLIPEmbedder':
                final_clip_tokenizer = VanillaClip(_clip_to_wrap)
            else:
                raise ValueError(f"Unsupported clip_tokenizer_model type: {typename}. Could not adapt to a known tokenizer interface.")

        vocab = {v: k for k, v in final_clip_tokenizer.vocab().items()}
        byte_decoder = final_clip_tokenizer.byte_decoder()

        def _tokens_to_text(tokens_list):
            nonlocal vocab, byte_decoder
            code = []
            ids = []
            current_ids = []
            # class_index = 0 # Seems unused
            def dump(last=False):
                nonlocal code, ids, current_ids
                words = [vocab.get(x, '') for x in current_ids]
                try:
                    word = bytearray([byte_decoder[x] for x in ''.join(words)]).decode('utf-8')
                except UnicodeDecodeError:
                    if last: # If it's the last segment and still fails, mark as error
                        word = '<ERR>' * len(current_ids)
                    # If not last and too long, try to process first token as error and retry rest
                    elif len(current_ids) > 4: # Heuristic: attempt to split if error on long multi-byte char
                        _id = current_ids[0]
                        ids.append(_id) # This seems to collect all processed ids, error or not
                        local_ids = current_ids[1:]
                        code.append(([_id], '<ERR>')) # Add error for the first token

                        current_ids = [] # Reset current_ids
                        for _id_inner in local_ids: # Process remaining tokens
                            current_ids.append(_id_inner)
                            dump() # Recursive call to dump
                        return
                    else: # If short and not last, just wait for more tokens
                        return
                word = word.replace('</w>', ' ') # Clean up subword indicators
                code.append((current_ids, word))
                ids.extend(current_ids) # Collect all processed ids
                current_ids = []

            for token in tokens_list: # Iterate over the input tokens
                token = int(token)
                current_ids.append(token)
                dump() # Try to decode after each token
            dump(last=True) # Final dump for any remaining tokens
            return [c for c in code if len(c[0]) != 0] # Filter out empty results
        return _tokens_to_text
    except Exception:
        traceback.print_exc()
        return None


def text_to_vectors(orig_text: str, clip_models: list, device: torch.device, str_escape_func=None):
    """
    Converts input text into a list of embedding vectors using the provided CLIP models.
    It handles standard text and can incorporate pre-defined textual inversion embeddings.

    Args:
        orig_text (str): The text prompt to convert.
        clip_models (list): A list of CLIP model objects. Each model should have
                            `tokenize_line()` and `encode_embedding_init_text()` methods,
                            and if it processes textual inversion, embeddings should have
                            `name`, `vec`, and `vectors` attributes. For SDXL,
                            `vec` can be a dict.
        device (torch.device): The PyTorch device to move tensors to.
        str_escape_func (callable, optional): A function to escape the string before tokenization.
                                             If None, no escaping is performed. Defaults to None.

    Returns:
        list: A list of lists of tensors. Each inner list corresponds to a CLIP model,
              and contains the embedding vector parts.
              Returns None if an error occurs or if token counts mismatch.
    """
    try:
        both_results = []
        # Determine if we are dealing with SDXL-style models (e.g., two CLIPs, 'clip_l', 'clip_g')
        # This is a heuristic; a more robust way might be needed if more diverse multi-CLIP setups exist.
        is_sdxl_style = len(clip_models) == 2 and hasattr(clip_models[0], 'layer_idx') # crude check for SDXL CLIPs

        for clip_idx, clip in enumerate(clip_models):
            lg_specific_key = None
            if is_sdxl_style:
                # Assign 'clip_l' to the first model, 'clip_g' to the second.
                # This assumes a specific order if two models are passed.
                lg_specific_key = 'clip_l' if clip_idx == 0 else 'clip_g'

            res = []
            text_to_process = orig_text.lstrip().lower()
            
            # Use provided escape function or default to identity
            current_str_escape_func = str_escape_func if callable(str_escape_func) else lambda x: x
            
            # Tokenize the entire line to get all tokens and TI fixes
            # The structure of `tokenize_line` and `fixes` is specific to A1111's CLIP wrapper.
            # ComfyUI's CLIP objects might have different interfaces.
            tokenized_line_result = clip.tokenize_line(current_str_escape_func(text_to_process))
            token_count = tokenized_line_result[1]
            all_tokens_info = tokenized_line_result[0][0] # Assuming this structure holds
            
            actual_tokens = all_tokens_info.tokens[1 : token_count + 1] # Extract relevant tokens
            fixes = all_tokens_info.fixes # Textual Inversion fixes

            if token_count >= len(all_tokens_info.tokens): # Basic validation
                return None # Or raise error

            current_token_idx = 0
            processed_text_len = 0

            for fix in fixes:
                ti_name = fix.embedding.name.lower()
                ti_tensor = fix.embedding.vec
                
                # Handle SDXL style TI where tensor is a dict
                if isinstance(ti_tensor, dict) and lg_specific_key:
                    ti_tensor = ti_tensor.get(lg_specific_key)
                    if ti_tensor is None:
                        print(f"Warning: TI {ti_name} missing key {lg_specific_key}")
                        # Skip this TI or handle error appropriately
                        continue 
                
                num_ti_vectors = fix.embedding.vectors
                ti_token_offset = fix.offset # This is the token offset

                # If the TI isn't at the current text position, encode the text before it
                if ti_token_offset > current_token_idx:
                    # This part is complex and relies on matching text segments to token segments.
                    # It's trying to find the exact text that corresponds to tokens[current_token_idx:ti_token_offset]
                    # This is prone to errors if tokenization of substrings doesn't align perfectly.
                    
                    # A simpler approach for refactoring might be to just encode text up to where TI is expected,
                    # but this loses the precise token matching of the original.
                    # For now, we try to keep the logic, but it's a candidate for simplification.
                    
                    # Attempt to find the text corresponding to tokens before TI
                    text_segment_to_find = text_to_process[processed_text_len : ] # Rough estimate
                    # This loop is to ensure the TI 'name' is found at the correct place
                    # after encoding the preceding text.
                    temp_search_offset = 0
                    found_match = False
                    while True:
                        # Find the TI name in the remaining text
                        actual_pos_of_ti_in_text = text_to_process.find(ti_name, processed_text_len + temp_search_offset)
                        if actual_pos_of_ti_in_text == -1:
                             # This TI name isn't in the rest of the text, major issue.
                            return None # Or raise

                        # Text before the TI
                        text_before_ti = text_to_process[processed_text_len : actual_pos_of_ti_in_text]
                        
                        # Tokenize this segment
                        tokens_for_segment = clip.tokenize_line(current_str_escape_func(text_before_ti))
                        num_tokens_for_segment = tokens_for_segment[1]
                        
                        # Check if the number of tokens matches the expected span
                        if num_tokens_for_segment == (ti_token_offset - current_token_idx):
                            # If tokens match, encode this text segment
                            encoded_segment_vectors = clip.encode_embedding_init_text(text_before_ti, num_tokens_for_segment)
                            encoded_segment_vectors = encoded_segment_vectors[:num_tokens_for_segment].to(device=device, dtype=torch.float32)
                            res.append((encoded_segment_vectors, text_before_ti, actual_tokens[current_token_idx:ti_token_offset]))
                            processed_text_len = actual_pos_of_ti_in_text
                            found_match = True
                            break
                        elif num_tokens_for_segment > (ti_token_offset - current_token_idx) and actual_pos_of_ti_in_text > 0 :
                            # This indicates an issue, perhaps the TI name appeared earlier than expected by token count
                            # or tokenization of segment is different. This part is tricky.
                            # Original code might have subtle bugs here.
                            # For now, if it's too many tokens, we assume the TI is further and continue search.
                             temp_search_offset = actual_pos_of_ti_in_text + len(ti_name) - processed_text_len
                             if temp_search_offset >= len(text_to_process) - processed_text_len : # boundary check
                                 return None # Cannot find a match
                        else: # Too few tokens, means TI is further, or name is part of a larger token
                             temp_search_offset = actual_pos_of_ti_in_text + len(ti_name) - processed_text_len
                             if temp_search_offset >= len(text_to_process) - processed_text_len :
                                 return None

                    if not found_match:
                        return None # Could not align text with tokens for the segment before TI

                # Now, add the TI embedding itself
                if not text_to_process[processed_text_len:].startswith(ti_name):
                     # This implies a mismatch between where we expect the TI and where it is.
                    return None # Or raise error

                ti_tensor_on_device = ti_tensor.to(device=device, dtype=torch.float32)
                res.append((ti_tensor_on_device, ti_name, None)) # TI embeddings don't have 'needed tokens'
                
                current_token_idx = ti_token_offset + num_ti_vectors
                processed_text_len += len(ti_name)
                text_to_process = text_to_process[processed_text_len:].lstrip() # Remove processed part
                processed_text_len = 0 # Reset for the new lstripped text_to_process

            # Process any remaining text after the last TI
            if text_to_process:
                # Tokenize remaining text
                remaining_tokens_info = clip.tokenize_line(current_str_escape_func(text_to_process))
                num_remaining_tokens = remaining_tokens_info[1]
                
                # Check if these tokens match the expected remaining actual_tokens
                expected_remaining_actual_tokens = actual_tokens[current_token_idx:]
                if remaining_tokens_info[0][0].tokens[1:num_remaining_tokens+1] != expected_remaining_actual_tokens:
                    # Mismatch between tokenizing remainder and original token stream
                    return None # Or raise error

                # Encode remaining text
                encoded_remainder_vectors = clip.encode_embedding_init_text(text_to_process, 999) # 999 for "all"
                encoded_remainder_vectors = encoded_remainder_vectors.to(device=device, dtype=torch.float32)
                res.append((encoded_remainder_vectors, text_to_process, expected_remaining_actual_tokens))

            both_results.append(res)
        return both_results
    except Exception:
        traceback.print_exc()
        return None


def text_to_tokens(text: str, clip_models: list):
    """
    Tokenizes the given text using a list of CLIP models.
    It currently assumes that if multiple CLIP models are provided, their tokenizations
    should be identical. If not, it prints a warning and returns None.

    Args:
        text (str): The text to tokenize.
        clip_models (list): A list of CLIP model objects, each having a `tokenize()` method.

    Returns:
        list: A list of token IDs. Returns the tokenization from the first model if consistent,
              otherwise None.
    """
    try:
        all_tokenizations = []
        for clip in clip_models:
            # `tokenize` typically returns a list of lists (for batch), so take the first.
            tokens = clip.tokenize([text])[0]
            all_tokenizations.append(tokens)

        if len(all_tokenizations) > 1:
            # Check for consistency if multiple tokenizers are used
            first_tokens = all_tokenizations[0]
            for i in range(1, len(all_tokenizations)):
                if (first_tokens - all_tokenizations[i]).abs().max().item() != 0: # Compare tensors
                    print('EM: text_to_tokens - Mismatch between CLIP tokenizations:', all_tokenizations)
                    return None # Mismatch
        return all_tokenizations[0] if all_tokenizations else None
    except Exception:
        traceback.print_exc()
        return None


def tokens_to_vectors(tokens_pair: list, clip_models: list, device: torch.device):
    """
    Converts lists of token IDs into embedding vectors using corresponding CLIP models.

    Args:
        tokens_pair (list): A list of token ID lists (or tensors). Each element in this list
                            corresponds to a CLIP model in `clip_models`.
                            E.g., for SDXL, this would be [clip_l_tokens, clip_g_tokens].
        clip_models (list): A list of CLIP model objects. Each model should have an interface
                            to get token embeddings (e.g., `model.token_embedding.wrapped` or
                            `transformer.text_model.embeddings.token_embedding.wrapped`).
        device (torch.device): The PyTorch device to move the resulting tensors to.

    Returns:
        list: A list of embedding vector tensors, one for each input token list.
              Returns None if an error occurs or if lengths mismatch (for multiple models).
    """
    try:
        result_vectors_list = []
        if len(tokens_pair) != len(clip_models):
            raise ValueError("Mismatch between number of token lists and CLIP models.")

        for clip, token_ids_for_clip in zip(clip_models, tokens_pair):
            current_clip_wrapped = clip.wrapped # Access the 'wrapped' model, common in A1111
            
            token_embedding_layer = None
            target_embedding_device = None

            if hasattr(current_clip_wrapped, 'model') and hasattr(current_clip_wrapped.model, 'token_embedding'):
                # Common path for some CLIP setups (e.g., OpenCLIP based from A1111)
                token_embedding_layer = current_clip_wrapped.model.token_embedding.wrapped
                target_embedding_device = token_embedding_layer.weight.device # Embeddings are on this device
            elif hasattr(current_clip_wrapped, 'transformer') and \
                 hasattr(current_clip_wrapped.transformer, 'text_model') and \
                 hasattr(current_clip_wrapped.transformer.text_model, 'embeddings') and \
                 hasattr(current_clip_wrapped.transformer.text_model.embeddings, 'token_embedding'):
                # Path for HuggingFace style CLIP models (e.g., ViT textual model)
                token_embedding_layer = current_clip_wrapped.transformer.text_model.embeddings.token_embedding.wrapped
                target_embedding_device = token_embedding_layer.weight.device
            else:
                raise AttributeError("Could not find token_embedding layer in the provided CLIP model.")

            # Ensure token_ids_for_clip is a tensor and on the correct device for the embedding lookup
            tokens_tensor = torch.tensor([token_ids_for_clip], dtype=torch.int, device=target_embedding_device)
            
            # Get embeddings and move to the specified output device
            vectors = token_embedding_layer(tokens_tensor).to(device)
            result_vectors_list.append(vectors)

        # Basic sanity check for SDXL-like paired outputs (ensure same sequence length)
        if len(result_vectors_list) > 1:
            if result_vectors_list[0].shape[1] != result_vectors_list[1].shape[1]: # Compare sequence length (dim 1)
                print('EM: tokens_to_vectors - Mismatch in sequence length between CLIP model outputs:', 
                      [v.shape for v in result_vectors_list])
                return None
        return result_vectors_list
    except Exception:
        traceback.print_exc()
        return None


def grab_vectors(text: str, clip_models: list, device: torch.device, str_escape_func=None):
    """
    A convenience function that converts text to embedding vectors using `text_to_vectors`
    and then concatenates the tensor parts for each CLIP model.

    Args:
        text (str): The text prompt to convert.
        clip_models (list): A list of CLIP model objects.
        device (torch.device): The PyTorch device for tensors.
        str_escape_func (callable, optional): String escaping function for `text_to_vectors`.

    Returns:
        list: A list of concatenated embedding tensors, one for each CLIP model.
              Returns None if `text_to_vectors` fails.
    """
    try:
        vector_parts_per_clip = text_to_vectors(text, clip_models, device, str_escape_func)
        if vector_parts_per_clip is None:
            return None

        concatenated_vectors = []
        for clip_idx, parts_for_one_clip in enumerate(vector_parts_per_clip):
            if not parts_for_one_clip: # If no parts (e.g., empty text resulted in empty list)
                # Create an empty tensor of appropriate shape (0, embedding_dim)
                # This requires knowing the embedding dimension of the clip model.
                # Fallback: call text_to_vectors with a comma to get a dummy tensor for shape.
                # This is inefficient and hacky. A better way would be to access model.text_projection.shape[-1] or similar.
                # For now, using the original hacky way:
                dummy_parts = text_to_vectors(',', clip_models, device, str_escape_func)
                if dummy_parts and len(dummy_parts) > clip_idx and dummy_parts[clip_idx]:
                     # Create an empty tensor with 0 tokens but correct embedding dimension
                    empty_tensor_shape = (0, dummy_parts[clip_idx][0][0].shape[-1])
                    concatenated_vectors.append(torch.empty(empty_tensor_shape, device=device, dtype=torch.float32))
                else: # Could not determine shape, append None or raise error
                    print(f"Warning: Could not determine embedding shape for empty tensor for clip {clip_idx}")
                    concatenated_vectors.append(torch.empty((0,0), device=device, dtype=torch.float32)) # Fallback
            else:
                # Concatenate all tensor parts for this specific CLIP model
                concatenated_vectors.append(torch.cat([tensor_part[0] for tensor_part in parts_for_one_clip]))
        
        # Sanity check for SDXL-like outputs (ensure same sequence length after cat)
        if len(concatenated_vectors) > 1:
            if concatenated_vectors[0].shape[0] != concatenated_vectors[1].shape[0]: # Compare num tokens (dim 0 after cat)
                print('EM: grab_vectors - Mismatch in token length after concatenation:',
                      [v.shape for v in concatenated_vectors])
                return None
        return concatenated_vectors
    except Exception:
        traceback.print_exc()
        return None


def tensor_info(tensor: torch.Tensor) -> str:
    """
    Generates an HTML string containing min, max, sum, abs sum, L2 norm, and std dev of a tensor.
    Used for debugging or displaying tensor information.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        str: An HTML formatted string with tensor statistics.
    """
    return '<td>{:>-14.8f}</td><td>{:>+14.8f}</td><td>{:>+14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td>'.format(
        tensor.min().item(), tensor.max().item(), tensor.sum().item(),
        tensor.abs().sum().item(), torch.linalg.norm(tensor, ord=2), tensor.std()
    ).replace(' ', '&nbsp;')


def merge_one_prompt(
    cache: dict,
    texts: dict,
    parts: dict,
    used: dict,
    prompt: str,
    prod: bool, # TODO: Document what 'prod' signifies
    only_count: bool,
    clip_models: list, # New argument for CLIP models
    device: torch.device, # New argument for device
    # merge_parser_func: callable, # Expected signature: (str, bool, list, torch.device) -> tuple
    # add_temp_embedding_func: callable # Expected signature: (torch.Tensor or list, dict, bool, bool, int) -> str
    # For now, assume merge_parser and add_temp_embedding are available globally or via 'helpers'
    # and will be refactored to use clip_models and device if they need them internally.
    # The direct calls below will be updated to pass these new args.
    str_escape_func = None # Optional: if merge_parser needs it for its own text_to_vector calls
):
    """
    Recursively parses a prompt string, replacing special syntax with merged embeddings.

    The parser identifies two main syntaxes:
    1. `"{'expression'}"`: This syntax indicates a recursive call to `merge_parser`
       (or an equivalent function that resolves expressions to embeddings). The
       result is an embedding that's temporarily stored and referenced.
    2. `"<'.name.'>"`: This syntax indicates a literal embedding name that should
       be fetched or processed, also resulting in a temporary embedding reference.

    The function iteratively finds these patterns, resolves them to embedding
    references (often temporary names generated by `add_temp_embedding`), and
    reconstructs the prompt string.

    Args:
        cache (dict): A cache, possibly for storing results of `add_temp_embedding` or similar.
        texts (dict): A dictionary to cache results of fully processed prompts to avoid re-processing.
        parts (dict): A dictionary to cache results of parsed sub-prompts (the `part` string).
        used (dict): A dictionary to track which temporary embedding names were used for which parts.
        prompt (str): The prompt string to parse.
        prod (bool): A boolean flag, purpose needs further documentation (e.g., production mode?).
                     It's passed to `add_temp_embedding`.
        only_count (bool): If True, `merge_parser` might only return counts or simple
                           representations instead of full tensors. Passed to `merge_parser`
                           and affects `add_temp_embedding` logic.
        clip_models (list): List of CLIP model objects needed by `merge_parser` or `add_temp_embedding`.
        device (torch.device): PyTorch device needed by `merge_parser` or `add_temp_embedding`.
        str_escape_func (callable, optional): String escaping function, if needed by merge_parser.

    Returns:
        tuple: (processed_prompt_string, error_message_or_None)
               If an error occurs, the string is None and error_message is set.
    """
    # Note: The original code had a commented-out SDXL check.
    # If len(get_model_clips()) > 1, it would return an error.
    # This refactored version receives clip_models, so such a check might be different.
    try:
        # cnt = 0 # Seems unused
        if prompt is None or prompt == '':
            return (prompt, None)

        # Cache for full prompt processing
        if texts is not None and prompt in texts:
            return (texts[prompt], None)

        original_prompt_for_cache = prompt
        current_pos = 0 # Tracks search position in the prompt string

        while True:
            # Determine if the next special syntax is `"{'...'"}` (curly) or `"<'...'>"` (angle)
            # Syntax: "{'recursive_part_or_expression'}"
            curly_brace_pos = prompt.find("{'", current_pos)
            # Syntax: "<'.literal_embedding_name_or_simple_expr.'>" (less clear if also recursive from context)
            angle_bracket_pos = prompt.find("<'", current_pos)

            # Choose the one that appears earliest, or if one is not found
            is_curly_syntax = False
            next_match_pos = -1

            if curly_brace_pos != -1 and angle_bracket_pos != -1:
                if curly_brace_pos < angle_bracket_pos:
                    next_match_pos = curly_brace_pos
                    is_curly_syntax = True
                else:
                    next_match_pos = angle_bracket_pos
            elif curly_brace_pos != -1:
                next_match_pos = curly_brace_pos
                is_curly_syntax = True
            elif angle_bracket_pos != -1:
                next_match_pos = angle_bracket_pos
            else: # No more special syntax found
                break # Exit the parsing loop

            # Check if the found pattern is part of an escaped sequence (e.g., an embedding name itself)
            # This uses em_regexp from helpers, assuming it's designed for this.
            # Example: <'my_embedding_{'detail'}'> might be valid if em_regexp handles it.
            # This part ensures we don't misinterpret parts of valid (e.g. TI) names.
            if em_regexp and em_regexp.match(prompt[next_match_pos:]):
                current_pos = next_match_pos + len(em_regexp.match(prompt[next_match_pos:]).group(0))
                continue

            # Find the corresponding closing bracket ('}' or '>')
            # This must correctly handle nested quotes inside the expression.
            closing_char = '}' if is_curly_syntax else '>'
            content_start_pos = next_match_pos + 2 # After "{'" or "<'"
            content_end_pos = -1
            search_for_closing_from = content_start_pos
            while True:
                content_end_pos = prompt.find(closing_char, search_for_closing_from)
                if content_end_pos == -1:
                    return (None, f"Not found closing '{closing_char}' after position {next_match_pos}")
                # Ensure quotes are balanced within the content part before this closing char
                if (prompt.count("'", content_start_pos, content_end_pos) % 2) == 0:
                    break # Found valid closing bracket
                search_for_closing_from = content_end_pos + 1 # Continue search after this quote

            # Extract the content (the part inside {'...'} or <'...'>)
            part_content = prompt[content_start_pos:content_end_pos].strip()
            
            resolved_embedding_ref = ""
            if part_content in parts: # Check if this exact part content was already processed
                resolved_embedding_ref = parts[part_content]
            else:
                # Call merge_parser (assumed external) to resolve the content.
                # It needs to be passed clip_models and device now.
                # TODO: Confirm exact signature of refactored merge_parser.
                (parsed_result, error_msg) = merge_parser(part_content, only_count, clip_models=clip_models, device=device, str_escape_func=str_escape_func)
                if error_msg is not None:
                    return (None, error_msg)

                # Call add_temp_embedding (assumed external) to get a reference string for the parsed result.
                # This also needs context if it interacts with CLIP models or embeddings.
                # TODO: Confirm exact signature of refactored add_temp_embedding.
                # Assuming it might need (parsed_result, cache, prod_flag, is_curly_syntax, count_if_any, clip_models, device)
                if only_count:
                    if parsed_result is None or parsed_result == 0: # Condition for empty result in count mode
                        resolved_embedding_ref = ''
                    else: # parsed_result here is likely a count
                        resolved_embedding_ref = add_temp_embedding(None, cache, prod, is_curly_syntax, parsed_result, clip_models=clip_models, device=device)
                else: # Not only_count, parsed_result should be tensor(s)
                    if parsed_result is None or (isinstance(parsed_result, list) and (not parsed_result or parsed_result[0].numel() == 0)): # Condition for empty tensor result
                        resolved_embedding_ref = ''
                    else:
                        resolved_embedding_ref = add_temp_embedding(parsed_result, cache, prod, is_curly_syntax, 0, clip_models=clip_models, device=device)
                
                if used is not None: # Track usage if 'used' dict is provided
                    used[resolved_embedding_ref] = part_content
                parts[part_content] = resolved_embedding_ref # Cache this part's resolution

            # Reconstruct the prompt string
            text_before_match = prompt[:next_match_pos].rstrip()
            # Add space if text_before_match is not empty and resolved_embedding_ref is not empty
            spacing = ' ' if text_before_match and resolved_embedding_ref else ''
            prefix = text_before_match + spacing + resolved_embedding_ref
            
            current_pos = len(prefix) # Update current_pos for next iteration's search start
            prompt = prefix + ' ' + (prompt[content_end_pos + 1:].lstrip()) # Add space after, then rest of prompt

        if texts is not None: # Cache the final processed prompt
            texts[original_prompt_for_cache] = prompt
        return (prompt, None)
    except Exception:
        traceback.print_exc()
        return (None, 'Fatal error during prompt merging.')
