"""
Helper functions, some of which are specific to the A1111 WebUI environment
and will need adaptation for other platforms like ComfyUI.
"""
import os
# import folder_paths # Not used in the provided snippet, can be removed if not added during refactor
from modules import shared # A1111 specific global for model access
# from modules import cmd_opts # A1111 specific global for command line options (implicitly used)
# from modules import sd_hijack # A1111 specific for embedding DB (implicitly used)

# TODO: Refactor functions using 'shared', 'cmd_opts', 'sd_hijack' to accept
# necessary data as parameters for ComfyUI compatibility.

def get_model_clips():
    """
    Retrieves CLIP model objects from the A1111 WebUI's global `shared.sd_model`.

    This function is specific to the AUTOMATIC1111 Stable Diffusion WebUI environment.
    It attempts to access `shared.sd_model.cond_stage_model` and then checks
    if it has an `embedders` attribute, which is characteristic of SDXL models
    (having two CLIP embedders). If so, it tries to return both. Otherwise,
    it assumes a single CLIP model (SD1.x, SD2.x) and returns it as a tuple.

    Returns:
        tuple: A tuple containing one or two CLIP model objects.
               E.g., `(clip_model,)` for SD1/2, or `(clip_l, clip_g)` for SDXL.
               Returns `(None,)` or potentially raises an error if `shared.sd_model`
               or its attributes are not available as expected.

    TODO:
        This function is highly dependent on the A1111 WebUI's global state
        (`shared.sd_model`). For use in other environments like ComfyUI, this
        approach is not suitable. ComfyUI loads models through its node system,
        and CLIP models would typically be passed as inputs to nodes.
        The refactored functions in `operators.py` now accept `clip_models`
        as an argument, making this function less critical for the core parsing
        logic in a ComfyUI context, but it's documented here for completeness
        regarding its original role.
    """
    # This line directly accesses A1111's global shared state.
    clip = shared.sd_model.cond_stage_model
    if hasattr(clip, 'embedders'): # Check for SDXL style multiple embedders
        try:
            # Attempt to return both CLIP models for SDXL
            return (clip.embedders[0], clip.embedders[1])
        except IndexError:
            # Fallback or error if embedders structure is not as expected
            # For simplicity, returning the main clip if specific embedders aren't found
            # This might need more robust error handling depending on expected structure.
            return (clip,)
    # Default to returning the single CLIP model (SD1 or SD2)
    return (clip,)


_merge_dir_cache = None # Renamed from merge_dir to avoid confusion with local var in func

def embedding_merge_dir(base_embeddings_dir: str) -> str:
    """
    Ensures the 'embedding_merge' subdirectory exists within the given
    base embeddings directory and returns its path.

    This function was refactored to remove A1111 WebUI specific dependencies
    like `cmd_opts.embeddings_dir` and direct manipulation of the embedding database.
    It now takes the base directory for embeddings as an argument.

    Args:
        base_embeddings_dir (str): The path to the main directory where embeddings
                                   are stored.

    Returns:
        str: The path to the 'embedding_merge' subdirectory.
             Returns None if the directory cannot be created or accessed.
    
    Note:
        The global variable `_merge_dir_cache` is used to store the path after the
        first call, but this caching is simple and might not be ideal if
        `base_embeddings_dir` could change in a more complex application.
    """
    global _merge_dir_cache
    if _merge_dir_cache and os.path.dirname(_merge_dir_cache) == base_embeddings_dir : # Basic cache check
        return _merge_dir_cache

    try:
        if not base_embeddings_dir or not isinstance(base_embeddings_dir, str):
            # print("Error: base_embeddings_dir must be a valid path string.") # Or raise error
            return None

        target_merge_dir = os.path.join(base_embeddings_dir, 'embedding_merge')
        
        # Create the directory if it doesn't exist.
        # os.makedirs will create parent directories if they don't exist (though unlikely here)
        # and will not raise an error if the directory already exists (exist_ok=True).
        os.makedirs(target_merge_dir, exist_ok=True)
        
        _merge_dir_cache = target_merge_dir # Update cache
        return target_merge_dir
    except Exception as e:
        # print(f"Error creating or accessing embedding_merge directory: {e}") # Or log error
        _merge_dir_cache = None # Clear cache on error
        return None
