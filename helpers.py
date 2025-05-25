import os
import folder_paths

def get_model_clips():
    clip = shared.sd_model.cond_stage_model
    if(hasattr(clip,'embedders')):
        try:
            return (clip.embedders[0],clip.embedders[1]) # SDXL
        except:
            pass
    return (clip,) # SD1 or SD2



merge_dir = None

def embedding_merge_dir():
    try:
        nonlocal merge_dir
        merge_dir = os.path.join(cmd_opts.embeddings_dir,'embedding_merge')
        # don't actually need this, since it is a subfolder which will be read recursively:
        #modules.sd_hijack.model_hijack.embedding_db.add_embedding_dir(merge_dir)
        os.makedirs(merge_dir)
    except:
        pass
