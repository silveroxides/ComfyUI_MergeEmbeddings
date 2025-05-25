import



def grab_embedding_cache():
    db = modules.sd_hijack.model_hijack.embedding_db
    field = '__embedding_merge_cache_'
    if hasattr(db,field):
        cache = getattr(db,field)
    else:
        cache = {'_':0,'-':0,'/':0}
        setattr(db,field,cache)
    return cache

def register_embedding(name,embedding):
    self = modules.sd_hijack.model_hijack.embedding_db
    model = shared.sd_model
    if hasattr(self,'register_embedding_by_name'):
        return self.register_embedding_by_name(embedding,model,name)
    # /modules/textual_inversion/textual_inversion.py
    try:
        ids = model.cond_stage_model.tokenize([name])[0]
        first_id = ids[0]
    except:
        return
    if embedding is None:
        if self.word_embeddings[name] is None:
            return
        del self.word_embeddings[name]
    else:
        self.word_embeddings[name] = embedding
    if first_id not in self.ids_lookup:
        if embedding is None:
            return
        self.ids_lookup[first_id] = []
    save = [(ids, embedding)] if embedding is not None else []
    old = [x for x in self.ids_lookup[first_id] if x[1].name!=name]
    self.ids_lookup[first_id] = sorted(old + save, key=lambda x: len(x[0]), reverse=True)
    return embedding

def make_temp_embedding(name,vectors,cache,fake):
    embed = None
    if name in cache:
        embed = cache[name]
        if fake>0:
            return
    else:
        if fake>0:
            if len(get_model_clips())>1:
                vectors = [torch.zeros((fake,16)),torch.zeros((fake,16))]
            else:
                vectors = [torch.zeros((fake,16))]
    shape = vectors[-1].size()
    if len(vectors)>1:
        vectors = {'clip_g':vectors[1],'clip_l':vectors[0]}
    else:
        vectors = vectors[0]
    if embed is None:
        embed = Embedding(vectors,name)
        cache[name] = embed
    embed.vec = vectors
    embed.step = None
    embed.vectors = shape[0]
    embed.shape = shape[-1]
    embed.cached_checksum = None
    embed.filename = ''
    register_embedding(name,embed)

def reset_temp_embeddings(prod,unregister):
    cache = grab_embedding_cache()
    num = cache[prod]
    cache[prod] = 0
    for a,b in (('<','>'),('{','}')):
        i = num
        while i>0:
            tgt = a+"'EM"+prod+str(i)+"'"+b
            if tgt in cache:
                embed = cache[tgt]
                if type(embed.vec)==dict:
                    for k,v in embed.vec.items():
                        embed.vec[k] = torch.zeros((0,v.shape[-1]),device=v.device)
                else:
                    embed.vec = torch.zeros((0,embed.vec.shape[-1]),device=embed.vec.device)
                embed.vectors = 0
                embed.cached_checksum = None
                del cache[tgt]
                if unregister:
                    register_embedding(tgt,None)
            i = i-1
    return cache

def add_temp_embedding(vectors,cache,prod,curly,fake):
    if fake>0:
        prod = '/'
        num = (cache[prod] or 0)
        if fake>num:
            cache[prod] = fake
        num = fake
    else:
        prod = '_' if prod else '-'
        num = 1+(cache[prod] or 0)
        cache[prod] = num
    name = "'EM"+prod+str(num)+"'"
    if curly:
        name = '{'+name+'}'
    else:
        name = '<'+name+'>'
    make_temp_embedding(name,vectors,cache,fake)
    return name

def min_or_all(a,b,n):
    if a>=0:
        if b>=0:
            if a<b:
                return a
            return b
        else:
            return a
    elif b>=0:
        return b
    return n

def need_save_embed(store,name,pair,tensors):
    if not store:
        return name
    name = ''.join( x for x in name if (x.isalnum() or x in '._- ')).strip()
    if name=='':
        return name
    try:
        if type(pair[0])==list:
            vectors = [torch.cat([r[0] for r in pair[0]])]
            if (len(pair)>1) and (pair[1] is not None):
                vectors.append(torch.cat([r[0] for r in pair[1]]))
        else:
            vectors = [pair[0]]
            if (len(pair)>1) and (pair[1] is not None):
                vectors.append(pair[1])
        target = os.path.join(merge_dir,name)
        if len(vectors)>1:
            pt = {
              'clip_g': vectors[1].cpu(),
              'clip_l': vectors[0].cpu(),
            }
        elif not tensors:
            pt = {
              'string_to_token': {
                '*': 265,
              },
              'string_to_param': {
                '*': vectors[0].cpu(),
              },
              'name': name,
              'step': 0,
              'sd_checkpoint': None,
              'sd_checkpoint_name': None,
            }
        if tensors:
            res = None
        else:
            torch.save(pt,target+'.pt')
            try:
                res = torch.load(target+'.pt',map_location='cpu')
            except:
                res = None
        if res is None:
            if len(vectors)==1:
                pt = {
                  'emb_params': vectors[0].cpu(),
                }
            from safetensors.torch import save_file
            save_file(pt,target+'.safetensors')
            try:
                os.unlink(target+'.pt')
            except:
                pass
        if tensors:
            if len(vectors)>1:
                for vector in vectors:
                    shape = vector.shape[-1]
                    if vector.abs().max().item() == 0:
                        shape = 0
                    if shape==768:
                        folder = os.path.join(merge_dir,'sd1')
                    else:
                        vector = None
                    try:
                        if vector is not None:
                            os.makedirs(folder)
                    except:
                        pass
                    target = os.path.join(folder,name)+'.safetensors'
                    if vector is not None:
                        from safetensors.torch import save_file
                        save_file({
                          'emb_params': vector.cpu(),
                        },target)
            else:
                folder = os.path.join(merge_dir,'sdxl')
                vector = vectors[0]
                shape = vector.shape[-1]
                if vector.abs().max().item() == 0:
                    shape = 0
                if shape==768:
                    s = list(vector.size())
                    s[-1] = 1280
                    pt = {
                      'clip_g': torch.zeros(s).cpu(),
                      'clip_l': vector.cpu(),
                    }
                else:
                    pt = None
                try:
                    if pt is not None:
                        os.makedirs(folder)
                except:
                    pass
                target = os.path.join(folder,name)+'.safetensors'
                if pt is not None:
                    from safetensors.torch import save_file
                    save_file(pt,target)
        try:
            modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        except:
            modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
        return ''
    except:
        traceback.print_exc()
        return name


fake_cached_params_counter = time.time()
def fake_cached_params(self,*ar,**kw):
    nonlocal fake_cached_params_counter
    fake_cached_params_counter += 1
    return (*(self.em_orig_cached_params(*ar,**kw)),id(_webui_embedding_merge_),fake_cached_params_counter)

cached_state = None
