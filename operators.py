import






def tokens_to_text():
    try:
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer
        class VanillaClip:
            def __init__(self, clip):
                self.clip = clip
            def vocab(self):
                return self.clip.tokenizer.get_vocab()
            def byte_decoder(self):
                return self.clip.tokenizer.byte_decoder
        class OpenClip:
            def __init__(self, clip):
                self.clip = clip
                self.tokenizer = open_clip.tokenizer._tokenizer
            def vocab(self):
                return self.tokenizer.encoder
            def byte_decoder(self):
                return self.tokenizer.byte_decoder
        clip = shared.sd_model.cond_stage_model
        if hasattr(clip,'embedders'):
            clip = clip.embedders[0]
        clip = clip.wrapped
        typename = type(clip).__name__.split('.')[-1]
        if typename=='FrozenOpenCLIPEmbedder':
            clip = OpenClip(clip)
        else:
            clip = VanillaClip(clip)
        vocab = {v: k for k, v in clip.vocab().items()}
        byte_decoder = clip.byte_decoder()
        def _tokens_to_text(tokens):
            nonlocal vocab, byte_decoder
            code = []
            ids = []
            current_ids = []
            class_index = 0
            def dump(last=False):
                nonlocal code, ids, current_ids
                words = [vocab.get(x, '') for x in current_ids]
                try:
                    word = bytearray([byte_decoder[x] for x in ''.join(words)]).decode('utf-8')
                except UnicodeDecodeError:
                    if last:
                        word = '<ERR>' * len(current_ids)
                    elif len(current_ids) > 4:
                        id = current_ids[0]
                        ids += [id]
                        local_ids = current_ids[1:]
                        code += [([id], '<ERR>')]

                        current_ids = []
                        for id in local_ids:
                            current_ids.append(id)
                            dump()
                        return
                    else:
                        return
                word = word.replace('</w>', ' ')
                code += [(current_ids, word)]
                ids += current_ids
                current_ids = []
            for token in tokens:
                token = int(token)
                current_ids.append(token)
                dump()
            dump(last=True)
            return [c for c in code if len(c[0])!=0]
        return _tokens_to_text
    except:
        traceback.print_exc()
        return None

def text_to_vectors(orig_text):
    try:
        both = []
        for clip,lg in zip(get_model_clips(),('clip_l','clip_g')):
            res = []
            text = orig_text.lstrip().lower()
            tokens = clip.tokenize_line(str_to_escape(text))
            count = tokens[1]
            tokens = tokens[0][0]
            fixes = tokens.fixes
            if count>=len(tokens.tokens):
                return None
            tokens = tokens.tokens[1:count+1]
            start = 0
            for fix in fixes:
                name = fix.embedding.name.lower()
                tensor = fix.embedding.vec
                if type(tensor)==dict:
                    tensor = tensor[lg]
                num = fix.embedding.vectors
                off = fix.offset
                if num!=tensor.size(0):
                    return None
                lenname = len(name)
                if off!=start:
                    test = 0
                    while True:
                        pos = text.find(name,test)
                        if pos<0:
                            return None
                        test = pos+lenname
                        sub = text[0:test]
                        part = clip.tokenize_line(str_to_escape(sub))
                        cnt = part[1]
                        part = part[0][0]
                        vec = off-start
                        need = tokens[start:off+num]
                        if part.tokens[1:cnt+1]==need:
                            trans = clip.encode_embedding_init_text(text,vec)
                            t = trans[:vec].to(device=devices.device,dtype=torch.float32)
                            res.append((t,sub[:pos],need[:vec]))
                            text = text[pos:]
                            start = off
                            break
                if text[0:lenname]!=name:
                    return None
                tensor = tensor.to(device=devices.device,dtype=torch.float32)
                res.append((tensor,name,None))
                start += num
                text = text[lenname:].lstrip()
            if text!='':
                part = clip.tokenize_line(str_to_escape(text))
                cnt = part[1]
                part = part[0][0]
                need = tokens[start:]
                if part.tokens[1:cnt+1]!=need:
                    return None
                trans = clip.encode_embedding_init_text(text,999)
                trans = trans.to(device=devices.device,dtype=torch.float32)
                res.append((trans,text,need))
            both.append(res)
        return both
    except:
        traceback.print_exc()
        return None

def text_to_tokens(text):
    try:
        both = []
        for clip in get_model_clips():
            tokens = clip.tokenize([text])[0]
            both.append(tokens)
        if len(both)>1:
            if (both[0]-both[1]).abs().max().item() != 0:
                print('EM: text_to_tokens',both)
                return None
        return both[0]
    except:
        return None

def tokens_to_vectors(pair):
    try:
        res = []
        for clip,arr in zip(get_model_clips(),pair):
            clip = clip.wrapped
            if hasattr(clip,'model') and hasattr(clip.model,'token_embedding'):
                tensor = torch.tensor([arr],dtype=torch.int,device=devices.device)
                tokens = clip.model.token_embedding.wrapped(tensor).to(devices.device)
            else:
                token_embedding = clip.transformer.text_model.embeddings.token_embedding
                tensor = torch.tensor([arr],dtype=torch.int,device=token_embedding.wrapped.weight.device)
                tokens = token_embedding.wrapped(tensor).to(devices.device)
            res.append(tokens)
        if len(res)>1:
            if len(res[0]) != len(res[1]):
                print('EM: tokens_to_vectors',res)
                return None
        return res
    except:
        traceback.print_exc()
        return None


def grab_vectors(text):
    try:
        both = []
        for res in text_to_vectors(text):
            if res is None:
                return None
            if len(res)==0:
                res = text_to_vectors(',')[len(both)][0][0][0:0]
            else:
                res = torch.cat([ten[0] for ten in res]);
            both.append(res)
        if len(both)>1:
            if len(both[0]) != len(both[1]):
                print('EM: grab_vectors',both)
                return None
        return both
    except:
        return None


def tensor_info(tensor):
    return '<td>{:>-14.8f}</td><td>{:>+14.8f}</td><td>{:>+14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td>'.format(tensor.min().item(),tensor.max().item(),tensor.sum().item(),tensor.abs().sum().item(),torch.linalg.norm(tensor,ord=2),tensor.std()).replace(' ','&nbsp;')

def merge_one_prompt(cache,texts,parts,used,prompt,prod,only_count):
    #if len(get_model_clips())>1:
    #    return (None,'To enable SDXL support switch to "sdxl" branch of https://github.com/klimaleksus/stable-diffusion-webui-embedding-merge')
    try:
        cnt = 0
        if (prompt is None) or (prompt==''):
            return (prompt,None)
        if texts is not None:
            if prompt in texts:
                return (texts[prompt],None)
        orig = prompt
        left = 0
        while True:
            curly = prompt.find("{'",left)
            left = prompt.find("<'",left)
            if (curly>=0 and curly<left) or (left<0):
                left = curly
                curly = True
            else:
                curly = False
            if left<0:
                if texts is not None:
                    texts[orig] = prompt
                return (prompt,None)
            eph = em_regexp.match(prompt[left:])
            if eph is not None:
                left += len(eph.group(0))
                continue
            right = left
            while True:
                right = prompt.find('}' if curly else '>',right+1)
                if right<0:
                    if curly:
                        return (None,'Not found closing "}" after "{\'"')
                    else:
                        return (None,'Not found closing ">" after "<\'"')
                if (prompt.count("'",left,right)&1)==0:
                    break
            part = prompt[left+1:right].strip()
            if part in parts:
                embed = parts[part]
            else:
                (res,err) = merge_parser(part,only_count)
                if err is not None:
                    return (None,err)
                if only_count:
                    if (res is None) or (res==0):
                        embed = ''
                    else:
                        embed = add_temp_embedding(None,cache,prod,curly,res)
                else:
                    if (res is None) or (res[0].numel()==0):
                        embed = ''
                    else:
                        embed = add_temp_embedding(res,cache,prod,curly,0)
                if used is not None:
                    used[embed] = part
                parts[part] = embed
            prefix = prompt[:left].rstrip()+' '+embed
            left = len(prefix)
            prompt = prefix+' '+(prompt[right+1:].lstrip())
    except:
        traceback.print_exc()
        return (None,'Fatal error?')
