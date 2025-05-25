import re
import os
import torch
import json
import html
import time
import types
import traceback
import threading
import open_clip.tokenizer













def gr_func(gr_name,gr_text,gr_radio,gr_tensors,store):
    with gr_lock:
        try:
            sd_models.reload_model_weights()
        except:
            pass
        gr_orig = gr_text
        font = 'font-family:Consolas,Courier New,Courier,monospace;'
        table = '<style>.webui_embedding_merge_table,.webui_embedding_merge_table td,.webui_embedding_merge_table th{border:1px solid gray;border-collapse:collapse}.webui_embedding_merge_table td,.webui_embedding_merge_table th{padding:2px 5px !important;text-align:center !important;vertical-align:middle;'+font+'font-weight:bold;}.webui_embedding_merge_table{margin:6px auto !important;}</style>'
        (reparse,request) = parse_infotext(gr_text)
        if reparse is not None:
            reparse = parse_mergeseq(reparse)
            if reparse is None:
                return ('<center><b>Prompt restore failed!</n></center>',gr_name,gr_orig)
            else:
                request = dict_replace(reparse,request)
                return ('<center><b>Prompt restored.</n></center>',gr_name,request)
        if gr_text[:1]=="'":
            (two,err) = merge_parser(gr_text,False)
            if (two is not None) and two[0].numel()==0:
                err = 'Result is ZERO vectors!'
            if err is not None:
                txt = '<b style="'+font+'">'+html.escape(err)+'</b>'
            else:
                txt = table
                both = False
                for res in two:
                    if res is None:
                        continue
                    if both:
                        txt += '<strong>↑ CLIP (L) / OpenClip (G) ↓</strong>'
                    txt += '<table class="webui_embedding_merge_table"><tr><th>Index</th><th>Min</th><th>Max</th><th>Sum</th><th>Abs</th><th>Len</th><th>Std</th>'
                    i = 1
                    for one in res:
                        txt += '<tr><td>{}</td>{}</tr>'.format(i,tensor_info(one))
                        i += 1
                    txt += '<tr><td colspan="7">&nbsp;</td></tr>'
                    txt += '<tr><td>ALL:</td>{}</tr>'.format(tensor_info(res))
                    txt += '</table>'
                    both = True
            return ('<center>'+txt+'</center>',need_save_embed(store,gr_name,two,gr_tensors),gr_orig)
        if gr_text.find("<'")>=0 or gr_text.find("{'")>=0:
            cache = reset_temp_embeddings('-',False)
            used = {}
            (mer,err) = merge_one_prompt(cache,None,{},used,gr_text,False,False)
            if err is not None:
                txt = '<b style="'+font+'">Embedding Merge failed - '+html.escape(err)+'</b>'
                return ('<center>'+txt+'</center>',gr_name,gr_orig)
            gr_text = mer
        by_none = 0
        by_comma = 1
        by_parts = 2
        by_words = 3
        by_tokens = 4
        by_vectors = 5
        tok2txt = tokens_to_text()
        if gr_radio!=by_comma:
            two = text_to_vectors(gr_text)
            if (gr_radio==by_none) and (two is not None) and (len(two[0])!=0):
                two = [[r] for r in two]
        else:
            two = [[],[]]
            split = gr_text.split(',')
            for part in split:
                one = text_to_vectors(part.strip())
                if one:
                    two[0].append(one[0])
                    if(len(one)>1):
                        two[1].append(one[1])
                    else:
                        two[1] = None
                else:
                    two = None
                    break
        if (two is None) or (len(two[0])==0):
            if gr_text.strip()=='':
                return ('',gr_name,gr_orig)
            txt = '<b>Failed to parse! (Possibly there are more than 75 tokens; or extra spaces inside embed names). Embeddings are not shown now:</b><br/><br/>'
            tokens = text_to_tokens(gr_text)
            if tokens:
                txt += table+'<tr><th>Index</th><th>Vectors</th><th>Text</th><th>Token</th></tr>'
                if tok2txt:
                    pairs = tok2txt(tokens)
                else:
                    pairs = [([tok],'<ERROR>') for tok in tokens]
                index = 1
                for arr, text in pairs:
                    length = len(arr)
                    if length==0:
                        continue
                    txt += '<tr><td>'+(str(index) if length==1 else str(index)+'-'+str(index+length-1))+'</td><td>'+str(length)+'</td><td>'+html.escape('"'+text+'"')+'</td><td>'+(', '.join([str(a) for a in arr]))+'</td></tr>'
                    index += length
                txt += '</table>'
            return ('<center>'+txt+'</center>',gr_name,gr_orig)
        both = []
        for res in two:
            if res is None:
                continue
            txt = '<table class="webui_embedding_merge_table"><tr><th>Index</th><th>Vectors</th><th>Text</th><th>Token</th><th>Min</th><th>Max</th><th>Sum</th><th>Abs</th><th>Len</th><th>Std</th></tr>'
            index = 1
            join = False
            if gr_radio==by_words:
                join = True
                gr_radio = by_tokens
            elif (gr_radio==by_none) or (gr_radio==by_comma):
                r_res = []
                for one in res:
                    r_tensor = []
                    r_name = ''
                    r_tokens = []
                    for tensor, name, tokens in one:
                        r_tensor.append(tensor)
                        if tok2txt and tokens and gr_radio==by_none:
                            split = tok2txt(tokens)
                            name = ''
                            tokens = []
                            for s_tokens, s_name in split:
                                name += s_name
                                tokens += s_tokens
                        r_name += name
                        if tokens:
                            r_tokens += tokens
                        else:
                            r_tokens += ['*_'+str(tensor.size(0))]
                            if gr_radio==by_none:
                                r_name += ' '
                    r_res.append((torch.cat(r_tensor),r_name,r_tokens))
                res = r_res
                gr_radio = by_parts
            for tensor, name, tokens in res:
                split = None
                size = tensor.size(0)
                span = ''
                if gr_radio!=by_parts:
                    span = ' rowspan="'+str(size)+'"'
                    if tokens and tok2txt:
                        split = tok2txt(tokens)
                        if join:
                            comb = []
                            last = -1
                            for s_arr, s_text in split:
                                if (last<0) or (comb[last][1][-1:]==' '):
                                    comb.append((s_arr,s_text))
                                    last += 1
                                else:
                                    comb[last] = (comb[last][0]+s_arr,comb[last][1]+s_text)
                            split = comb
                    if gr_radio==by_tokens:
                        if split is not None:
                            span = ' rowspan="'+str(len(split))+'"'
                        else:
                            span = ''
                if gr_radio==by_vectors:
                    head = '<td'+span+'>'+str(size)+'</td>'
                else:
                    head = '<td'+span+'>'+(str(index) if size==1 else str(index)+'-'+str(index+size-1))+'</td><td'+span+'>'+str(size)+'</td>'
                if split is None:
                    head += '<td'+span+'>'+html.escape('"'+name+'"')+'</td>'
                if (gr_radio==by_vectors) or ((gr_radio==by_tokens) and (tokens is not None)):
                    i = 0
                    part = 0
                    j = 0
                    ten = None
                    column = ''
                    toks = None
                    for one in list(tensor):
                        index += 1
                        i += 1
                        use = one
                        if split is not None:
                            if part==0:
                                pair = split[j]
                                part = len(pair[0])
                                if gr_radio==by_tokens:
                                    column = '<td>'+html.escape('"'+pair[1]+'"')+'</td>'
                                    toks = ', '.join([str(t) for t in pair[0]])
                                else:
                                    column = '<td rowspan="'+str(part)+'">'+html.escape('"'+pair[1]+'"')+'</td>'
                                j += 1
                        part -= 1
                        if gr_radio==by_tokens:
                            if ten==None:
                                ten = []
                            ten.append(one)
                            if part>0:
                                continue
                            use = torch.stack(ten)
                            tok = toks if tokens else '*'
                        else:
                            tok = tokens[i-1] if tokens else '*_'+str(i)
                        txt += '<tr>{}{}<td>{}</td>{}</tr>'.format(('<td>'+str(index-1)+'</td>' if gr_radio==by_vectors else '')+head,column,tok,tensor_info(use))
                        column = ''
                        head = ''
                        ten = None
                else:
                    index += size   
                    txt += '<tr>{}<td>{}</td>{}</tr>'.format(head,', '.join([str(t) for t in tokens]) if tokens else '*',tensor_info(tensor))
            txt += '</table>'
            both.append(txt)
        txt = table+'<strong>↑ CLIP (L) / OpenClip (G) ↓</strong>'.join(both)
        return ('<center>'+txt+'</center>',need_save_embed(store,gr_name,two,gr_tensors),gr_orig)



class EmbeddingMergeExtension(scripts.Script):
    def title(self):
        return 'Embedding Merge'
    def show(self,is_img2img):
        return scripts.AlwaysVisible
    def process(self,p):
        if hasattr(_webui_embedding_merge_,'embedding_merge_extension'):
            getattr(_webui_embedding_merge_,'embedding_merge_extension')(p,None)
    def postprocess(self,p,processed):
        if hasattr(_webui_embedding_merge_,'embedding_merge_extension'):
            getattr(_webui_embedding_merge_,'embedding_merge_extension')(p,processed)

script_callbacks.on_ui_tabs(_webui_embedding_merge_())
script_callbacks.on_infotext_pasted(_webui_embedding_merge_.on_infotext_pasted)
script_callbacks.on_script_unloaded(_webui_embedding_merge_.on_script_unloaded)
try:
    script_callbacks.on_model_loaded(_webui_embedding_merge_.on_model_loaded)
except:
    pass
