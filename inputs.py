import


reg_clean = re.compile(r'\s+')
reg_oper = re.compile(r'(=?)(?:([*/,])([+-]?[0-9]*(?:\.[0-9]*)?(?:L|G)?)|:([+-]?)(-?[0-9]+))')
sdxl_sizes = {
  'L': 768,
  'G': 1280,
}
em_regexp = re.compile(r"<'EM[_/-]\d+'>|{'EM[_/-]\d+'}")

def str_to_escape(line):
    res = re.sub(r'([()[\]\\])',r'\\\1',line)
    return res

def dict_replace(di,text):
    for key in di:
        text = text.replace(key,di[key])
    return text

def merge_parser(text,only_count):
    clips = get_model_clips()
    vocab = None
    def check_vocab(token2):
        nonlocal vocab
        if vocab is None:
            vocab = []
            for clip in clips:
                wrapped = clip.wrapped
                typename = type(wrapped).__name__.split('.')[-1]
                if typename=='FrozenCLIPEmbedder':
                    voc = wrapped.tokenizer.get_vocab()
                elif typename=='FrozenOpenCLIPEmbedder':
                    voc = open_clip.tokenizer._tokenizer.encoder
                else:
                    return True
                vocab.append({v: k for k, v in voc.items()})
        t = token2[0]
        if len(vocab)>1:
            if len(token2)>1:
                return (t in vocab[0]) and (token2[1] in vocab[1])
            return (t in vocab[0]) and (t in vocab[1])
        return t in vocab[0]
    orig = '"'+text+'"'
    text = text.replace('\0',' ')+' '
    length = len(text)
    arr = []
    left = 0
    quot = False
    join = False
    while left<length:
        pos = text.find("'",left)
        if pos<0:
            pos = length
        take = text[left:pos]
        if left>0:
            if take=='' and not quot:
                join = True
            elif quot:
                if join:
                    arr[-1] = (arr[-1][0]+"'"+take,True)
                    join = False
                else:
                    arr.append((take,True))
            else:
                arr.append((take.strip(),False))
        quot = not quot
        left = pos+1
    if not quot:
        return (None,'Last quote not closed in '+orig)
    if len(arr)>0 and arr[-1][0]=='':
        arr.pop()
    actions = []
    combine = False
    for param, quot in arr:
        one = param
        if quot:
            if combine:
                actions[-1]['V'] = param
                combine = False
            else:
                actions.append({
                  'A': None,
                  'V': param,
                  'O': one,
                })
            continue
        elif combine:
            return (None,'Wrong concatenation "'+param+'" in '+orig)
        param = reg_clean.sub('',param)
        while param!='':
            m = reg_oper.match(param)
            if not m:
                if param=='+' or param=='-':
                    actions.append({
                      'A': False,
                      'V': param=='+',
                      'O': one,
                    })
                    break
                return (None,'Wrong expression "'+param+'" in '+orig)
            m_flag = m.group(1)=='='
            m_mul = m.group(2)
            m_val = m.group(3)
            m_shift = m.group(4)
            m_size = m.group(5)
            m_tok = -1
            m_clip = None
            if m_val is not None:
                if len(m_val)>0:
                    m_clip = m_val[-1]
                    if (m_clip=='L') or (m_clip=='G'):
                        m_val = m_val[:-1]
                        if len(clips)<2:
                            return (None,'Suffix L or G can be used with SDXL models only: "'+param+'" in '+orig)
                    else:
                        m_clip = None
                if m_mul==',':
                    if m_flag:
                        return (None,'Concatenation doesn\'t support \'=\' prefix: "'+param+'" in '+orig)
                    if m_clip is not None:
                        return (None,'Concatenation doesn\'t support L or G suffix: "'+param+'" in '+orig)
                    if (len(m_val)>0) and (m_val[0]=='0'):
                        if m_val=='0':
                            m_tok = 0
                        elif m_val=='00':
                            m_tok = -2
                        elif m_val=='000':
                            m_tok = -3
                        elif m_val=='0000':
                            m_tok = -4
                        else:
                            m_tok = None
                    elif m_val=='':
                        m_tok = -5
                        combine = True
                        m_val = None
                    else:
                        m_tok = to_int(m_val)
                        if (m_tok is not None) and not (m_tok>=0):
                            m_tok = None
                    if m_tok is None:
                        return (None,'Bad param for concatenation "'+param+'" in '+orig)
                else:
                    m_val = to_float(m_val)
                    if m_val is None:
                        return (None,'Bad param for multiplication "'+param+'" in '+orig)
                    m_mul = m_mul=='*'
                m_size = -1
                m_shift = 0
            else:
                m_size = int(m_size)
                if m_shift=='+':
                    m_shift = m_size
                    m_size = -1
                elif m_shift=='-':
                    m_shift = -m_size
                    m_size = -1
                else:
                    m_shift = 0
                m_val = 1
                m_mul = None
            actions.append({
              'A': True,
              'V': m_val,
              'W': m_mul,
              'S': m_size,
              'R': m_shift,
              'F': m_flag,
              'T': m_tok,
              'C': m_clip,
              'O': one,
            })
            param = param[len(m.group(0)):]
    if combine:
        return (None,'Unfinished concatenation in '+orig)
    actions.append({
      'A': None,
      'V': None,
    })
    can_file = True
    can_add = False
    can_mul = False
    for act in actions:
        act['M'] = False
        A = act['A']
        if A==None:
            if act['V']==None:
                if can_file:
                    return (None,'Need quoted string after last + or - in '+orig)
                act['M'] = True
                break
            if can_file:
                can_add = True
                can_mul = True
                can_file = False
            else:
                return (None,'Quoted string without preceding + or - at \''+act['O']+'\' in '+orig)
        elif A==True:
            if can_mul:
                can_file = False
                can_add = True
                can_mul = True
                if act['F']:
                    act['M'] = True
            else:
                return (None,'Cannot multiply or modify at "'+act['O']+'" in '+orig)
        else:
            if can_add:
                can_file = True
                can_mul = False
                can_add = False
                act['M'] = True
            else:
                return (None,'Cannot merge at "'+act['O']+'" in '+orig)
    left = None
    right = None
    add = 0
    for act in actions:
        if act['M'] and (left is not None):
            if add!=0:
                if only_count:
                    if left>right:
                        right = left
                else:
                    (vectors1_0,length1_0) = left[0].size()
                    (vectors2_0,length2_0) = right[0].size()
                    (vectors1_1,length1_1) = left[1].size() if len(left)>1 else (vectors1_0,length1_0)
                    (vectors2_1,length2_1) = right[1].size() if len(right)>1 else (vectors2_0,length2_0)
                    if (length1_0!=length2_0) or (length1_1!=length2_1) or (vectors1_0!=vectors1_1) or (vectors2_0!=vectors2_1) or (len(left)!=len(right)):
                        return (None,'Cannot merge different embeddings in '+orig)
                    if vectors1_0!=vectors2_0:
                        if vectors1_0<vectors2_0:
                            target = [torch.zeros(vectors2_0,length1_0).to(device=devices.device,dtype=torch.float32)]
                            target[0][0:vectors1_0] = left[0]
                            if len(left)>1:
                                target.append(torch.zeros(vectors2_1,length1_1).to(device=devices.device,dtype=torch.float32))
                                target[1][0:vectors1_1] = left[1]
                            left = target
                        else:
                            target = [torch.zeros(vectors1_0,length2_0).to(device=devices.device,dtype=torch.float32)]
                            target[0][0:vectors2_0] = right[0]
                            if len(right)>1:
                                target.append(torch.zeros(vectors1_1,length2_1).to(device=devices.device,dtype=torch.float32))
                                target[1][0:vectors2_1] = right[1]
                            right = target
                    if add>0:
                        right[0] = left[0]+right[0]
                        if len(left)>1 and len(right)>1:
                            right[1] = left[1]+right[1]
                    else:
                        right[0] = left[0]-right[0]
                        if len(left)>1 and len(right)>1:
                            right[1] = left[1]-right[1]
            left = None
        A = act['A']
        if A==None:
            line = act['V']
            if line==None:
                return (right,None)
            right = grab_vectors(line)
            if right==None:
                return (None,'Failed to parse \''+line+'\' in '+orig)
            if only_count:
                right = right[0].size(0)
        elif A==False:
            if act['V']:
                add = 1
            else:
                add = -1
            left = right
            right = None
        else:
            s = act['S']
            r = act['R']
            t = act['T']
            if only_count:
                if t!=-1:
                    right += 1
                elif (r==0)and(s>=0):
                    right = s
            else:
                if t!=-1:
                    if t<0:
                        if t==-2:
                            t = [clip.id_pad for clip in clips]
                        elif t==-3:
                            t = [clip.id_end for clip in clips]
                        elif t==-4:
                            t = [clip.id_start for clip in clips]
                        else:
                            res = grab_vectors(act['V'])
                            t = None
                            if res is None:
                                return (None,'Failed to parse \''+act['V']+'\' in '+orig)
                    else:
                        if len(clips)>1:
                            t = [t,t]
                        else:
                            t = [t]
                    if t is not None:
                        if not check_vocab(t):
                            return (None,'Unknown token value \''+str(t[0])+'\' in '+orig)
                        res = tokens_to_vectors(t)
                    if res is None:
                        return (None,'Failed to convert token \''+str(t)+'\' in '+orig)
                    if right is None:
                        right = res
                    else:
                        if len(right)>1 and len(res)>1:
                            right = [torch.cat([right[0],res[0]]),torch.cat([right[1],res[1]])]
                        else:
                            right = [torch.cat([right[0],res[0]])]
                elif r!=0:
                    right[0] = right[0].roll(r,dims=0)
                    if len(right)>1:
                        right[1] = right[1].roll(r,dims=0)
                else:
                    if s>=0:
                        (vectors,length) = right[0].size()
                        if vectors>s:
                            if len(right)>1:
                                right = [right[0][0:s],right[1][0:s]]
                            else:
                                right[0] = right[0][0:s]
                        elif vectors<s:
                            target = [torch.zeros(s,length).to(device=devices.device,dtype=torch.float32)]
                            target[0][0:vectors] = right[0]
                            if len(right)>1:
                                (vectors,length) = right[1].size()
                                target.append(torch.zeros(s,length).to(device=devices.device,dtype=torch.float32))
                                target[1][0:vectors] = right[1]
                            right = target
                    elif act['W']==True:
                        if act['C']==None:
                            right = [r*act['V'] for r in right]
                        else:
                            s = sdxl_sizes[act['C']]
                            right = [(r*act['V'] if r.shape[-1]==s else r) for r in right]
                    elif  act['W']==False:
                        if act['C']==None:
                            right = [r/act['V'] for r in right]
                        else:
                            s = sdxl_sizes[act['C']]
                            right = [(r/act['V'] if r.shape[-1]==s else r) for r in right]
    return (right,None)
