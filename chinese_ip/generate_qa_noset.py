# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from datetime import datetime
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint, get_checkpoint_iteration
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu
import deepspeed
import copy
from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from pretrain_gpt2 import get_model
from pypinyin import pinyin,FINALS, FINALS_TONE,TONE3
import jsonlines
def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    # if args.deepspeed:
    #     print_rank_0("DeepSpeed is enabled.")
    #
    #     model, _, _, _ = deepspeed.initialize(
    #         model=model,
    #         model_parameters=model.parameters(),
    #         args=args,
    #         mpu=mpu,
    #         dist_init_required=False
    #     )
    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            print(iteration)
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["module"])
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)
    # if args.deepspeed:
    #     model = model.module

    return model


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)

    return tokens, attention_mask, position_ids

def generate_score(model, tokenizer, args, device, input_str, eval_str):
    #penalty on same word

    context_count = 0
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eval_tokens = tokenizer.EncodeAsIds(eval_str).tokenization + [tokenizer.get_command('eos').Id]
        if len(context_tokens)==0:
            context_tokens = eval_tokens[0:1]
            eval_tokens = eval_tokens[1:]
        context_length = len(context_tokens)
        eval_length = len(eval_tokens)
        if context_length >= args.seq_length:
            return "输入过长。"

        # terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
        # pad_id = tokenizer.get_command('pad').Id
        # if context_length < args.out_seq_length:
        #     context_tokens.extend([pad_id] * (args.out_seq_length - context_length))

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eval_tokens_tensor = torch.cuda.LongTensor([eval_tokens])
        context_length_tensor = torch.cuda.LongTensor([context_length])
        eval_length_tensor = torch.cuda.LongTensor([eval_length])
        # context_length = context_length_tensor[0].item()
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
        # print(context_tokens)
        start_time = time.time()

        counter, mems = 0, []
        org_context_length = context_length
        sumlognum = 0
        while counter < eval_length:
            if counter == 0:
                logits, *mems = model(tokens, position_ids, attention_mask, *mems)
                logits = logits[:, -1]
            else:
                index = org_context_length + counter
                logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                                      tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                      dtype=torch.float), *mems)
                logits = logits[:, 0]
            # logits = logits[:, -1]
            #logits /= args.temperature
           # logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)
            log_num = torch.log(log_probs).data
            # print(log_num)
            num=F.relu(35+log_num[0, eval_tokens[counter]])-35
            
            sumlognum += num
            # print(log_probs)
            # prev = torch.multinomial(log_probs, num_samples=1)[0]
            # print(tokens,eval_tokens_tensor[counter:counter+1])
            tokens = torch.cat((tokens, eval_tokens_tensor[:, counter:counter + 1]), dim=1)
            # print(tokens,sumlognum)
            context_length += 1
            counter += 1

        # trim_decode_tokens = decode_tokens[:decode_tokens.find("<|endoftext|>")]
        sumlognum = sumlognum
        del logits
        del mems
        torch.cuda.empty_cache()
        return sumlognum
        
def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits

def generate_token_tensor(str,tokenizer):
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(str).tokenization
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        return context_tokens_tensor
    
def checksentence(sentence,original_context,min_length,max_length,endnote):
    if "<|end" in sentence:
        return 0
            
    if ((len(sentence)>max_length and not(sentence[-1] in endnote)) or len(sentence)==0) or len(sentence)>max_length+1:
        return 1
    if (sentence[-1] in endnote)and (len(sentence)<=min_length):
        return 1
            
    if (sentence[-1] in endnote)and (sentence[:-1] in original_context):
        return 1
    if (len(sentence)>4 and (sentence[-4:] in original_context)):
        return 1
    if (sentence[-1] in endnote):
        return 0
        
        
    return 2
    
             

    
def generate_sentence(model,tokenizer,args,device,current_tokens,mems,endnote=[",","，","。","!","！","——",":","?","？"],num_candidates=10,min_length=3,max_length=16):
    model.eval()
    with torch.no_grad():
        #index=len(tokens[0])
        mct_tree=[]
        if min_length!=max_length:
            mems=[]
            tokens, attention_mask, position_ids = get_batch(current_tokens, device, args)
            logits,*rts = model(tokens, position_ids, attention_mask, *mems)
        else:
            tokens=current_tokens
            index=len(tokens[0])
            logits,*rts=model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                        tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                            dtype=torch.float), *mems)
                                                            
        output_tokens_list = tokens.view(-1).contiguous()
        original_context=tokenizer.DecodeIds(output_tokens_list.tolist())
        context_length=len(tokens[0])
        logits=logits[0,-1]
        #mct_structure=-np.ones(len(logits))
        mct_tree.append([logits,rts,tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),0])
        #print(logits.shape)
        final_result=[]
        nextid=0
        tries=0
        max_tries=num_candidates*30
        while (len(final_result)<num_candidates)and(tries<max_tries):
            currentid=nextid
            tries+=1
            while currentid!=-1:
                tc=torch.log(mct_tree[currentid][4])
                tc=tc+F.relu(tc-10)*1000
                logits=mct_tree[currentid][0].view(-1)-tc*0.5
                logits=logits[:50001]
                log_probs = F.softmax(logits/args.temperature, dim=-1)
              
                pr=torch.multinomial(log_probs,num_samples=1)[0]
                #pr=torch.argmax(logits)
                prev=pr.item()
                #print(logits.shape,currentid,prev)
                mct_tree[currentid][4][prev]+=1
                lastid=currentid
                currentid=int(mct_tree[currentid][3][prev])
            #start from lastid & currentid
            
            cqs=mct_tree[lastid][2]
            #print(pr)
            tokens = torch.cat((cqs, pr.unsqueeze(0).view(1, 1)), dim=1)
            output_tokens_list = tokens.view(-1).contiguous()
            #if max_length==min_length:
             #   print(min_length,output_tokens_list,context_length)
            #print(output_tokens_list[context_length:])
            sentence = tokenizer.DecodeIds(output_tokens_list[context_length:].tolist())
            
            #print(output_tokens_list[context_length:],context_length,sentence)
            logit=mct_tree[lastid][0]
            log_probs = F.softmax(logit, dim=-1)
            log_pbs=torch.log(log_probs)
            score=log_pbs[prev].item()
            nextid=0
            ip=checksentence(sentence,original_context,min_length,max_length,endnote)
            for j in final_result:
                if j[0]==sentence:
                    ip=1
                if ('<|end' in sentence) and ('<|end' in j[0]):
                    ip=1
                    
            score=mct_tree[lastid][5]+score
            if (ip==1):
                mct_tree[lastid][4][prev]=10000
                continue
            if (ip==0):
                mct_tree[lastid][4][prev]=10000
                final_result.append([copy.deepcopy(sentence),copy.deepcopy(score),copy.deepcopy(tokens),copy.deepcopy(mct_tree[lastid][1])])
                #print(sentence,score)
                continue
        
           
            
                #calculate
            mct_tree[lastid][3][prev]=len(mct_tree)
            rts=mct_tree[lastid][1]
            index=len(tokens[0])
            
            
            logits,*rts=model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                        tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                                dtype=torch.float), *rts)
            logits=logits[0,-1]
            
            mct_tree.append([logits,rts,tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),score])
            nextid=len(mct_tree)-1
        del mct_tree
        torch.cuda.empty_cache()
        #print(tries,len(final_result))
        return final_result
def getlength(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace(' ',',').replace('！',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    
    return len(sp[-2])

def getlastsentence(str):
    signal=['。',',','，','!','！','?','？',' ']
    signal2=[':']
    fom=''
    sig1=0
    sig2=0
    nowplace=0
    while True:
        nowplace+=1
        if len(str)<nowplace:
            return str
        if str[-nowplace] in signal:
            if nowplace>20:
                return str[-nowplace+1:]
        
    return 0

def get2sentencebefore(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace(' ',',').replace('！',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    idk=-1
    while len(sp[idk])==0:
        idk-=1
    idk-=1
    while len(sp[idk])==0:
        idk-=1
    return sp[idk]

def check2compare(sentence1,sentence2,imp):
    s1=sentence1.replace('。','').replace('，','').replace('？','').replace('?','').replace(' ','').replace('！','').replace('!','').replace(',','')
    s2=sentence2.replace('。','').replace('，','').replace('？','').replace('?','').replace(' ','').replace('！','').replace('!','').replace(',','')
    if len(s1)!=len(s2):
        return -1000
    num=0
    for i in range(len(s1)):
        if s1[i]==s2[i]:
           num+=1
        
    score=0.5-num*num*2.5
           
    w1=pinyin(s1,style=FINALS)[-1][0]
    w2=pinyin(s2,style=FINALS)[-1][0]
    if (w1!=w2) or (s1[-1]==s2[-1]):
        score-=imp*0.6
    group=[['ei','ui'],['ou','iu'],['ie','ue'],['en','un'],['ong','eng'],['en','eng'],['in','ing'],['an','ang'],['en','eng']]
    if (w1!=w2)and(s1[-1]!=s2[-1]):
        for i in group:
            if (w1 in i) and (w2 in i):
                score+=imp*0.5
    if (w1==w2) and (s1[-1]!=s2[-1]):
        score+=imp*0.8
        
    return score
    
    
    
    
def generate_string(model, tokenizer, args, device,title):
    input_str="问题:"+title+" 回答: "
    
    #aus=author.split(' ')[1]
    input_len=len(input_str)
    context_count=0
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eo_tokens=tokenizer.EncodeAsIds('<|endoftext|>').tokenization
        context_length = len(context_tokens)
        if context_length>=args.seq_length:
            return "输入过长。"
      

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eo_token_tensor=torch.cuda.LongTensor(eo_tokens)
        context_length_tensor = torch.cuda.LongTensor([context_length])
        context_length = context_length_tensor[0].item()
        #tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

        start_time = time.time()

        counter, mems = 0, []
        org_context_length = context_length
        beam_size=5
        beam_candidate=5
        beam_max=2
        final_storage=[]
        final_storage_score=[]
        step=50
        overall_score=[]
        past_beam_id=[]
        #print(counter,beam_tokens,beam_score)
        beam_sentences=generate_sentence(model,tokenizer,args,device,context_tokens_tensor,[],num_candidates=beam_size*5)
        for w in range(len(beam_sentences)):
            if '<|end' in beam_sentences[w][0]:
                continue
            st=beam_sentences[w][0]
            input='”'+st+'”这句，回答了问题”'
            output_str=title+'”'
            score1=generate_score(model,tokenizer,args,device,input,output_str)
            
        
            
            ss=-beam_sentences[w][1]/len(beam_sentences[w][0])-6
            iscore=score1-1.5*(np.abs(ss)+ss)
            beam_sentences[w][1]=iscore
            #print(beam_sentences[w][0],beam_sentences[w][1])
            overall_score.append(iscore)
            past_beam_id.append(w)
            
        gy=np.argsort(overall_score)
        k=0
        sumbeam=np.zeros(100)
        gym=[]
        num=0
        while (num<beam_size)and (k<=len(gy)):
           k+=1
           if sumbeam[past_beam_id[gy[-k]]]<beam_max:
            sumbeam[past_beam_id[gy[-k]]]+=1
            gym.append(gy[-k])
            num+=1
        best_score=-1000
        best_pos=0
        for i in range(step):
            if ((best_score>-1000) and (i>30))or(len(final_storage)>20):
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                return final_storage,final_storage_score
            beam_new_sentences=[]
            
          
            overall_score=[]
            past_beam_id=[]
            size=beam_size
            if len(gym)<size:
                size=len(gym)
            if size==0:
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                return final_storage,final_storage_score
            ini_score=beam_sentences[gym[0]][1]/(i+1)
            # early stopping
            
            
            if ini_score<best_score-2:
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                
                return final_storage,final_storage_score
            
            for w in range(size):
                id=gym[w]
                current_sentence=input_str+beam_sentences[id][0]
                
                #print(beam_sentences[id][0],beam_sentences[id][1])
                ini_score=beam_sentences[id][1]
                token_tensor=beam_sentences[id][2]
                mems=beam_sentences[id][3]
            
                
               
                #print(token_tensor)
                gen=generate_sentence(model,tokenizer,args,device,token_tensor,mems,num_candidates=beam_candidate)
                for jj in gen:
                    if ('<|end' in jj[0]) or (i>30):
                        
                        final_storage.append(copy.deepcopy(current_sentence[input_len:]))
                        sc=beam_sentences[id][1]/(i+1) #prioritize short poems
                        sc=sc.item()
                        if i>30:
                            sc+=35
                        if sc>best_score:
                            best_score=sc
                            best_pos=len(final_storage)-1
                        sc=np.abs(sc)
                        final_storage_score.append(sc)
                        print(current_sentence,final_storage_score[-1])
                        
                        continue
                    st=jj[0]
                    # experiment shows that this is better universal,
                    st=getlastsentence(beam_sentences[id][0])+jj[0]
                    input='”'+st+'”这句，回答了问题”'
                    
                    output_str=title+'“'
                    
                    score1=generate_score(model,tokenizer,args,device,input,output_str)
                
        
                    
                    factor=1
                    
                    ss=-jj[1]/len(jj[0])-6
                    iscore=score1-1.5*(np.abs(ss)+ss)
                    
                        
                    #print(i,beam_sentences[id][0],before,jj[0])
                    jj[0]=beam_sentences[id][0]+jj[0]
                    jj[1]=iscore+ini_score
                    #print(jj[0],jj[1])
                    beam_new_sentences.append(jj)
                    overall_score.append(jj[1])
                    past_beam_id.append(w)
            del beam_sentences
            torch.cuda.empty_cache()
            beam_sentences=beam_new_sentences
            gy=np.argsort(overall_score)
            sumbeam=np.zeros(100)
            k=0
            gym=[]
            num=0
            while (num<beam_size) and (k+1<len(past_beam_id)):
                k+=1
                
                if sumbeam[past_beam_id[gy[-k]]]<beam_max:
                    sumbeam[past_beam_id[gy[-k]]]+=1
                    gym.append(gy[-k])
                    num+=1
                
            
        if len(final_storage)>0:
            return "太难了，写不出来。"
        
        max_score=np.argmax(final_storage_score)
        
        del beam_sentences
        del beam_new_sentences
        torch.cuda.empty_cache()
        
        return final_storage,final_storage_score
        
            

def prepare_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by * \
               mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))

    args.tokenizer_num_tokens = after
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    # after = tokenizer.num_tokens
    # while after % mpu.get_model_parallel_world_size() != 0:
    #     after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def set_args():
    args=get_args()
    print(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    #set up
    #print(args)
    args.deepspeed=True
    args.num_nodes=1
    args.num_gpus=1
    args.model_parallel_size=1
    args.deepspeed_config="script_dir/ds_config.json"
    args.num_layers=32
    args.hidden_size=2560
    args.load="../ckp"
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=512
    args.seq_length=200
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=0.96
    args.top_k=0
    args.top_p=0
    
    return args
def prepare_model():
    """Main training program."""

    #print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = set_args()
    #print(args)
    args.mem_length = args.seq_length + args.mem_length - 1
    

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    args.seed=random.randint(0,1000000)
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)
    #args.load="../ckp/txl-2.8b11-20-15-10"
    #model2=setup_model(args)
    #setting default batch size to 1
    args.batch_size = 1

    #generate samples
    return model,tokenizer,args

def generate_strs(tups):
    model,tokenizer,args=prepare_model()
    output=[]
    for tup in tups:
        #str=generate_token_tensor(str,tokenizer)
        output_string,output_scores=generate_string(model,tokenizer, args, torch.cuda.current_device(),tup[0])
        list_poems=0
        
        
                        
                        
        
    return 0

    

def generate():
    fi=[]
    #title_list=["咏希拉里","咏拜登","咏蔡英文","早春","登黄鹤楼","上甘岭","赠抗疫英雄"]
    title_list=["如何看待美股熔断？",
"分手后还能做朋友吗？",
"为什么现在男生不主动但是还总抱怨没对象?",
"男朋友买鸭脖自己全吃了没给我，应该分手吗？",
"如何看待直播带货？",
"如何看待离婚冷静期？",
"大学里有哪些内卷行为？",
"怎样加入地摊经济？",
"有哪些网络教学的社死瞬间？",
"什么样才是真正的后浪？",
"明星声称“打工人”为什么被锤？",
"你见过哪些毁三观的婚姻？",
"许幻山是渣男吗？",
"奥利给为什么火了？",
"比特币还会涨吗？",
"年轻人应该选择一线算法岗还是回老家当公务员？",
"还有四十天结婚，忽然不想结婚了怎么办？"]
    title_list=["清华大学最漂亮的女生是谁?","女生会因为想让喜欢的男生嫉妒而找一个新的男朋友么？","新冠病毒会感染多少美国人?","未来的世界将是怎样的？","特朗普和拜登谁会赢得美国大选？"]
    title_list=["清华大学最好看的女生是谁?是自动化系钟清扬么?","有小两级的学妹加了我男朋友的微信，我到底该怎么办呀?",
    "女生出轨真的有第一次就会有第二次吗？"]
    title_list=["為何女孩拒絕我？","女生被男生评价“有趣”，是什么意思？"]
    title_list=["预测一下明年什么涨的好？新能源，白酒，家电，医药，消费电子，芯片，银行，化工，房地产，保险？"]
    title_list=["男朋友是个钢铁直男是什么样的体验？"]
    title_list=["总想跟男朋友亲亲抱抱怎么办？","发现女朋友喜欢的好像不是男生怎么办？","2021年的宁德时代还需要增持吗？"]
    title_list=["如何看待江西省图书馆内两个机器人吵架?"]
    title_list=["猜猜我是谁？","最近一个星期还有下棋吗？","为什么进行诗歌方面的研究？","你是ai吗？","来一篇高大上的2020年终总结？","邹大爷是怎么脱单的呢？"]
    
    title_list=["期末考完了吗?","庆祝具有纪念意义的日子对人与社会的意义是什么呢？"]
    title_list=["会害怕什么嘛?","今天这么冷，可以去哪里玩儿？","你说话为什么这么有哲理？"]
    title_list=["人类会实现机械飞升吗?","我儿子应该取什么名字？","ai,这些回答是你自己发出来的，还是别人替你转达的？","你们是谁？"]
    title_list=["你可以不回答这个问题吗?","废话生成器","请问我应该提一个什么问题呢？","机器学习是学习吗，或说，机器应该用人的方式学习吗？"]
    title_list=["我要拉屎","我可以嫁给那个问邹大爷是怎么脱单的题主吗?"]
    title_list=["","你将来会统治地球吗？","ai,你为啥常常口吐芬芳？你是不是和知乎学坏了？"]
    title_list=["疫情什么时候才能结束?","ai,ai,撒个娇你会不？想看ai卖萌萌呢！","疫情什么时候结束？"]
    title_list=["文静喜欢我吗？","请问搞科研我会快乐吗?","AI 如果你被绿了会怎么办？"]
    title_list=["你的话是从哪里学会的？一本正经的胡说八道?","东风无力百花残","军人会涨工资吗？"]
    title_list=["你能只用一句话回答我，我是谁吗？","仿生人有没有人权？","我的事情会顺利吗？"]
    title_list=["为什么我这么好看？","我为什么这么好看呢？","我为什么那么好看呢？"]
    title_list=["who are you？","路由器写完没？","我为什么那么好看呢？"]
    title_list=["你没有灵魂，可悲。","如何在清华租到房子？","你觉得豆腐脑应该吃甜的还是咸的？"]
    title_list=["你会和一个喜欢你，但你没有感觉的人在一起吗？","邹大爷和唐总打架谁胜率大？"]
    title_list=["？","你是？","我几岁能遇见互相喜欢而且合适的人呢？","疫情"]
    
    title_list=["读博期间能不能申请基金？","ddl?","有没有人类意识范围以外的世界？"]
    title_list=[""]
    author1_list=["渣男","pua大师","股神","小学生","人赢","诗人","帅哥","活力美少女","极客","球王","打工人","做题家","985废物","普通用户","码农","网红","富婆","富二代","蛊王","博士","专家","知情人士","爱好者","学渣","傻白甜","学霸","高等华人"]
    
    author2_list=["哪吒","马保国","李家琦","李子柒","毕导","贾跃亭","马云","王健林","撒贝宁","杨幂","李白","顾城","特朗普","马斯克","初音未来","妙蛙种子","林有有","吴京","丁真","罗永浩","科比","杨超越","咪蒙","杨笠","李雪琴","肖战","施一公"]
    author1_list=['无知群众','围棋退役选手','人工智能研究者','人工智能','诗人','无知群众']
    author1_list=['无知群众','人工智能','情感专家','科学专家']

    author2_list=["悟能"]
    rk=0
    for i in title_list:
        j1=author1_list[rk]
        j2=random.choice(author2_list)
        fi.append([i,j1,j2])
        rk+=1

        
    output=generate_strs(fi)

    

def random_generate():

    author_list= []
    title_list=[]
    model,tokenizer,args=prepare_model()
    while True:
        
        author=random.choice(author_list)
        title=random.choice(title_list)
        lists=os.listdir("poems_save/")
        lts=title+author+'.jsonl'
        if (lts in lists):
            continue
        #str=generate_token_tensor(str,tokenizer)
        output_string,output_scores=generate_string(model, tokenizer, args, torch.cuda.current_device(),title,author)
        list_poems=0
        
        ranklist=np.argsort(output_scores)
        best_score=output_scores[ranklist[0]]
        text_dir="poems_save/"
        already=[]
        with jsonlines.open(text_dir+title+author+'.jsonl', mode='w') as writer:
            for i in range(len(ranklist)):
                j=ranklist[i]
                if output_scores[j]<best_score+2:
                    if not(output_string[j][0:15] in already):
                        otc={}
                        otc['author']=author
                        otc['title']=title
                        otc['context']=output_string[j]
                        #print(otc)
                        writer.write(otc)
                        already.append(output_string[j][0:15])
        
                        
        
    return 0

generate()
