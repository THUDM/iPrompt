# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for generating text."""

import copy
import json
import os
import time

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import get_tokenizer
from megatron import mpu
from megatron.training import communicate
from megatron.utils import get_ltor_masks_and_position_ids
import numpy as np
import pynvml

def get_batch(context_tokens):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.view(args.micro_batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def generate_samples_input_from_file(model):

    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
   
    fname = open(args.sample_input_file, "r")
    all_raw_text = fname.readlines()
    input_count = len(all_raw_text)
    input_pos = 0
    if args.sample_output_file is None:
        sample_output_file = args.sample_input_file + ".out"
        print('`sample-output-file` not specified, setting '
                  'it to {}'.format(sample_output_file))
        
    fname_out = open(sample_output_file, "w+")

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
               and mpu.get_tensor_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                input_pos += 1
                if input_pos == input_count:
                    raw_text = "stop"
                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(input_info_tensor,
                                         group=mpu.get_model_parallel_group())
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if mpu.get_tensor_model_parallel_rank() == 0 \
               and args.pipeline_model_parallel_size > 1:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(context_length,
                                                        dtype=torch.int64,
                                                        device=torch.device("cuda"))
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):
                pass

            if mpu.get_tensor_model_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():
                    os.system('clear')
                    print("\nContext:", raw_text, flush=True)

                    fname_out.write("\nContext:")
                    fname_out.write(raw_text)

                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                    trim_decode_tokens = tokenizer.detokenize(
                        decode_tokens)[raw_text_len:]
                    print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                    fname_out.write("\n\nMegatron-LM:")
                    fname_out.write(trim_decode_tokens)
                    fname_out.write("\n")

            raw_text = None
            context_count += 1

def generate_one_text(model,tokenizer,args,input):
    model.eval()
    with torch.no_grad():
        raw_text_len=len(input)
        context_tokens=tokenizer.tokenize(input)
        context_length = len(context_tokens)
        #print(context_length,context_tokens)
        input_info = [0, raw_text_len, context_length]
        input_info_tensor = torch.cuda.LongTensor(input_info)
        terminate_runs = 0
        
        
        
        context_tokens, context_lengths = pad_batch([context_tokens],
                                                tokenizer.eod, args)
        #print(args.seq_length)
        #print(context_lengths,context_tokens)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        
        eos_id = tokenizer.eod
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
       
        layer_past=None
        counter=0
        is_done=torch.zeros([1]).byte().cuda()
        done=False
        while (counter<args.seq_length-context_length and (done==False)):
            #only the recompute runs correctly
            
            output, layer_past = model( tokens,position_ids,
                                                  attention_mask,
                                                  layer_past=None,
                                                  get_key_value=True)
                
            #logits = output[:, -1].view(batch_size, -1).contiguous()
                
            logits = output[:, context_length - 1, :]
            logits = logits.float()
            logits /= args.temperature
            logits = top_k_logits(logits, top_k=args.top_k,
                                          top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1).view(-1)
            #print(prev)
            tokens[:, context_length] = prev.view(-1)
            done_token = (prev == eos_id).byte()
            is_done = is_done | done_token

            done = torch.all(is_done)
            context_length+=1
            counter+=1
        decode_tokens=tokens[:,:context_length]
        decode_tokens = decode_tokens[0].cpu().numpy().tolist()
        trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
        score=calculate_perplexity(model,tokenizer,args,input,trim_decode_tokens)
        print(score)
        return trim_decode_tokens
            
        
                

def generate_samples_interactive(model, print_frequency=24):

    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            
            os.system('clear')
            raw_text = input("\nContext prompt (stop to exit) >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("\nContext prompt (stop to exit) >>> ")
            raw_text_len = len(raw_text)

            if "stop" in raw_text:
                terminate_runs = 1
            else:
                context_tokens = tokenizer.tokenize(raw_text)
                context_length = len(context_tokens)

                if context_length >= (args.seq_length // 2):
                    print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                    continue
            
            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            
          
            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            #print(context_tokens_tensor)
            token_stream = get_token_stream(model, [context_tokens])

            for counter, decode_tokens in enumerate(token_stream):
                if counter % print_frequency != 0:
                    continue

                os.system('clear')
                print("\nContext:", raw_text, flush=True)

                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

           
            os.system('clear')
            print("\nContext:", raw_text, flush=True)

            if not isinstance(decode_tokens, list):
                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
            trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[raw_text_len:]
            print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            input("\nPress Enter to continue >>>")

            raw_text = None
            context_count += 1



def generate_samples_unconditional(model):

    args = get_args()
    tokenizer = get_tokenizer()

    num_samples = args.num_samples
    context_tokens = [[tokenizer.eod]
                      for _ in range(args.micro_batch_size)]
    ctr = 0
    while True:
        start_time = time.time()
        for token_stream in get_token_stream(model,
                                             copy.deepcopy(context_tokens)):
            pass
        if mpu.is_pipeline_last_stage() and \
           mpu.get_tensor_model_parallel_rank() == 0:
            if ctr % args.log_interval == 0:
                print('Avg s/batch:',
                      (time.time() - start_time) / min(args.log_interval, ctr + 1))
                start_time = time.time()
            length = len(token_stream)
            token_batch = token_stream[0].cpu().numpy().tolist()
            length_batch = token_stream[1].cpu().numpy().tolist()
            assert len(length_batch) == args.micro_batch_size
            for tokens, length in zip(token_batch, length_batch):
                tokens = tokens[1:length - 1]
                text = tokenizer.detokenize(tokens)
                is_finished = length < args.seq_length - 1
                datum = {'text': text, 'length': length - 1, 'finished': is_finished}
                yield datum
                ctr += 1
                if ctr >= num_samples:
                    break
        else:
            for _ in range(args.micro_batch_size):
                yield None
                ctr += 1
                if ctr >= num_samples:
                    break
        if ctr >= num_samples:
            break


def generate_and_write_samples_unconditional(model):

    args = get_args()
    assert args.genfile is not None
    with open(args.genfile, 'w') as f:
        for datum in generate_samples_unconditional(model):
            if mpu.is_pipeline_last_stage() and \
               mpu.get_tensor_model_parallel_rank() == 0:
                f.write(json.dumps(datum) + '\n')


def pad_batch(batch, pad_id, args):

    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def get_token_stream(model, context_tokens):

    args = get_args()
    tokenizer = get_tokenizer()

    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eod, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor,
                                                 context_length_tensor,
                                                 attention_mask, position_ids)
    for tokens, lengths in batch_token_iterator:
        context_length += 1
        if tokens is not None:
            yield tokens[:, :context_length], lengths
        else:
            yield None, None


def switch(val1, val2, boolean):

    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_step(model, tokens, position_ids, attention_mask, tokentype_ids,
                 layer_past=None, get_key_value=None,
                 forward_method_parallel_output=None):

    # Hidden size changes when not using recompute, need to tell communicate()
    # the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]
    #print(tokens,position_ids,attention_mask)
    output_tensor = model(tokens, position_ids, attention_mask,
                                  tokentype_ids=tokentype_ids,
                                  layer_past=layer_past)
       
    args.seq_length = orig_seq_length
    if get_key_value:
        return output_tensor, layer_past
    return output_tensor,layer_past


def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          maxlen=None, type_ids=None):

    args = get_args()
    tokenizer = get_tokenizer()

    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        eos_id = tokenizer.eod

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = args.seq_length - 1
            if maxlen > (org_context_length + args.out_seq_length):
                maxlen = org_context_length + args.out_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length <= (maxlen):
            if args.recompute:
                output,layer_past = forward_step(model, tokens,
                                      position_ids,
                                      attention_mask,
                                      tokentype_ids=type_ids,
                                      get_key_value=True,
                                      forward_method_parallel_output=False)
               
                logits = output[:, context_length - 1, :]
            else:
                types2use = None
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                    if type_ids is not None:
                        types2use = type_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length - 1].view(
                        batch_size, -1)
                    positions2use = position_ids[:, context_length - 1].view(
                        batch_size, -1)
                    if type_ids is not None:
                        types2use = type_ids[:, context_length - 1].view(
                            batch_size, -1)
                output, layer_past = forward_step(model, tokens2use,
                                                  positions2use,
                                                  attention_mask,
                                                  layer_past=layer_past,
                                                  get_key_value=True,
                                                  tokentype_ids=types2use,
                                                  forward_method_parallel_output=False)
                
                logits = output[:, -1].view(batch_size, -1).contiguous()

            if mpu.is_pipeline_last_stage():
                if args.greedy:
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= args.temperature
                    logits = top_k_logits(logits, top_k=args.top_k,
                                          top_p=args.top_p)
                    log_probs = F.softmax(logits, dim=-1)
                    prev = torch.multinomial(log_probs, num_samples=1).view(-1)

                started = context_lengths <= context_length

                new_tokens = switch(
                    tokens[:, context_length].view(-1), prev, started)
                tokens[:, context_length] = new_tokens
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                done_token = (prev == eos_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                yield tokens, lengths

            else:
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_last_rank()
                    group = mpu.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None
                else:
                    yield None, None

                done = torch.cuda.ByteTensor([0])
                src = mpu.get_pipeline_model_parallel_last_rank()
                group = mpu.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break


def calculate_perplexity(model,tokenizer,args,input,output):
    model.eval()
    with torch.no_grad():
        raw_text_len=len(input)
        context_tokens=tokenizer.tokenize(input)
        output_context_tokens=tokenizer.tokenize(output)
        output_length=len(output_context_tokens)
        context_length = len(context_tokens)
        #print(context_length,context_tokens)
        input_info = [0, raw_text_len, context_length]
        input_info_tensor = torch.cuda.LongTensor(input_info)
        terminate_runs = 0
        
        
        
        context_tokens, context_lengths = pad_batch([context_tokens],
                                                tokenizer.eod, args)
        #print(args.seq_length)
        #print(context_lengths,context_tokens)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        output_tokens_tensor=torch.cuda.LongTensor([output_context_tokens])
        
        eos_id = tokenizer.eod
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
       
        layer_past=None
        counter=0
        is_done=torch.zeros([1]).byte().cuda()
        done=False
        score=0
        while (counter<output_length and (done==False)):
            #only the recompute runs correctly
            
            output, layer_past = model( tokens,position_ids,
                                                  attention_mask,
                                                  layer_past=None,
                                                  get_key_value=True)
                
            #logits = output[:, -1].view(batch_size, -1).contiguous()
                
            logits = output[:, context_length - 1, :]
            logits = logits.float()
            log_probs = F.softmax(logits, dim=-1)
            prev=output_context_tokens[counter]
            log_num=torch.log(log_probs).data
            score+=log_num[0,prev]
            #print(prev)
            tokens[:, context_length] = output_tokens_tensor[:,counter]
            
            context_length+=1
            counter+=1

        return score
        

def checksentence(sentence,original_context,min_length,max_length,endnote):
    if "<|end" in sentence:
        return 0
            
    if ((len(sentence)>max_length and not(sentence[-1] in endnote)) or len(sentence)==0) or len(sentence)>max_length+1:
        return 1
    if (sentence[-1] in endnote)and (len(sentence)<=min_length):
        return 1
        
    if (len(sentence)>15):
        for i in range(15,len(sentence)):
            if sentence[i-15:i] in original_context:
                return 1
    
    if (sentence[-1] in endnote):
        return 0
        
        
    return 2

def generate_sentence(model,tokenizer,args,input,endnote=[",",".","!",":","?","\n",">"],num_candidates=10,min_length=2,max_length=200):

    
    model.eval()
    #pynvml.nvmlInit()
    with torch.no_grad():
        #index=len(tokens[0])
        raw_text_len=len(input)
        context_tokens=tokenizer.tokenize(input)
        context_length = len(context_tokens)
        
        context_tokens, context_lengths = pad_batch([context_tokens],
                                                tokenizer.eod, args)
       
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        
        
        mct_tree=[]
        
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
        logits= model(tokens, position_ids, attention_mask, layer_past=None)
                                                            
        
        logits=logits[:,context_length-1,:]
        logits = top_k_logits(logits, top_k=args.top_k,
                                          top_p=args.top_p)
        # add "context length" to specify where the generated answer is.
        ll=logits.view(-1)
        #print(logits)
        mct_tree.append([logits,tokens,-np.ones(len(ll)),torch.ones(len(ll)).cuda(),context_length,0])
        #print(logits.shape)
        final_result=[]
        nextid=0
        tries=0
        max_tries=num_candidates*30
        max_tries=150
        while (len(final_result)<num_candidates)and(tries<max_tries):
            currentid=nextid
            tries+=1
            #print(tries,len(mct_tree),len(ll))
            while currentid!=-1:
                tc=torch.log(mct_tree[currentid][3])
                tc=tc+F.relu(tc-10)*1000
                logits=mct_tree[currentid][0].view(-1)-tc*0.5
                log_probs = F.softmax(logits/args.temperature, dim=-1)
              
                pr=torch.multinomial(log_probs,num_samples=1)
                prev=pr[0].item()
                mct_tree[currentid][3][prev]+=1
                lastid=currentid
                currentid=int(mct_tree[currentid][2][prev])
        
            cqs=mct_tree[lastid][1]
            clen=mct_tree[lastid][4]
            cqs[0,clen]=pr
            output_tokens_list = cqs.view(-1).contiguous()
            sentence = tokenizer.detokenize(output_tokens_list[context_length:clen+1].tolist())
            logit=mct_tree[lastid][0]
            log_probs = F.softmax(logit, dim=-1)
            log_pbs=torch.log(log_probs)
            score=log_pbs[0,prev].item()
            nextid=0
        
            ip=checksentence(sentence,input,min_length,max_length,endnote)
            if clen>950:
                ip=1
            for j in final_result:
                if j[0]==sentence:
                    ip=1
                if ('<|end' in sentence) and ('<|end' in j[0]):
                    ip=1
                    
            score=mct_tree[lastid][5]+score
            if (ip==1):
                mct_tree[lastid][3][prev]=100000
                continue
            if (ip==0):
                mct_tree[lastid][3][prev]=100000
                final_result.append([copy.deepcopy(sentence),copy.deepcopy(score),clen-context_length+1])
               # meminfo = pynvml.nvmlDeviceGetMemoryInfo(0)
                #print(len(final_result),sentence,score)
                continue

            
                #calculate
            mct_tree[lastid][2][prev]=len(mct_tree)
            
            logits=model(cqs,position_ids, attention_mask, layer_past=None)
            
            logits=logits[:,clen,:]
            logits = top_k_logits(logits, top_k=args.top_k,
                                          top_p=args.top_p)
            mct_tree.append([logits,cqs,-np.ones(len(ll)),torch.ones(len(ll)).cuda(),clen+1,score])
            
            torch.cuda.empty_cache()
            nextid=len(mct_tree)-1
        del mct_tree
        torch.cuda.empty_cache()
        #print(tries,len(final_result))
        return final_result

def getlastsentence(str):
    signal=[',','!','?','\n','.']
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
            if nowplace>70:
                return str[-nowplace+1:]
        
    return 0
    
def generate_string(model, tokenizer, args,input):
    input_str=input
    
    input_len=len(input_str)
    context_count=0
    model.eval()
    with torch.no_grad():
        
        start_time = time.time()

        beam_size=5
        beam_candidate=10
        beam_max=2
        final_storage=[]
        final_storage_score=[]
        step=50
        overall_score=[]
        past_beam_id=[]
        #print(counter,beam_tokens,beam_score)
        beam_sentences=generate_sentence(model,tokenizer,args,input_str,num_candidates=beam_size*5)
        #print(beam_sentences)
        for w in range(len(beam_sentences)):
            if '<|end' in beam_sentences[w][0]:
                continue
            st=beam_sentences[w][0]
            input="\""+st+"\" discusses the topic \""
            output_str=input+"\""
            score1=calculate_perplexity(model,tokenizer,args,input,output_str)
            
            ss=-beam_sentences[w][1]/beam_sentences[w][2]-10
            iscore=score1-0.25*(np.abs(ss)+ss)
            
            beam_sentences[w][1]=iscore
            overall_score.append(iscore.cpu())
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
        best_score=-2500
        best_pos=0
        for i in range(step):
            if ((best_score>-2500) and (i>30))or(len(final_storage)>20):
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
           
            for w in range(size):
                id=gym[w]
                current_sentence=input_str+beam_sentences[id][0]
                
                #print(beam_sentences[id][0],beam_sentences[id][1])
                ini_score=beam_sentences[id][1]
               

                gen=generate_sentence(model,tokenizer,args,current_sentence,num_candidates=beam_candidate)
                for jj in gen:
                    if ('<|end' in jj[0]) or (i>30):
                        
                        final_storage.append(copy.deepcopy(current_sentence[input_len:]))
                        sc=beam_sentences[id][1]/(i+1)
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
                    input="\""+st+"\" discusses the topic \""
                    
                    output_str=input_str+"\""
                    
                    score1=calculate_perplexity(model,tokenizer,args,input,output_str)
                
                    
                    factor=1
                    
                    ss=-jj[1]/jj[2]-10
                    iscore=score1-0.25*(np.abs(ss)+ss)
                    
                        
                    #print(i,beam_sentences[id][0],before,jj[0])
                    jj[0]=beam_sentences[id][0]+jj[0]
                    jj[1]=iscore+ini_score
                    #print(jj[0],jj[1])
                    beam_new_sentences.append(jj)
                    overall_score.append(jj[1].cpu())
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
