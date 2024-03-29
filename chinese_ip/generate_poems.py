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

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from pretrain_gpt2 import get_model

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
            print('0')
            iteration, release, success = get_checkpoint_iteration(args)
            print('1')
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            print('2')
            checkpoint = torch.load(path)
            print('3')
            model.load_state_dict(checkpoint["module"])
            print('4')
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
def generate_sentence(model,tokenizer,args,current_tokens,mems,available_endnote=[",","，","?","？"],num_candidates=10,min_length=4,max_length=7):
    with torch.no_grad():
        output_tokens_list = current_tokens.view(-1).contiguous()
        original_context=tokenizer.DecodeIds(output_tokens_list.tolist())
        context_length=len(original_context)
        
        index=len(current_tokens[0])
        mct_tree=[]
        logits, *rts = model(current_tokens[:, index - 1: index], beam_tokens[w].new_ones((1, 1)) * (index - 1),
                        beam_tokens[w].new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                                dtype=torch.float), *mems)
        
        #mct_structure=-np.ones(len(logits))
        mct_tree.append([logits,rts,current_tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),0])
        
        final_result=[]
        while len(final_result)<num_candidates:
            currentid=0
            while currentid!=-1:
                logits=mct_tree[currentid][0]-torch.log(mct_tree[currentid][0][4])
                prev=torch.argmax(logits)[0]
                mct_tree[currentid][4][prev.data[0].cpu().numpy()]+=1
                lastid=currentid
                currentid=mct_tree[currentid][3][prev]
            #start from lastid & currentid
            cqs=mct_tree[lastid][2]
            tokens = torch.cat((cqs, prev.view(1, 1)), dim=1)
            output_tokens_list = tokens.view(-1).contiguous()
            decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
            sentence=decode_tokens[context_length:]
            
            if "<|end" in sentence:
                final_result.append([sentence,mct_tree[lastid][5]])
                print(mct_tree[lastid][5])
                continue
            
            if len(sentence)>max_length:
                continue
            if (sentence[-1] in endnote)and (len(sentence)<=min_length):
                continue
            if (sentence[-1] in endnote)and (sentence[:-1] in original_context):
                continue
            if (sentence[-1] in endnote):
                final_result.append([sentence,mct_tree[lastid][5]])
                print(sentence,score)
                continue
                #calculate
                
            rts=mct_tree[lastid][1]
            index=len(cqs[0])
            score=mct_tree[lastid].data[0][prev].cpu().numpy()
            
            logits,*rts=model(cqs[:, index - 1: index], cqs.new_ones((1, 1)) * (index - 1),
                        cqs.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                                dtype=torch.float), *rts)
            score=mct_tree[lastid][5]+score
            mct_tree.append([logits,rts,current_tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),score])
            
     return final_result
        
        
        
def checkpoem(s):  #check the score of poem
    w=s.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace('<','').replace(' ','').replace('>','').replace('《','').replace('》','')
    if ':' in w:
        return 0
    if '：' in w:
        return 0
    sentence=w.split(',')
    lengthofpoem=len(sentence[0])
    if not(lengthofpoem in [4,5,7]):
        return 0
    for i in range(len(sentence)):
        if len(sentence[i]!=lengthofpoem) and (i!=len(sentence)-1):
            return 0
        if i>=1:
            for j in range(len(sentence[i])):
                if sentence[i][j]==sentence[i-1][j]:
                    return 0
    #移除合掌，
    return 1
        
    

def generate_samples(model, tokenizer, args, device):
    
    context_count=0
    model.eval()
    output_path = "./samples"
    print("We're in.1")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("We're in.2")
    output_path = os.path.join(output_path, f"sample-{datetime.now().strftime('%m-%d-%H-%M')}.txt")
    with torch.no_grad(), open(output_path, "w") as output:
        while True:
            print("We're in.")
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs=0

            if mpu.get_model_parallel_rank() == 0:
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")
           
                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    output.write(raw_text)
                    context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
                    context_length = len(context_tokens)

                    if context_length >=args.seq_length:
                        print("\nContext length", context_length, \
                            "\nPlease give smaller context (half of the sequence length)!")
                        continue
            else:
                context_tokens = tokenizer.EncodeAsIds("EMPTY TEXT").tokenization
                context_length = len(context_tokens)
            
            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            # pad_id = tokenizer.get_command('pad').Id
            # if context_length < args.out_seq_length:
            #     context_tokens.extend([pad_id] * (args.out_seq_length - context_length))

            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            context_length_tensor = torch.cuda.LongTensor([context_length])

            torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

            context_length = context_length_tensor[0].item()
            tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

            start_time = time.time()

            counter, mems = 0, []
            org_context_length = context_length
            while counter < (args.out_seq_length - org_context_length):
                if counter == 0:
                    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
                else:
                    index = org_context_length + counter
                    logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                                          tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                          dtype=torch.float), *mems)
                logits = logits[:, -1]
                logits /= args.temperature
                logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)            
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)[0]
                tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
                context_length += 1
                counter += 1

                output_tokens_list = tokens.view(-1).contiguous()
                decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())

                is_end = prev == args.eod_token
                if mpu.get_model_parallel_rank() == 0 and (counter % 128 == 0 or is_end):
                   os.system('clear')
                   print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                   print("\nContext:", raw_text, flush=True)
                   trim_decode_tokens = decode_tokens[len(raw_text):decode_tokens.find("<|endoftext|>")]
                   print("\nGPT2:", trim_decode_tokens, flush=True)
                if is_end:
                   break
                
            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                output_tokens_list = tokens.view(-1).contiguous()
                decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
                trim_decode_tokens = decode_tokens[len(raw_text):decode_tokens.find("<|endoftext|>")]
                print("\nGPT2:", trim_decode_tokens, flush=True)
                output.write(trim_decode_tokens + "\n")
            raw_text = None

            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

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

def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    #setting default batch size to 1
    args.batch_size = 1

    #generate samples
    print('everything prepared!')
    generate_samples(model, tokenizer, args, torch.cuda.current_device())
    

if __name__ == "__main__":
    main()



