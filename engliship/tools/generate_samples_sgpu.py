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

"""Sample Generate GPT"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import (GPTModel,
                            GPTModelFirstStage,
                            GPTModelLastStage,
                            GPTModelIntermediateStage)
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_sgpu import generate_one_text
from megatron.text_generation_sgpu import generate_string
import numpy as np

def model_provider():
    """Build the model."""

    print_rank_0('building GPT model ...')
    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        # Determine model based on position of stage in pipeline.
        if mpu.is_pipeline_first_stage():
            model = GPTModelFirstStage(num_tokentypes=0)
        elif mpu.is_pipeline_last_stage():
            model = GPTModelLastStage(
                num_tokentypes=0, parallel_output=False)
        else:
            model = GPTModelIntermediateStage(
                num_tokentypes=0)
    else:
        model = GPTModel(num_tokentypes=0, parallel_output=False)

    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')

    return parser


def main():
    """Main program."""
    drmode=0
    mode=0
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    # Set up model and load checkpoint.
    model = get_model(model_provider)
    args=get_args()
    tokenizer=get_tokenizer()
    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    
    # Generate samples.
    if drmode==1:
        f=open("questions.txt",'r')
        if mode==0:
            dir="qa_345M"
        else:
            dir="qa_345M_ip"
    if drmode==0:
        f=open("para.txt",'r')
        if mode==0:
            dir="pa_345M"
        else:
            dir="pa_345M_ip"
            
    qs=f.readlines()
    question_list=[]
    import json
    for i in qs:
        question_list.append(i)
    f.close()
    fdir=os.listdir()
    
    if not(dir in fdir):
        os.mkdir(dir)
    import random
    import jsonlines
    while True:
        
        q=random.choice(question_list)
        lists=os.listdir(dir)
        question=q
        lts=question[:20]+'.jsonl'
        if (lts in lists):
            continue
        #str=generate_token_tensor(str,tokenizer)
        
        if mode==0:
            output_string=generate_one_text(model, tokenizer, args,question)
            print(question,output_string)
            
            text_dir=dir+"/"
            already=[]
            with jsonlines.open(text_dir+question[:20]+'.jsonl', mode='w') as writer:
                
                otc={}
                otc['question']=question
                otc['answer']=output_string
                            #print(otc)
                writer.write(otc)
        else:
            output_string,output_scores=generate_string(model, tokenizer, args,question)
            ranklist=np.argsort(output_scores)
            best_score=output_scores[ranklist[0]]
            text_dir=dir+"/"
            already=[]
            with jsonlines.open(text_dir+question[:20]+'.jsonl', mode='w') as writer:
                
                otc={}
                otc['question']=question
                otc['answer']=output_string[ranklist[0]]
                            #print(otc)
                writer.write(otc)
            
                        

if __name__ == "__main__":

    main()
