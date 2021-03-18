# InversePrompting

Code:
The code is provided in the code package.

Train:
scripts/ds_pretrain_gpt2_29B.sh

Generate:
scripts/generate_text.sh

Generate Poems:
generate_pms_refined.py  --Inverse Prompting +Reinforcement Learning for TCP Generation

QA:
generate_qa_desc.py  --Inverse Prompting for QA


Data:

Pre-trained Model: 
To be released 


Generated TCPs:

jiuge:data/poems_jiuge.jsonl
jiuge generated from http://jiuge.thunlp.org/

IP+RL: data/poems_ip_rl.zip
IP-only: data/poems_ip_norl.zip
Base Model: data/poems_noip.zip

QAs:

CPM: data/qa_cpm.zip
IP: data/qa_ip.zip
base model: data/qa_basemodel.zip
Human: data/qa_human.jsonl

Human Evaluation Raw Data (results listed in paper): 

based on evaluator: data/user-records.jsonl
based on prompts:
QA: data/qa-records.jsonl
poem: data/poem-records.jsonl






