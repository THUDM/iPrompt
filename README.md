# InversePrompting

# Paper: Controllable Generation from Pre-trained Language Models via Inverse Prompting

Code:
The code is provided in the "chinese_ip" and "english_ip" package.

Chinese Inverse Prompting:

edited from https://github.com/THUDM/Chinese-Transformer-XL

Train:
scripts/ds_pretrain_gpt2_29B.sh

Direct Generation:
scripts/generate_text.sh

Generate Poems:
python generate_pms_refined.py  --Inverse Prompting for TCP Generation

Generate QA:
python generate_qa_desc.py  --Inverse Prompting for QA

English Inverse Prompting: 

edited from megatron-lm, follow its guide to download model weights and put them under the correct path, then run

python tools/generate_samples_sgpu.py --use-set 1

for inverse prompting.

Data:

Chinese Language Model:

See https://github.com/THUDM/Chinese-Transformer-XL

English Language Model:

See https://github.com/NVIDIA/Megatron-LM

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

If you have any questions, please contact zoux18@mails.tsinghua.edu.cn 

Please cite

``@article{zou2021controllable,

  title={Controllable Generation from Pre-trained Language Models via Inverse Prompting},
  
  author={Zou, Xu and Yin, Da and Zhong, Qingyang and Yang, Hongxia and Yang, Zhilin and Tang, Jie},
  
  journal={arXiv preprint arXiv:2103.},
  
  year={2021}
  
}
