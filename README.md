# InversePrompting

# Paper: Controllable Generation from Pre-trained Language Models via Inverse Prompting

Code:
The code is provided in the "chinese_ip" and "english_ip" package.

Chinese Inverse Prompting:

edited from https://github.com/THUDM/Chinese-Transformer-XL

Train:
<pre>
bash scripts/ds_pretrain_gpt2_29B.sh
</pre>

Direct Generation:
<pre>
bash scripts/generate_text.sh
</pre>
Generate Poems:
<pre>
python generate_pms_refined.py  --Inverse Prompting for TCP Generation
</pre>
Generate QA:
<pre>
python generate_qa_desc.py  --Inverse Prompting for QA
</pre>
English Inverse Prompting: 

edited from megatron-lm, follow its guide to download model weights and put them under the correct path, then run
<pre>
python tools/generate_samples_sgpu.py --use-set 1
</pre>
for inverse prompting.

Data:

Chinese Language Model:

See https://github.com/THUDM/Chinese-Transformer-XL

English Language Model:

See https://github.com/NVIDIA/Megatron-LM

Generated TCPs:

jiuge:<pre>data/poems_jiuge.jsonl</pre>
jiuge generated from http://jiuge.thunlp.org/

IP+RL: <pre>data/poems_ip_rl.zip</pre>
IP-only: <pre>data/poems_ip_norl.zip</pre>
Base Model: <pre>data/poems_noip.zip</pre>

QAs:

CPM: <pre>data/qa_cpm.zip</pre>
IP: <pre>data/qa_ip.zip</pre>
base model: <pre>data/qa_basemodel.zip</pre>
Human: <pre>data/qa_human.jsonl</pre>

Human Evaluation Raw Data (results listed in paper): 

based on evaluator: <pre>data/user-records.jsonl</pre>
based on prompts:
QA: <pre>data/qa-records.jsonl</pre>
poem: <pre>data/poem-records.jsonl</pre>

Paper: full version of paper(generated using XeLaTeX) is included in this repo. The arXiv version uses pdflatex and tables with Chinese characters are transferred to English as pdflatex does not allow UTF-8 characters(non-English languages) presence. 
<pre>
paper.pdf
</pre>
If you have any questions, please contact zoux18@mails.tsinghua.edu.cn 

Please cite
<pre>
@article{zou2021controllable,
  title={Controllable Generation from Pre-trained Language Models via Inverse Prompting},
  author={Zou, Xu and Yin, Da and Zhong, Qingyang and Yang, Hongxia and Yang, Zhilin and Tang, Jie}, 
  journal={arXiv preprint arXiv:2103.},  
  year={2021}  
}
</pre>
