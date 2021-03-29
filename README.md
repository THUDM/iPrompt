# InversePrompting

# Paper: Controllable Generation from Pre-trained Language Models via Inverse Prompting

Code:
The code is provided in the "chinese_ip" and "english_ip" package.

Chinese Inverse Prompting:

based on https://github.com/THUDM/Chinese-Transformer-XL

Packages Required
<pre>
torch,apex,boto3,sentencepiece,nltk,jsonlines,filelock,deepspeed,pypinyin,pandas
</pre>

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

based on megatron-lm https://github.com/NVIDIA/Megatron-LM, follow its guide to download model weights and put them under the correct path, then run
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

There's also a demo where you can try your own questions/titles for QA/poem generation.

QA:
https://pretrain.aminer.cn/app/qa

Poem Generation: 
https://pretrain.aminer.cn/apps/poetry.html

Note that the demo version is updating frequently and may be different from the repo version. 

Some examples of poems it generates:

<pre>
咏特朗普

天下岂有华盛顿,外强中衰叹累累。
白宫总统成陪衬,螳臂挡车虎尾寒。
坐观美国朝野势,风雨飘摇现暴难。
拜登再任难抵挡,明年恐将命归残。
</pre>

<pre>
夜过虹桥机场 

卢浦斜晖里,西楼醉客行。
影侵双塔晚,灯落一城明。
空客还频顾,航灯未可惊。
空留城市夜,月映水帘星。
</pre>

<pre>
排队购房作 

向晚万人候,售楼幢馅齐。
验资堪买主,瞧室亦堪栖。
回柱瞻佳处,连楼仰远姿。
殷勤申买者,莫待扣扉期。
</pre>

<pre>
论资本主义 

若为自由故,如今逐利逃。
入城操法律,两股战空槽。
漂白藏珠玉,欢呼夺锦袍。
管窥矜势利,夸视堕尘劳。
</pre>

<pre>
赠美国友人

清远寄吴士,华州逢旧知。
大洋环万里,学馆阻三时。
道别殷勤意,地连海峤西。
同来艰运日,异域远风姿。
</pre>

<pre>
安克雷奇中美会谈

特务狂声振,朗官降虏庭。
普天皆窃笑,攻守几无惊。
入市商人拜,国殇将士迎。
会同诛狡寇,世界定清明。
</pre>



If you have any questions, please contact zoux18@mails.tsinghua.edu.cn 

Please cite
<pre>
@article{zou2021controllable,
  title={Controllable Generation from Pre-trained Language Models via Inverse Prompting},
  author={Zou, Xu and Yin, Da and Zhong, Qingyang and Yang, Hongxia and Yang, Zhilin and Tang, Jie}, 
  journal={arXiv preprint arXiv:2103.10685},  
  year={2021}  
}
</pre>
