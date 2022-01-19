# InversePrompting

# Paper: Controllable Generation from Pre-trained Language Models via Inverse Prompting

# For Faster Implementation (10x speed and higher quality and support English with a 10b model), see https://github.com/THUDM/GLM-iprompt

Code:
The code is provided in the "chinese_ip" and "english_ip" package.

Chinese Inverse Prompting:

based on https://github.com/THUDM/Chinese-Transformer-XL

Packages Required
<pre>
torch,apex,boto3,sentencepiece,nltk,jsonlines,filelock,deepspeed=0.3.16,pypinyin,pandas
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
比特币

外挖无穷洞,机神犹未休。
卡中窥币影,池里验沙流。
屡载吸金主,孤深渍盗求。
方知区块链,本是古来游。
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
杨振宁院士百岁生日

星河耿耿落辉英,无限苍穹任去行。
宇宙旋还成壮史,三光断续又分明。
青榆绿水芝兰秀,邓老仙山桂蕊轻。
闪闪寰球同寿者,几人夔魅合清声。 
</pre>

<pre>
中国足球

寿张运动人,比类皆惊队。
南建蹴鞠师,山呼千万众。
我军蒙汉旗,乘夏凌秋阵。
流血触藩篱,破头何所恨。
</pre>

<pre>
江城子 通胀

混全钢铁伏完坚。铸山钱,水淹天。蛇吞象箸,狐食虎餐前。半化半人残骨贱,丸美药,不传偏。
饱谙此术雇员闲。算来年,利究颠。元轻钞重,市物贵颠连。通缩预期成祸兆,君看取,券如烟。
</pre>

<pre>
沁园春 黑洞  

无数光年,亘古纠缠,暗物质围。引力何须问,超球绕日,火轮灭迹,散为千堆。奇点协常,类星暴起,巨穴茫茫冰壁垂。空区哪,似可凭依拟,地底窥来。
知君才调横飞,逸气黒旋磨折堕微。掩鼻偷看怯,魔方急掷,骇人大半,怕放谁回。题破乾坤,猜中月魄,悟入风云际会开。留丹灶,令心明透拜,俱向尘埃。
</pre>



If you have any questions, or if you wanna to ask for  fine-tuned models (诗/词) based on self-training, please contact zoux18@mails.tsinghua.edu.cn 

<pre>
@inproceedings{10.1145/3447548.3467418,
author = {Zou, Xu and Yin, Da and Zhong, Qingyang and Yang, Hongxia and Yang, Zhilin and Tang, Jie},
title = {Controllable Generation from Pre-Trained Language Models via Inverse Prompting},
year = {2021},
isbn = {9781450383325},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3447548.3467418},
doi = {10.1145/3447548.3467418},
booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
pages = {2450–2460},
numpages = {11},
keywords = {controllable generation, poem generation, language modeling, beam search, machine question answering},
location = {Virtual Event, Singapore},
series = {KDD '21}
}
</pre>

