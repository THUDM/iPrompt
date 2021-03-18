import jsonlines
def generate(fname):
    import json
    json_file = open(fname, 'r')
    w = json_file.readlines()
    sumscore = 0
    sumsample = 0
    total_length = 0
    has={}
    has['跟异性同桌做过']=1
    with jsonlines.open('qa.jsonl', mode='w') as writer:
        for line in w:
            json_decode = json.loads(line)
            context_str = json_decode['prompt']
            text_str = json_decode['text']
            while '<n><n>' in text_str:
                text_str=text_str.replace('<n><n>','<n>')
            
            if len(context_str)>0:
                
                ct=context_str.split("问题描述")
                question=ct[0][3:]
                des=ct[1].split("回答")[0][1:]
                if len(des)+len(question)+len(text_str)<512:
                    if not(('数学' in question)or('物理' in question)):
                        if not('[图片' in question+des+text_str):
                            if not(question[:7] in has):
                                if question[-1]in ['?','？']:

                                    sumsample+=1
                                    otc={}
                                    otc['question']=question
                                    otc['desc']=des
                                    otc['answer']=text_str
                                    writer.write(otc)
                                    has[question[:7]]=1
            if sumsample>=100:
                return 0
    return 1


generate("../test.json")
