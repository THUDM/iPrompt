
def cilin():
    f=open("cilin.txt",'r')
    t=f.readlines()
    bu=0
    nowsb=0
    allbu=[]
    def_chr=['[',']','(',')','\n',' ','，','\u3000','。','《','》']
    allsb=[[],[]]
    worddict={}
    shengdict={}
    for line in t:
        if len(line)<5:
            continue
        if ('第' in line) and ('部' in line):
            bu+=1
            allbu.append([])
        if ('平声' in line):
            nowsb=0
        if ('仄声' in line):
            nowsb=1
        if ('入声' in line):
            nowsb=1
            
        if ('【' in line):
            tks=line.split('】')[1]
            currentst1=0
            currentst2=0
            for num in range(len(tks)):
                char=tks[num]
                if currentst1+currentst2==0:
                    if not(char in def_chr):
                        allbu[-1].append(char)
                        allsb[nowsb].append(char)
                        if char in worddict:
                            if not(bu in worddict[char]):
                                worddict[char].append(bu)
                        else:
                            worddict[char]=[bu]
                        if char in shengdict:
                            if not(nowsb in shengdict[char]):
                                shengdict[char].append(nowsb)
                        else:
                            
                            shengdict[char]=[nowsb]
                            
                if char=='[':
                    currentst1=1
                if char==']':
                    currentst1=0
                if char=='(':
                    currentst2=1
                if char==')':
                    currentst2=0
    
    
    return worddict,shengdict,allbu,allsb
    #print(allbu[0])

        
                
def checkrhyself(sentence,shengdict,rhy):
    if len(sentence)==0:
        return 0
    st=sentence
    fullst=False
    while (len(st)>0 and st[-1] in [',','。','，','?','？','!','！']):
        st=st[:-1]
        fullst=True
    
    l1=pinyin(st,style=TONE3)

    if len(l1)<len(st):
        print(l1,sentence)
        return 1
    for i in l1:
        if len(i[0])<2:
            return 1
    if len(st)<=3:
        return 2
    
    
    for i in st:
        if not(i in shengdict):
            print(sentence,i)
            return 1
      
    srhy=rhy%3
    
    pz1=shengdict[st[1]]
    if srhy!=0:
        if len(pz1)==1:
            if srhy-pz1[0]!=1:
                return 1
    if srhy==0:
        if len(pz1)==1:
            srhy=pz1[0]+1
        
    if len(st)>=4:
        pz2=shengdict[st[3]]
        if srhy!=0:
            if len(pz2)==1:
                if pz2[0]+srhy!=2:
                    return 1
        if srhy==0:
            if len(pz2)==1:
                srhy=2-pz2[0]
    #shry: 1: 010  2:101
    
    if len(st)>=6:
        
        pz3=shengdict[st[5]]
        if srhy!=0:
            if len(pz3)==1:
                if srhy-pz3[0]!=1:
                    return 1
    
    if fullst:
        if len(sentence)<6:
            return 1
            
        pz11=shengdict[st[-3]]
        pz12=shengdict[st[-2]]
        pz13=shengdict[st[-1]]
        if pz11[0]+pz12[0]+pz13[0]==0:
            return 1
        if pz11[0]+pz12[0]+pz13[0]==3:
            return 1
        wq=rhy//3
        if wq>0:
            if len(pz13)==1:
                if wq-pz13[0]!=1:
                    return 1
        #print(sentence,pz1,rhy,srhy)
        
    return 2
        
    
def checkrhy(sentence,last,imp,shengdict,req=0):
    
    while (len(sentence)>0 and (sentence[-1] in [',','。','，','?','？','!','！'])):
        sentence=sentence[:-1]
    if len(sentence)==0:
        return 0
        
    while last[-1] in [',','。','，','?','？','!','！']:
        last=last[:-1]
    l1=pinyin(sentence,style=TONE3)
    l2=pinyin(last,style=TONE3)
        #print(l1,l2)
    disobey=0
    if len(l1)!=len(sentence):
        return -1000

    for i in range(len(sentence)):
        if (i<len(l1)) and (i<len(l2)):
            if not(sentence[i] in shengdict):
                print(sentence,sentence[i])
                return -1000
            st1=shengdict[sentence[i]]
            if not(last[i] in shengdict):
                print(last,last[i])
                return -1000
            sr1=shengdict[last[i]]
            dst=0
            if len(st1)+len(sr1)==2:
                if (req==1 and i%2==1):
                    if st1[0]+sr1[0]==1:
                        dst=1
                else:
                    if st1[0]+sr1[0]!=1:
                        dst=1
                
            if dst==1:
                if req==0:
                    disobey+=0.35
                if i%2==1:
                    disobey+=0.35
                    if req==1:
                        disobey+=0.2
                if i==len(l2)-1:
                    disobey+=0.65
                    if req==1:
                        disobey+=0.35
                
    disobey*=imp
    disobey=-5*disobey/len(l2)
    for i in range(len(l1)):
        for j in range(i+2,len(l1)):
            if l1[i][0][:-1]==l1[j][0][:-1]:
                disobey-=7/len(l1)
    return disobey

def checksentence(sentence,original_context,min_length,max_length,endnote,dic,wdic,curvote=0,yayun=None,rhy=0):
    
    if "<|end" in sentence:
        return 0
   
    if "的" in sentence[1:]:
        return 1
    if "些" in sentence[1:]:
        return 1
    if "么" in sentence[1:]:
        return 1
   
    if len(sentence)==0:
        return 1
    if ((len(sentence)>max_length and not(sentence[-1] in endnote)) or len(sentence)==0) or len(sentence)>max_length+1:
        return 1
    if (sentence[-1] in endnote)and ((len(sentence)<=min_length) or (len(sentence)==7)):
        return 1
            
    if (sentence[-1] in endnote)and (sentence[:-1] in original_context):
        return 1
    
    mdisobey=0
    illegal_notes=[' ',':','《','》','‘','“','-','——','⁇','[','【','】',']','.','、','(','（',')','）','·']
    if '。' in endnote:
        illegal_notes.extend([',','，'])
    else:
        illegal_notes.append('。')
    for i in range(10):
        illegal_notes.append(str(i))
    for i in range(64,123):
        illegal_notes.append(chr(i))
    for note in illegal_notes:
        if note in sentence:
            return 1
    last=getlastsentence(original_context)
    if min_length==max_length:
        imp=1
        if (',' in last) or('，' in last):
            imp=1.5
            
        if curvote==0:
            rt=checkrhy(sentence,last,imp,dic,req=1)
        else:
            rt=checkrhy(sentence,last,imp,dic)
        if rt<-0.75:
            return 1
        
        
    for i in range(len(sentence)):
       # if sentence[i]=="柯":
        #    print(sentence[i],last[i],sentence[i]==last[i])
        if min_length==max_length:
            if (i<len(last)-1) and (sentence[i]==last[i]):
                #print(sentence,last)
                return 1
                
            
        
        if i<len(sentence)-3:
            if sentence[i:i+3] in original_context:
                return 1
            if sentence[i:i+2] in sentence[:i]:
                return 1
    
    if checkrhyself(sentence,dic,rhy)==1:
        return 1
    cc=curvote
    if yayun is None:
        cc=0
    if (cc==1 and len(sentence)>=max_length):
        
        for i in sentence[:-1]:
            if not(i in wdic):
                print(sentence,i)
                return 1
                
        
        final1=wdic[sentence[max_length-1]]
        final2=[]
        for i in yayun:
            final2.append(wdic[i])

        doc=0
        for i in final1:
            doc=1
            for td in final2:
                if not(i in td):
                    doc=0
            if doc==1:
                break
                
       
        if doc==0:
            return 1
            
            
    if (sentence[-1] in endnote):
        return 0
        
        
    return 2
    
def getrhy(sentence,rhy,wdic):
    if rhy%3!=0:
        return rhy%3
    a=wdic[sentence[1]]
    if len(a)==1:
        if a[0]==0:
            return 1
        else:
            return 2
    b=wdic[sentence[3]]
    if len(b)==1:
        if b[0]==0:
            return 2
        else:
            return 1
    if len(sentence)>6:
        c=wdic[sentence[5]]
        if c[0]==0:
            return 1
        else:
            return 2
    return 0
    

