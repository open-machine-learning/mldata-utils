import numpy
import re
import random


def check_split_str(split_str):
    if re.match("((^|,)(\d+:\d+|\d+))+$",split_str)==None:
        return False

    tk=[]
    tk_sort=[]
    tk_pr=[]
    tk_po=[]
    split_str=str(split_str)
    tk_split=re.findall("(^|,)(\d+:\d+)|(\d+)",split_str)

    for i in range(len(tk_split)):
        if tk_split[i][2]!='':
            tk.append(tk_split[i][2])
            tk_pr.append(int(tk[-1]))
            tk_po.append(int(tk[-1])+1)
        if tk_split[i][1]!='':
            tk.append(tk_split[i][1])
            tk_pr.append(int(tk[-1].split(":")[0]))
            tk_po.append(int(tk[-1].split(":")[1]))

    tk=numpy.array(tk)
    tk_pr=numpy.array(tk_pr)
    tk_po=numpy.array(tk_po)

    sindex=tk_pr.argsort()
    tk=tk[sindex]
    tk_pr=tk_pr[sindex]
    tk_po=tk_po[sindex]

            
    for i in range(len(tk)):
        if tk_pr[i]>=tk_po[i]:
            return False
    return True


def expand_split_str(split_str):

    tk=[]
    tk_sort=[]
    tk_pr=[]
    tk_po=[]
    split_str=str(split_str)
    tk_split=re.findall("(^|,)(\d+:\d+)|(\d+)",split_str)
    for i in range(len(tk_split)):
        if tk_split[i][2]!='':
            tk.append(tk_split[i][2])
            tk_pr.append(int(tk[-1]))
            tk_po.append(int(tk[-1])+1)
        if tk_split[i][1]!='':
            tk.append(tk_split[i][1])
            tk_pr.append(int(tk[-1].split(":")[0]))
            tk_po.append(int(tk[-1].split(":")[1]))

    tk=numpy.array(tk)
    split_pr=numpy.array(tk_pr)
    split_po=numpy.array(tk_po)
    out_set = set()
    for l in range(len(split_pr)):
        for i in range(split_po[l]-split_pr[l]):
            out_set.add(split_pr[l]+i)

    out = [i for i in out_set]        
    out.sort()
    out_str=[str(i) for i in out]
    return out_str

def reduce_split_str(split_str):
    tk=[]
    tk_sort=[]
    tk_pr=[]
    tk_po=[]

    if type(split_str) in [list,numpy.ndarray,numpy.matrix]:
        split_str=', '.join([str(i) for i in split_str])
    else:    
        split_str=str(split_str)

    tk_split=re.findall("(^|,)(\d+:\d+)|(\d+)",split_str)
    for i in range(len(tk_split)):
        if tk_split[i][2]!='':
            tk.append(tk_split[i][2])
            tk_pr.append(int(tk[-1]))
            tk_po.append(int(tk[-1])+1)
        if tk_split[i][1]!='':
            tk.append(tk_split[i][1])
            tk_pr.append(int(tk[-1].split(":")[0]))
            tk_po.append(int(tk[-1].split(":")[1]))

    tk=numpy.array(tk)
    tk_pr=numpy.array(tk_pr)
    tk_po=numpy.array(tk_po)

    sindex=tk_pr.argsort()
    tk=tk[sindex]
    tk_pr=tk_pr[sindex]
    tk_po=tk_po[sindex]

    if len(tk) == 1:
        return tk
    
    tk_reduce=[]
    akt=0
    red_ind=0
    aktstr=str(tk_pr[0])
    for i in range(1,len(tk)):
        if (tk_pr[i]==tk_po[i-1]):
            akt+=1
            
        else:
            if akt>0:
                aktstr+=":"+str(tk_po[i-1])
                tk_reduce.append(aktstr)
                aktstr=str(tk_pr[i])
                akt=0
            else:
                tk_reduce.append(tk[i-1])
                aktstr=str(tk_pr[i])
        if i==len(tk)-1:
            if akt>0:
                aktstr+=":"+str(tk_po[i])
                tk_reduce.append(aktstr)
                aktstr=tk_pr[i]
            else:
                tk_reduce.append(tk[i])


    


    return tk_reduce

def main():
    print check_split_str("1,2,3,4,5:7,27,28:32,32:30")
    x = reduce_split_str("1")
    print x
    y= expand_split_str("1,2,3:7,27:33")
    print y
    

if __name__=="__main__":
    main()
