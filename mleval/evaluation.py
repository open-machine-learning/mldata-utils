import numpy

def calcbal(out, lab):
    pidx=(lab==+1)
    perr=numpy.mean(out[pidx]>0)
    nidx=(lab==-1)
    nerr=numpy.mean(out[nidx]<0)
    return 0.5 * (perr+nerr)

def accuracy(out, lab):
    return numpy.mean(numpy.sign(out) == lab)

def error(out, lab):
    return numpy.mean(numpy.sign(out) != lab)

def get_tp(out, lab):
    pidx=(lab==+1)
    return numpy.sum(numpy.sign(out[pidx])>0)

def get_tn(out, lab):
    nidx=(lab==-1)
    return numpy.sum(numpy.sign(out[nidx])<0)

def get_fp(out, lab):
    nidx=(lab==-1)
    return numpy.sum(numpy.sign(out[nidx])>0)

def get_fn(out, lab):
    pidx=(lab==+1)
    return numpy.sum(numpy.sign(out[pidx])<0)

def calcwracc(out, lab):
    tp=get_tp(out, lab)
    fp=get_fp(out, lab)
    tn=get_tn(out, lab)
    fn=get_fn(out, lab)
    return tp/(tp+fn) - fp/(fp+tn)

def calcf1(out, lab):
    tp=get_tp(out, lab)
    fp=get_fp(out, lab)
    fn=get_fn(out, lab)
    return 2.0*tp/(2.0*tp+fp+fn)

def calccc(out, lab):
    tp=float(get_tp(out, lab))
    fp=float(get_fp(out, lab))
    tn=float(get_tn(out, lab))
    fn=float(get_fn(out, lab))
    return (tp*tn-fp*fn)/numpy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

def calcroc(out, lab):
    idx = numpy.argsort(out)
    tp=numpy.concatenate( ([1], 1-numpy.cumsum(lab[idx]>0)/float(numpy.sum(lab > 0))) )
    fp=numpy.concatenate( ([1], 1-numpy.cumsum(lab[idx]<0)/float(numpy.sum(lab < 0))) )
    return (fp,tp)

def calcrocscore(out, lab):
    a,b=calcroc(out,lab)
    score = 0.5*numpy.sum((a[:-1]-a[1:])*(b[:-1]+b[1:]))
    return score

def calcprc(out, lab):
    pmap=(lab==1)
    idx = numpy.argsort(-out)
    pmap=pmap[idx]

    recall=numpy.cumsum(pmap)/numpy.double(sum(lab==1))
    precision=numpy.cumsum(pmap)/(numpy.arange(len(pmap),dtype=numpy.double)+1.0)
    return (precision,recall)

def calcprcscore(out, lab):
    a,b=calcprc(out,lab)
    score = 0.5*numpy.sum((a[:-1]+a[1:])*(b[1:]-b[:-1]))
    return score

def compute_area(a,b):
    return 0.5*numpy.sum(numpy.abs(a[:-1]-a[1:])*numpy.abs(b[:-1]+b[1:]))

def alternative_calcprcscore(out, lab):
    """
    code translated from image challenge
    """
    idx = numpy.argsort(-out)
    tp=lab[idx]>0
    fp=lab[idx]<0
    fp=1.0*numpy.cumsum(fp)
    tp=1.0*numpy.cumsum(tp)
    rec=tp/numpy.sum(lab>0)
    prec=tp/(fp+tp)
    ap=0
    for t in numpy.arange(0,1.01, 0.01):
        el=prec[rec>=t]
        if len(el)>0:
            p=numpy.max(el)
        else:
            p=0
        ap=ap+p/101.0
    return ap
