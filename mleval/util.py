# Some helper functions follow

def register(dic, name, fun):
    dic[name]=(fun, fun.__doc__)

def compute_area(a,b):
    """ Helper function to compute area """
    return 0.5*numpy.sum(numpy.abs(a[:-1]-a[1:])*numpy.abs(b[:-1]+b[1:]))

def alternative_calcprcscore(out, lab):
    """
    For sanity checking only: auRPC compute function 
    from PASCAL Image challenge.

    code translated from matlab code from image challenge
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
