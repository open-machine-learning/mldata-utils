import numpy

"""
dictionary of known performance measures:

keys are human readable names
values are tuples of 
(functioname, application domain ('Classification', 'Regression', ...), description)

pm['Balanced Error'] = (calcbal, 'Classification',
"Computes the Balanced Error. Expects labels to be +/-1 and predictions sign(output)"

"""
pm=dict()

def calcbal(out, lab):
    """
      Computes the Balanced Error.
      Expects labels to be +/-1 and predictions sign(output)
    """
    pidx=(lab==+1)
    perr=numpy.mean(out[pidx]>0)
    nidx=(lab==-1)
    nerr=numpy.mean(out[nidx]<0)
    return 0.5 * (perr+nerr)
pm['Balanced Error'] = (calcbal, 'Classification', calcbal.__doc__)

def accuracy(out, lab):
    """
      Computes Accuracy.
      Expects labels to be +/-1 and predictions sign(output)
    """
    return numpy.mean(numpy.sign(out) == lab)
pm['Accuracy'] = (accuracy, 'Classification', accuracy.__doc__)

def error(out, lab):
    """
      Computes Error Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    return numpy.mean(numpy.sign(out) != lab)
pm['Error'] = (error, 'Classification', error.__doc__)

def get_tp(out, lab):
    """
      Computes True Positive Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    pidx=(lab==+1)
    return numpy.sum(numpy.sign(out[pidx])>0)
pm['True Positive Rate'] = (get_tp, 'Classification', get_tp.__doc__)

def get_tn(out, lab):
    """
      Computes True Negative Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    nidx=(lab==-1)
    return numpy.sum(numpy.sign(out[nidx])<0)
pm['True Negative Rate'] = (get_tn, 'Classification', get_tn.__doc__)

def get_fp(out, lab):
    """
      Computes False Positive Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    nidx=(lab==-1)
    return numpy.sum(numpy.sign(out[nidx])>0)
pm['False Positive Rate'] = (get_fp, 'Classification', get_fp.__doc__)

def get_fn(out, lab):
    """
      Computes False Negative Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    pidx=(lab==+1)
    return numpy.sum(numpy.sign(out[pidx])<0)
pm['False Negative Rate'] = (get_fn, 'Classification', get_fn.__doc__)

def calcwracc(out, lab):
    """
      Computes Weighted Relative Accuracy.
      Expects labels to be +/-1 and predictions sign(output)
    """
    tp=get_tp(out, lab)
    fp=get_fp(out, lab)
    tn=get_tn(out, lab)
    fn=get_fn(out, lab)
    return tp/(tp+fn) - fp/(fp+tn)
pm['Weighted Relative Accuracy'] = (calcwracc, 'Classification', calcwracc.__doc__)

def calcf1(out, lab):
    """
      Computes the F1 score.
      Expects labels to be +/-1 and predictions sign(output)
    """
    tp=get_tp(out, lab)
    fp=get_fp(out, lab)
    fn=get_fn(out, lab)
    return 2.0*tp/(2.0*tp+fp+fn)
pm['F1 Score'] = (calcf1, 'Classification', calcf1.__doc__)

def calccc(out, lab):
    """
      Computes the cross correlation coefficient.
      Expects labels to be +/-1 and predictions sign(output)
    """
    tp=float(get_tp(out, lab))
    fp=float(get_fp(out, lab))
    tn=float(get_tn(out, lab))
    fn=float(get_fn(out, lab))
    return (tp*tn-fp*fn)/numpy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
pm['Cross Correlation Coefficient'] = (calccc, 'Classification', calccc.__doc__)

def calcroc(out, lab):
    """
      Computes the ROC curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    idx = numpy.argsort(out)
    tp=numpy.concatenate( ([1], 1-numpy.cumsum(lab[idx]>0)/float(numpy.sum(lab > 0))) )
    fp=numpy.concatenate( ([1], 1-numpy.cumsum(lab[idx]<0)/float(numpy.sum(lab < 0))) )
    return (fp,tp)
pm['ROC Curve'] = (calcroc, 'Classification', calcroc.__doc__)

def calcrocscore(out, lab):
    """
      Computes the area under the ROC curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    a,b=calcroc(out,lab)
    score = 0.5*numpy.sum((a[:-1]-a[1:])*(b[:-1]+b[1:]))
    return score
pm['Area under ROC Curve'] = (calcrocscore, 'Classification', calcrocscore.__doc__)

def calcprc(out, lab):
    """
      Computes the precision recall curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    pmap=(lab==1)
    idx = numpy.argsort(-out)
    pmap=pmap[idx]

    recall=numpy.cumsum(pmap)/numpy.double(sum(lab==1))
    precision=numpy.cumsum(pmap)/(numpy.arange(len(pmap),dtype=numpy.double)+1.0)
    return (precision,recall)
pm['Precision Recall Curve'] = (calcprc, 'Classification', calcprc.__doc__)

def calcprcscore(out, lab):
    """
      Computes the area under the precision recall curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    a,b=calcprc(out,lab)
    score = 0.5*numpy.sum((a[:-1]+a[1:])*(b[1:]-b[:-1]))
    return score
pm['Area under Precision Recall Curve'] = (calcprcscore, 'Classification', calcprcscore.__doc__)



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
