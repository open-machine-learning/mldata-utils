import numpy
from util import register

pm = dict()

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
register(pm, 'Balanced Error', calcbal)

def accuracy(out, lab):
    """
      Computes Accuracy.
      Expects labels to be +/-1 and predictions sign(output)
    """
    return numpy.mean(numpy.sign(out) == lab)
register(pm, 'Accuracy', accuracy)

def error(out, lab):
    """
      Computes Error Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    return numpy.mean(numpy.sign(out) != lab)
register(pm, 'Error', error)

def get_tp(out, lab):
    """
      Computes True Positive Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    pidx=(lab==+1)
    return numpy.sum(numpy.sign(out[pidx])>0)
register(pm,'True Positive Rate', get_tp)

def get_tn(out, lab):
    """
      Computes True Negative Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    nidx=(lab==-1)
    return numpy.sum(numpy.sign(out[nidx])<0)
register(pm, 'True Negative Rate', get_tn)

def get_fp(out, lab):
    """
      Computes False Positive Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    nidx=(lab==-1)
    return numpy.sum(numpy.sign(out[nidx])>0)
register(pm, 'False Positive Rate', get_fp)

def get_fn(out, lab):
    """
      Computes False Negative Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    pidx=(lab==+1)
    return numpy.sum(numpy.sign(out[pidx])<0)
register(pm, 'False Negative Rate', get_fn)

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
register(pm, 'Weighted Relative Accuracy', calcwracc)

def calcf1(out, lab):
    """
      Computes the F1 score.
      Expects labels to be +/-1 and predictions sign(output)
    """
    tp=get_tp(out, lab)
    fp=get_fp(out, lab)
    fn=get_fn(out, lab)
    return 2.0*tp/(2.0*tp+fp+fn)
register(pm, 'F1 Score', calcf1)

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
register(pm, 'Cross Correlation Coefficient', calccc)

def calcroc(out, lab):
    """
      Computes the ROC curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    idx = numpy.argsort(out)
    tp=numpy.concatenate( ([1], 1-numpy.cumsum(lab[idx]>0)/float(numpy.sum(lab > 0))) )
    fp=numpy.concatenate( ([1], 1-numpy.cumsum(lab[idx]<0)/float(numpy.sum(lab < 0))) )
    score = 0.5*numpy.sum((fp[:-1]-fp[1:])*(tp[:-1]+tp[1:]))
    curve= {'x' : fp,
            'y' : tp,
            'x_name' : 'False Positive Rate',
            'y_name' : 'True Positive Rate'}
    return (score, 'Curve', curve)
register(pm, 'ROC Curve', calcroc)

def calcrocscore(out, lab):
    """
      Computes the area under the ROC curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    
    return calcroc(out,lab)[0]
register(pm, 'Area under ROC Curve', calcrocscore)

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
    score = 0.5*numpy.sum((precision[:-1]+precision[1:])*(recall[1:]-recall[:-1]))
    curve= {'x' : precision,
            'y' : recall,
            'x_name' : 'Precision',
            'y_name' : 'Recall'}
    return (score, 'Curve', curve)
register(pm, 'Precision Recall Curve', calcprc)

def calcprcscore(out, lab):
    """
      Computes the area under the precision recall curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    return calcprc(out,lab)[0]
register(pm, 'Area under Precision Recall Curve', calcprcscore)
