import numpy

"""
dictionaries of known performance measures:

keys are human readable names, values are tuples of (functioname, description)

pm              - is a dictionary of all the measures
pm_hierarchy    - is a dictionary of tasks specific perf measures (e.g. for
                    Classification, Regression, ...).

To add a new measure just add a function, e.g. 

def calcbal(out, lab):
    "Computes the Balanced Error"
    ...

and add it to the the appropriate dictionary

cpm['Balanced Error'] = (calcbal, calcbal.__doc__)

"""

rpm = dict() # regression measures
cpm = dict() # two-class classification measures

pm_hierarchy = { 'Regression': rpm, 'Classification': cpm }

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
cpm['Balanced Error'] = (calcbal, calcbal.__doc__)

def accuracy(out, lab):
    """
      Computes Accuracy.
      Expects labels to be +/-1 and predictions sign(output)
    """
    return numpy.mean(numpy.sign(out) == lab)
cpm['Accuracy'] = (accuracy, accuracy.__doc__)

def error(out, lab):
    """
      Computes Error Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    return numpy.mean(numpy.sign(out) != lab)
cpm['Error'] = (error, error.__doc__)

def get_tp(out, lab):
    """
      Computes True Positive Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    pidx=(lab==+1)
    return numpy.sum(numpy.sign(out[pidx])>0)
cpm['True Positive Rate'] = (get_tp, get_tp.__doc__)

def get_tn(out, lab):
    """
      Computes True Negative Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    nidx=(lab==-1)
    return numpy.sum(numpy.sign(out[nidx])<0)
cpm['True Negative Rate'] = (get_tn, get_tn.__doc__)

def get_fp(out, lab):
    """
      Computes False Positive Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    nidx=(lab==-1)
    return numpy.sum(numpy.sign(out[nidx])>0)
cpm['False Positive Rate'] = (get_fp, get_fp.__doc__)

def get_fn(out, lab):
    """
      Computes False Negative Rate.
      Expects labels to be +/-1 and predictions sign(output)
    """
    pidx=(lab==+1)
    return numpy.sum(numpy.sign(out[pidx])<0)
cpm['False Negative Rate'] = (get_fn, get_fn.__doc__)

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
cpm['Weighted Relative Accuracy'] = (calcwracc, calcwracc.__doc__)

def calcf1(out, lab):
    """
      Computes the F1 score.
      Expects labels to be +/-1 and predictions sign(output)
    """
    tp=get_tp(out, lab)
    fp=get_fp(out, lab)
    fn=get_fn(out, lab)
    return 2.0*tp/(2.0*tp+fp+fn)
cpm['F1 Score'] = (calcf1, calcf1.__doc__)

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
cpm['Cross Correlation Coefficient'] = (calccc, calccc.__doc__)

def calcroc(out, lab):
    """
      Computes the ROC curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    idx = numpy.argsort(out)
    tp=numpy.concatenate( ([1], 1-numpy.cumsum(lab[idx]>0)/float(numpy.sum(lab > 0))) )
    fp=numpy.concatenate( ([1], 1-numpy.cumsum(lab[idx]<0)/float(numpy.sum(lab < 0))) )
    return (fp,tp)
cpm['ROC Curve'] = (calcroc, calcroc.__doc__)

def calcrocscore(out, lab):
    """
      Computes the area under the ROC curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    a,b=calcroc(out,lab)
    score = 0.5*numpy.sum((a[:-1]-a[1:])*(b[:-1]+b[1:]))
    return score
cpm['Area under ROC Curve'] = (calcrocscore, calcrocscore.__doc__)

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
cpm['Precision Recall Curve'] = (calcprc, calcprc.__doc__)

def calcprcscore(out, lab):
    """
      Computes the area under the precision recall curve.
      Expects labels to be +/-1 and real-valued predictions
    """
    a,b=calcprc(out,lab)
    score = 0.5*numpy.sum((a[:-1]+a[1:])*(b[1:]-b[:-1]))
    return score
cpm['Area under Precision Recall Curve'] = (calcprcscore, calcprcscore.__doc__)


def mean_absolute_error(self, out, truth):
    """
      Regression evaluation using Mean Absolute Error
      Expects out and truth to be real valued vectors
    """
    return numpy.mean(numpy.abs(out-truth))
rpm['Mean Absolute Error'] = (mean_absolute_error, mean_absolute_error.__doc__)

def root_mean_squared_error(self, out, truth):
    """
      Regression evaluation using Root Mean Squared Error (RMSE)
      Expects out and truth to be real valued vectors
    """
    return numpy.sqrt(numpy.mean((out-truth)**2))
rpm['Root Mean Squared Error'] = (root_mean_squared_error, root_mean_squared_error.__doc__)

pm = dict(cpm)
pm.update(rpm)

# Some helper functions follow

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
