import numpy
from .util import register

pm = dict()

def accuracy(out, lab):
    """
      Computes Accuracy.
      Expects labels and predictions to be integers
    """
    return numpy.mean(out == lab)
register(pm, 'Accuracy', accuracy)

def error(out, lab):
    """
      Computes Error Rate.
      Expects labels and predictions to be integers
    """
    return numpy.mean(out != lab)
register(pm, 'Error', error)
