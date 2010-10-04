import numpy
from util import register

pm = dict() # regression measures

def mean_absolute_error(out, truth):
    """
      Regression evaluation using Mean Absolute Error
      Expects out and truth to be real valued vectors
    """
    return numpy.mean(numpy.abs(out-truth))
register(pm, 'Mean Absolute Error', mean_absolute_error)

def root_mean_squared_error(out, truth):
    """
      Regression evaluation using Root Mean Squared Error (RMSE)
      Expects out and truth to be real valued vectors
    """
    return numpy.sqrt(numpy.mean((out-truth)**2))
register(pm, 'Root Mean Squared Error', root_mean_squared_error)


