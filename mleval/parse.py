import numpy

def parse_floats(fil):
    """
    Parse file and return numpy array
    """

    values=list()
    for l in fil.xreadlines():
        values.append( [ float(e) for e in l.split() ] )

    return numpy.array(values)
