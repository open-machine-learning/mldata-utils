import h5py, numpy
from scipy.io import savemat, loadmat
from basehandler import BaseHandler


class H5_MAT(BaseHandler):
    """Handle Matlab files."""

    def __init__(self, *args, **kwargs):
        super(H5_MAT, self).__init__(*args, **kwargs)


    def read(self):
        matf = loadmat(self.fname, matlab_compatible=True)
        if matf.has_key('__globals__'):
            del matf['__globals__']
        data = matf
        ordering = matf.keys()

        return {
            'name': self.get_name(),
            'comment': 'matlab',
            'names': [],
            'ordering': ordering,
            'data': data,
        }


    def write(self, data):
        m = {}
        for i in xrange(len(data['ordering'])):
            m[str(i)] = numpy.array(data['data'][i][...].tolist())
        savemat(self.fname, m, appendmat=False)
