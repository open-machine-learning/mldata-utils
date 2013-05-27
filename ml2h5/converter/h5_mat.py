import h5py, numpy
from scipy.io import savemat, loadmat
from ml2h5.converter.basehandler import BaseHandler

import sys
if sys.version < '3':
    def u(x):
        return unicode(x)
else:
    def u(x):
        return str(x)


class H5_MAT(BaseHandler):
    """Handle Matlab files."""

    def __init__(self, *args, **kwargs):
        super(H5_MAT, self).__init__(*args, **kwargs)


    def read(self):
        matf = loadmat(self.fname,
                squeeze_me=False,
                chars_as_strings=True,
                mat_dtype=True,
                struct_as_record=True)

        for k in ('__header__', '__globals__', '__version__'):
            if k in matf:
                del matf[k]

        for k in list(matf.keys()):
            if matf[k].dtype == numpy.object: # asume cell of strings
                cell = []
                for i in matf[k][0]:
                    if len(i[0]):
                        cell.append(str(i[0]))
                    else:
                        cell.append('')
                matf[k]=cell
            elif matf[k].shape[0] == 1: # row vectors are expanded
                matf[k]=matf[k][0]

        data = matf
        if 'mldata_descr_ordering' in matf:
            ordering = matf['mldata_descr_ordering']
            del(matf['mldata_descr_ordering'])
            
        else:        
            ordering = list(matf.keys())

            def strip_type(x):
                if x.startswith('double'):
                    return int(x[6:])
                elif x.startswith('int'):
                    return int(x[3:])
                elif x.startswith('str'):
                    return int(x[3:])
                else:
                    return x
            ordering.sort(cmp=lambda x,y: cmp(strip_type(x),strip_type(y)))
        return {
            'name': self.get_name(),
            'comment': 'matlab',
            'names': [],
            'ordering': ordering,
            'data': data,
        }


    def write(self, data):
        group=self.get_data_group(data)
        d=data[group]
        for k in list(d.keys()):
            if type(d[k])==list and len(d[k])>0 and type(d[k][0])==str:
                cell = array([ numpy.array(u(i)) for i in d[k] ], dtype=numpy.object)
                d[k] = cell
            elif type(d[k])==numpy.ndarray and len(d[k])>0 and type(d[k][0])==str:
                cell = numpy.empty((1,len(d[k])), dtype=numpy.object)
                for i in range(len(d[k])):
                    cell[0,i]=numpy.array(u(d[k][i]), dtype='U')
                d[k] = cell
            elif type(d[k])==numpy.ndarray and len(d[k])>0 and d[k][0].dtype == numpy.object:
                cell = numpy.empty((1,len(d[k])), dtype=numpy.object)
                for i in range(len(d[k])):
                    cell[0,i]=numpy.array(u(d[k][i]), dtype='U')
                d[k] = cell

        d['mldata_descr_ordering']=numpy.array(data['ordering'],dtype=numpy.object)
        savemat(self.fname, d,
                appendmat=False,
                oned_as='row',
                format='5',
                long_field_names=True)
