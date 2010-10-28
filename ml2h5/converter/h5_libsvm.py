import numpy, h5py, os, copy
from scipy.sparse import csc_matrix
from basehandler import BaseHandler
import ml2h5.converter


class H5_LibSVM(BaseHandler):
    """Handle LibSVM files."""

    def __init__(self, *args, **kwargs):
        """Constructor.

        @ivar is_multilabel: if data is of type multilabel
        @type is_multilabel: boolean
        """
        super(H5_LibSVM, self).__init__(*args, **kwargs)
        self.is_multilabel = False

    #def convert_sparse(self, spmatrix):
    #    A=spmatrix
    #    if A.nnz/numpy.double(A.shape[0]*A.shape[1]) < 0.5: # sparse
    #        data['data_indices'] = A.indices
    #        data['data_indptr'] = A.indptr
    #        data['data_data'] = A.data
    #    else: # dense
    #        data['data'] = A.todense()

    def get_comment(self):
        return 'LibSVM'

    def read(self):
        """Retrieves a SciPy Compressed Sparse Column matrix and labels from file.
        @return: compressed sparse column matrix + labels
        @rtype: list of scipy.sparse.csc_matrix and label tuple/2-d tuple (multilabel)
        """
        indices_var = []
        indptr_var = [0]
        ptr_var = 0
        data_var = []

        indices_lab = []
        indptr_lab = [0]
        ptr_lab = 0
        data_lab = []

        label = []

        for line in file(self.fname).xreadlines():
            items = line.split()
            lab=items[0]
            dat=items[1:]

            lab=[int(float(i)) for i in lab.split(',')]

            if len(lab)>1:
                self.is_multilabel = True

            if self.is_multilabel:
                for idx in lab:
                    indices_lab.append(int(idx))
                    data_lab.append(1.)
                    ptr_lab += 1
                indptr_lab.append(ptr_lab)
            else: # only single label -> value is actual value
                label.append(lab[0])

            for d in dat:
                v=d.split(':')
                indices_var.append(int(v[0]) - 1)
                data_var.append(numpy.double(v[1]))
                ptr_var += 1
            indptr_var.append(ptr_var)

        if self.is_multilabel:
            label = csc_matrix( (numpy.array(data_lab), numpy.array(indices_lab),
                    numpy.array(indptr_lab)) )
        else:
            label = numpy.array(label)

        data = csc_matrix( (numpy.array(data_var), numpy.array(indices_var),
            numpy.array(indptr_var)) )

        return {
            'name': self.get_name(),
            'comment': 'LibSVM',
            'ordering': ['label', 'data'],
            'names': ['label', 'data'],
            'data': { 'label' : label, 'data' : data}
        }


    def write(self, data):
        names=('label','data')
        if not set(data['data'].keys()).issubset(set(names)):
            raise ml2h5.converter.ConversionError('libsvm format needs data or label')

        if type(data['data']['data']) != csc_matrix:
            raise ml2h5.converter.ConversionError('libsvm format needs csc_matrix data')

        libsvm = open(self.fname, 'w')

        if 'label' in data['data'].keys():
            if type(data['data']['label']) == csc_matrix:
                self.is_multilabel = True
            else:
                self.is_multilabel = False

        lengths=dict()
        for o in names:
            try:
                lengths[o]=data['data'][o].shape[1]
            except (AttributeError, IndexError):
                lengths[o]=len(data['data'][o])
        l=set(lengths.values())
        assert(len(l)==1)
        l=l.pop()

        for i in xrange(l):
            out = []
            for o in names:
                d=data['data'][o]
                if o == 'label':
                    if self.is_multilabel:
                        labels = []
                        indptr=d.indptr
                        indices=d.indices
                        dat=d.data
                        for j in xrange(indptr[i],indptr[i+1]):
                            labels.append(str(indices[j]+1))

                        out.append(','.join(labels))
                    else:
                        out.append(str(d[i]))
                else:
                    indptr=d.indptr
                    indices=d.indices
                    dat=d.data
                    for j in xrange(indptr[i],indptr[i+1]):
                        out.append(str(indices[j]+1) + ':' + str(dat[j]))
            libsvm.write(" ".join(out) + "\n")

        libsvm.close()

        return True
