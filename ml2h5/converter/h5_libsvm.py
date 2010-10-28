import numpy, h5py, os, copy
from scipy.sparse import csc_matrix
from basehandler import BaseHandler



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
        print "start"
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

            lab=[int(i) for i in lab.split(',')]

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

        print "done"
        return {
            'name': self.get_name(),
            'comment': 'LibSVM',
            'ordering': ['label', 'data'],
            'names': ['label', 'data'],
            'data': { 'label' : label, 'data' : data}
        }


    def write(self, data):
        """ this needs to `transpose' the data """
        libsvm = open(self.fname, 'w')

        if 'label' in data.keys():
            if len(data['data']['label'][0]) == 1:
                is_multilabel = False
            else:
                is_multilabel = True

        lengths=dict()
        for o in data['ordering']:
            x=numpy.array(data['data'][o])
            if numpy.issubdtype(x.dtype, numpy.int):
                data['data'][o]=x.astype(numpy.float64)
            try:
                lengths[o]=data['data'][o].shape[1]
            except (AttributeError, IndexError):
                lengths[o]=len(data['data'][o])
        l=set(lengths.values())
        assert(len(l)==1)
        l=l.pop()

        for i in xrange(l):
            out = []
            for o in data['ordering']:
                d=data['data'][o]
                if o == 'label':
                    if is_multilabel:
                        labels = []
                        for j in xrange(len(d[i])):
                            if d[i][j] == 1:
                                labels.append(str(j))
                        out.append(','.join(labels))
                    else:
                        out.append(str(data['label'][i][0]))
                else:
                    for j in xrange(len(d[i])):
                        out.append(str(j+1) + ':' + str(d[i][j]))
                        #try:
                        #  line.extend(map(str, d[:,i]))
                        #except:
                        #  line.append(str(d[i]))
            libsvm.write(" ".join(out) + "\n")

        libsvm.close()

        return True
