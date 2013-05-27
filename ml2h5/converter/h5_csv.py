import numpy, h5py, os, copy, numpy
from scipy.sparse import csc_matrix
from ml2h5.converter.basehandler import BaseHandler
import ml2h5.converter

COMMENT = '# '
SEPERATOR = ','

class H5_CSV(BaseHandler):
    """Handle CSV files.

    @ivar attribute_names_first: first line contains attributes
    @type attribute_names_first: boolean
    """

    def __init__(self, *args, **kwargs):
        super(H5_CSV, self).__init__(*args, **kwargs)
        self.attribute_names_first = False
        self.seperator = SEPERATOR # maybe more flexible in the future


    def _find_nan(self, value):
        """Find NaN values and make them proper numpy.nan.

        Used by map function in get_contents()

        @param value: value to check if it is NaN
        @type value: string
        """
        if value == '?':
            return numpy.nan
        else:
            # str() important! when run thru webserver it will be unicode
            # otherwise and somehow results somewhere in nan being converted
            # to string 'na'...
            return str(value)


    def read(self):
        """
        """
        data = {}
        names = []
        ordering = []

        infile = open(self.fname, 'r')
        parsed = []
        anf = self.attribute_names_first # copy value
        for line in infile:
            if line == '\n' or line == '\n\r':
                continue
            l = line.strip().split(self.seperator)
            if anf:
                names = l
                anf = False
            else:
                l = list(map(self._find_nan, l))
                parsed.append(l)
        infile.close()
        A = numpy.matrix(parsed).T

        for i in range(A.shape[0]):
            items = A[i].tolist()[0]
            t = self.get_datatype(items)
            if t == numpy.int32:
                name = 'int' + str(i)
            elif t == numpy.double:
                name = 'double' + str(i)
            else:
                name = 'str' + str(i)
            data[name] = numpy.array(items).astype(t)
            ordering.append(name)

        if not names:
            names = copy.copy(ordering)
        ddict = {
            'name': self.get_name(),
            'comment': 'CSV',
            'names': names,
            'ordering': ordering,
            'data': data,
        }

        if self.merge == True:
            ddict = self._get_merged(ddict)
        return ddict

    def write(self, data):
        lengths=dict()
        for o in data['ordering']:
            try:
                lengths[o]=data['data'][o].shape[1]
            except (AttributeError, IndexError):
                lengths[o]=len(data['data'][o])
        l=set(lengths.values())
        assert(len(l)==1)
        l=l.pop()

        csv = open(self.fname, 'w')
        for i in range(l):
            line=[]
            for o in data['ordering']:
                d=data['data'][o]
                if type(d)==csc_matrix:
                    raise ml2h5.converter.ConversionError("Sparse matrices are not supported in CSV files")
                try:
                    line.extend(list(map(str, d[:,i])))
                except:
                    try:
                        if len(d[i]):
                            line.append(str(d[i]))
                        else:
                            line.append('')
                    except TypeError:
                        line.append(str(d[i]))
            csv.write(self.seperator.join(line) + "\n")
        csv.close()

        return True
