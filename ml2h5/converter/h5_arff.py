import h5py, numpy, copy
import ml2h5.converter
from ml2h5.converter import arff
from ml2h5.converter.basehandler import BaseHandler
from scipy.sparse import csc_matrix


class H5_ARFF(BaseHandler):
    """Handle ARFF files.

    It uses the module arff provided by the dataformat project:
    http://mloss.org/software/view/163/
    """

    def __init__(self, *args, **kwargs):
        super(H5_ARFF, self).__init__(*args, **kwargs)


    def _get_types(self, af):
        """Get types from Arff file

        @param af: opened ARFF file
        @type af: arff.ArffFile
        @return: types in arff file
        @rtype: numpy.array
        """
        # need to get it in the right order
        types = []
        for name in af.attributes:
            t = af.attribute_types[name]
            if af.attribute_data[name]:
                t += ':' + ','.join(af.attribute_data[name])
            types.append(t)
        return numpy.array(types)


    def read(self):
        data = {}
        names = []
        ordering = []
        af = arff.ArffFile.load(self.fname)
        for name in af.attributes:
            names.append(name)
            data[name] = []

        for item in af.data:
            for i in range(len(data)):
                data[names[i]].append(item[i])
        # conversion to proper data types
        for name, values in data.items():
            if af.attribute_types[name] == 'date':
                t = self.str_type
            else:
                t = self.get_datatype(values)
            data[name] = numpy.array(values).astype(t)
        ddict= {
            'name': af.relation,
            'comment': af.comment,
            'types': self._get_types(af),
            'names':names,
            'ordering':copy.copy(names),
            'data':data,
        }
        

        if self.merge == True:
            ddict = self._get_merged(ddict)
        return ddict
    def check_sparse(self, data):
        for k in list(data.keys()):
            d=data[k]
        if type(d)==csc_matrix:
            raise ml2h5.converter.ConversionError("Sparse matrices are not supported in ARFF files")

    def write(self, data):
        group=self.get_data_group(data)
        d=data[group]
        self.check_sparse(d)

        af = arff.ArffFile()
        af.data = self.get_data_as_list(data)
        if 'names' in data and len(data['names']):
            af.attributes = data['names']
        else:
            attrs=[]
            idx=0
            for o in data['ordering']:
                try:
                    if len(d[o].shape)==1:
                        num=1
                    else:
                        num=d[o].shape[0]

                    if d[o].dtype == numpy.double:
                        prefix='double'
                    elif d[o].dtype in [numpy.int32, numpy.int64 ]:
                        prefix='int'
                    else:
                        prefix='str'
                except AttributeError: # list - so assume list of strings
                    num=1
                    prefix='str'

                for i in range(num):
                    attrs.append('%s%d' % (prefix,idx))
                    idx+=1

            if len(attrs)==len(data['ordering']):
                attrs=data['ordering']

            af.attributes = attrs

        af.relation = data['name']
        af.comment = data['comment']

        # handle arff types
        if 'types' in data:
            types = data['types']
        else:
            types = []
            for name in af.attributes:
                if name.startswith('int') or name.startswith('double'):
                    types.append('numeric')
                elif name.startswith('date'):
                    types.append('date')
                else:
                    types.append('string')

        for i in range(len(types)):
            t = types[i].split(':')
            af.attribute_types[af.attributes[i]] = t[0]
            if len(t) == 1:
                af.attribute_data[af.attributes[i]] = None
            else:
                af.attribute_data[af.attributes[i]] = t[1].split(',')

        af.save(self.fname)
