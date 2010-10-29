import os, h5py, numpy
from scipy.sparse import csc_matrix

from ml2h5 import VERSION_MLDATA, COMPRESSION
from ml2h5.converter import ALLOWED_SEPERATORS


class BaseHandler(object):
    """Base handler class.

    It is the base for classes to handle different data formats.
    It implicitely handles HDF5.

    @cvar str_type: string type to be used for variable length strings in h5py
    @type str_type: numpy.dtype

    @ivar fname: name of file to handle
    @type fname: string
    @ivar seperator: seperator to seperate variables in examples
    @type seperator: string
    """
    str_type = h5py.new_vlen(numpy.str)


    def __init__(self, fname, seperator=None):
        """
        @param fname: name of in-file
        @type fname: string
        @param seperator: seperator used to seperate examples
        @type seperator: string
        """
        self.fname = fname
        self.set_seperator(seperator)


    def set_seperator(self, seperator):
        """Set the seperator to seperate variables in examples.

        @param seperator: seperator to use
        @type seperator: string
        """
        if seperator in ALLOWED_SEPERATORS:
            self.seperator = seperator
        else:
            raise AttributeError(_("Seperator '%s' not allowed!" % seperator))


    def warn(self, msg):
        """Print a warning message.

        @param msg: message to print
        @type msg: string
        """
        return
        print 'WARNING: ' + msg

    def _convert_to_ndarray(self,path,val):
        """converts a attribut to a set of ndarrays depending on the datatype  

        @param path: path of the attribute in the h5 file
        @type path: string 
        @param val: data of the attribute
        @type val: csc_matrix/ndarray  
        @rtype: list of ndarrays 
        """
        A=val
        out=[]
        if type(A)==csc_matrix: # sparse
            out.append((path+'_indices', A.indices))
            out.append((path+'_indptr', A.indptr))
            out.append((path, A.data))
        else: # dense
            out.append((path, numpy.array(A)))
        return out

    def get_data_as_list(self,data):
        """ this needs to `transpose' the data """

        dl=[]

        lengths=dict()
        for o in data['ordering']:
            x=data['data'][o]
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
            line=[]
            for o in data['ordering']:
                try:
                    line.extend(data['data'][o][:,i])
                except:
                    line.append(data['data'][o][i])
            dl.append(line)
        return dl

    def _get_complex_data(self, h5):
        """Get 'complex' data structure.

        @param h5: HDF5 file
        @type h5: File object
        @return: blob of data
        @rtype: list of lists
        """
        # de-merge
        data = []
        for name in h5['/data_descr/ordering']:
            block = h5['/data/' + name][:]
            if type(block[0])== numpy.ndarray:
                for i in xrange(len(block)):
                    data.append(block[i])
            else:
                data.append(block)

        return numpy.matrix(data).T.tolist()


    def get_name(self):
        """Get dataset name from non-HDF5 file

        @return: comment
        @rtype: string
        """
        # without str() it might barf
        return str(os.path.basename(self.fname).split('.')[0])


    def get_datatype(self, values):
        """Get data type of given values.

        @param values: list of values to check
        @type values: list
        @return: data type to use for conversion
        @rtype: numpy.int32/numpy.double/self.str_type
        """
        dtype = None

        for v in values:
            if isinstance(v, int):
                dtype = numpy.int32
            elif isinstance(v, float):
                dtype = numpy.double
            else: # maybe int/double in string
                try:
                    tmp = int(v)
                    if not dtype: # a previous nan might set it to double
                        dtype = numpy.int32
                except ValueError:
                    try:
                        tmp = float(v)
                        dtype = numpy.double
                    except ValueError:
                        return self.str_type

        return dtype


    def read(self):
        """Get data and description in-memory 

        Retrieve contents from file.

        @return: example names, ordering and the examples
        @rtype: dict of: list of names, list of ordering and dict of examples
        """
        # we want the exception handled elsewhere
        h5 = h5py.File(self.fname, 'r')
        contents = {
            'names': h5['/data_descr/names'][...].tolist(),
            'ordering': h5['/data_descr/ordering'][...].tolist(),
            'name': h5.attrs['name'],
            'comment': h5.attrs['comment'],
            'mldata': h5.attrs['mldata'],
            'data': dict(),
        }

        if '/data_descr/types' in h5:
            contents['types'] = h5['/data_descr/types'][...]

        for name in contents['ordering']:
            vname='/data/' + name
            sp_indices=vname+'_indices'
            sp_indptr=vname+'_indptr'

            if sp_indices in h5['/data'] and sp_indptr in h5['/data']:
                contents['data'][name] = csc_matrix((h5[vname], h5[sp_indices], h5[sp_indptr])
            )
            else:
                contents['data'][name] = numpy.array(h5[vname],order='F')

        h5.close()
        return contents



    def _get_merged(self, data):
        """Merge given data where appropriate.

        String arrays are not merged, but all int and all double are merged
        into one matrix.

        @param data: data structure as returned by read()
        @type data: dict
        @return: merged data structure
        @rtype: dict
        """
        merged = {}
        ordering = []
        path = ''
        idx_int = 0
        idx_double = 0
        merging = None
        for name in data['ordering']:
            val = data['data'][name]

            if type(val) == csc_matrix:
                merging = None
                path = name
                merged[path] = val
                ordering.append(path)
                continue

            if name.endswith('_indices') or name.endswith('_indptr'):
                merging = None
                path = name
                merged[path] = val
                continue

            if len(val) < 1: continue

            t = type(val[0])
            if t == numpy.int32:
                if merging == 'int':
                    merged[path].append(val)
                else:
                    merging = 'int'
                    path = 'int' + str(idx_int)
                    ordering.append(path)
                    merged[path] = [val]
                    idx_int += 1
            elif t == numpy.double:
                if merging == 'double':
                    merged[path].append(val)
                else:
                    merging = 'double'
                    path = 'double' + str(idx_double)
                    ordering.append(path)
                    merged[path] = [val]
                    idx_double += 1
            else: # string or matrix
                merging = None
                if name.find('/') != -1: # / sep belongs to hdf5 path
                    path = name.replace('/', '+')
                    data['ordering'][data['ordering'].index(name)] = path
                else:
                    path = name
                ordering.append(path)
                merged[path] = val

        data['data'] = merged
        data['ordering'] = ordering
        return data


    def write(self, data):
        """Write given data to HDF5 file.

        @param data: data to write to HDF5 file.
        @type data: dict of lists
        """
        # we want the exception handled elsewhere
        h5 = h5py.File(self.fname, 'w')
        h5.attrs['name'] = data['name']
        h5.attrs['mldata'] = VERSION_MLDATA
        h5.attrs['comment'] = data['comment']

        try:
            data = self._get_merged(data)

            group = h5.create_group('/data')
            for path, val in data['data'].iteritems():
                for path, val in self._convert_to_ndarray(path,val):
                    group.create_dataset(path, data=val, compression=COMPRESSION)

            group = h5.create_group('/data_descr')
            names = numpy.array(data['names']).astype(self.str_type)
            if names.size > 0: # simple 'if names' throws exception if array
                group.create_dataset('names', data=names, compression=COMPRESSION)
            ordering = numpy.array(data['ordering']).astype(self.str_type)
            if ordering.size > 0:
                group.create_dataset('ordering', data=ordering, compression=COMPRESSION)
            if 'types' in data:
                types = numpy.array(data['types']).astype(self.str_type)
                group.create_dataset('types', data=types, compression=COMPRESSION)
        except: # just do some clean-up
            h5.close()
            os.remove(self.fname)
            raise
        else:
            h5.close()
