"""
Handle Data objects and files.

This module heavily relies on the functionality required for http://mldata.org
"""

import h5py, numpy, os, copy, tarfile, zipfile, bz2, gzip
from scipy.sparse import csc_matrix

from . import fileformat, converter
from . import NUM_EXTRACT, LEN_EXTRACT
from __future__ import unicode_literals

def get_num_instattr(fname):
    """Retrieve number of instances and number of attributes from given HDF5 file.

    @param fname: filename to retrieve data from
    @type fname: string
    @return: number of instances and number of attributes
    @rtype: tuple containing 2 integers
    """
    if not h5py.is_hdf5(fname):
        return (-1,-1)

    try:
        h5 = h5py.File(fname, 'r')
        num_inst = 0
        num_attr = 0
        for name in h5['/data_descr/ordering']:
            vname = '/data/' + name
            sp_indptr = vname+'_indptr'
            sp_indices = vname+'_indices'

            if sp_indptr in h5['/data'] and sp_indices in h5['/data']:
                A = csc_matrix(
                        (h5[vname], h5[sp_indices], h5[sp_indptr])
                        )
                num_inst += A.shape[1]
                num_attr += A.shape[0]
            else:
                if len(h5[vname].shape) == 2:
                    num_attr += h5[vname].shape[0]
                    num_inst = h5[vname].shape[1]
                else:
                    num_inst = h5[vname].shape[0]
                    num_attr += 1
        h5.close()
    except:
        num_inst = -1
        num_attr = -1

    return (num_inst, num_attr)


def _get_extract_data(h5):
    """Get extract data from given HDF5 object.

    @param h5: HDF5 file to get extract from
    @type h5: HDF5 file
    @return: data extract
    @rtype: list
    """
    overlength=False
    overwidth=False
    cur_line=0
    extract = []
    try:

        for dset in list(h5['/data_descr/ordering']):

            if cur_line==NUM_EXTRACT:
                overwidth=True    
                break 


            dset = '/data/' + dset

            dset_indptr = dset+'_indptr'
            dset_indices = dset+'_indices'

            if dset_indptr in h5['/data'] and dset_indices in h5['/data']:
                # taking all data takes to long for quick viewing, but having just
                # this extract may result in less columns displayed than indicated
                # by attributes_names

                # reduce columns
                indices = h5[dset_indices][:h5[dset_indptr][NUM_EXTRACT+1]]
                data = (h5[dset][:h5[dset_indptr][NUM_EXTRACT+1]])
                indptr = h5[dset_indptr][:NUM_EXTRACT+1]
                csc_shape = [min(NUM_EXTRACT,max(indices)),min(NUM_EXTRACT,len(indptr))]
                if len(indptr)<len(h5[dset_indptr]):
                    overlength=True
                if any(indices > NUM_EXTRACT):
                    overwidth=True    
                # reduce rows

                for i in range(len(indptr)-1):
                    for j in range(indptr[i],indptr[i+1]):   
                        if indices[j]>=NUM_EXTRACT:   
                            indptr[i+1:]-=numpy.ones(len(indptr[i+1:]))
                data=data[indices < NUM_EXTRACT]
                indices=indices[indices < NUM_EXTRACT]

                # empty array exemption
                if data.shape==(0,) or indices.shape==(0,):
                    data=numpy.array([0])
                    indices=data
                A=csc_matrix((data, indices, indptr),csc_shape).todense()
                if A.shape[0]+cur_line > NUM_EXTRACT:
                    extract.extend(A[:-cur_line].tolist())
                else:
                    extract.extend(A.tolist())

                cur_line+=A.shape[0]
            else:
                if type(h5[dset][0]) == numpy.ndarray:
                    last = len(h5[dset][0])
                    if last > NUM_EXTRACT: 
                        last = NUM_EXTRACT
                        overlength=True
                    app_lines=range(len(h5[dset]))
                    if len(app_lines) + cur_line > NUM_EXTRACT: 
                        app_lines = range(NUM_EXTRACT - cur_line )
                        overwidth=True 
                    for i in app_lines:
                        extract.append(h5[dset][i][:last])
                    cur_line+=len(app_lines)
                else:
                    cur_line+=1
                    last = len(h5[dset])
                    if last > NUM_EXTRACT: 
                        last = NUM_EXTRACT
                        overlength=True
                    extract.append(h5[dset][:last])
        if overwidth:
            extract.append(['...' for i in extract[0]])    
        extract = numpy.matrix(extract).T

        # convert from numpy array to list, if necessary
        t = type(extract[0])
        if t == numpy.ndarray or t == numpy.matrix:
            extract = [y for x in extract for y in x.tolist()]
        if overlength:
            extract.append(['...' for i in extract[0]])        
        # cut all values if necessary
        for l in range(len(extract)):
            for c in range(len(extract[l])):
                extract[l][c]=str(extract[l][c])
                if len(extract[l][c])>LEN_EXTRACT:
                    if extract[l][c][0]=='-':
                        extract[l][c]=extract[l][c][:LEN_EXTRACT-1]+'...'
                    else:
                        extract[l][c]=extract[l][c][:LEN_EXTRACT-2]+'...'
    
        

    except KeyError:
        pass
    except ValueError:
        pass
    except IndexError:
        pass
    return extract


def get_extract(fname):
    """Get an extract of an HDF5 Data file.

    @param fname: filename to get extract from
    @type fname: string
    @return: extract of an HDF5 file
    @rtype: dict with HDF5 attribute/dataset names as keys and their data as values
    """
    format = fileformat.get(fname)
    if fname.endswith('.h5'):
        fname_h5 = fname
    else:
        return get_unparseable(fname)

    extract = {
        'mldata': 0,
        'name': '',
        'comment': '',
        'names': [],
        'types': [],
        'data': [],
    }

    if not h5py.is_hdf5(fname):
        return extract

    h5 = h5py.File(fname, 'r')

    attrs = ('mldata', 'name', 'comment')
    for a in attrs:
        if a in h5.attrs:
            extract[a] = h5.attrs[a]
    if '/data_descr/names' in h5:
        extract['names'] = h5['/data_descr/names'][:].tolist()[:NUM_EXTRACT]
        extract['names_cut'] = copy.copy(extract['names'])
        for i in range(len(extract['names_cut'])):
            if len(extract['names_cut'][i]) > LEN_EXTRACT: 
                extract['names_cut'][i] = extract['names_cut'][i][:LEN_EXTRACT-2] + '...'
        if len(list(h5['/data_descr/names'])) > NUM_EXTRACT:
            extract['names_cut'].append('...')        

    if '/data_descr/types' in h5:
        extract['types'] = h5['/data_descr/types'][:].tolist()[:NUM_EXTRACT]

    extract['data'] = _get_extract_data(h5)

    h5.close()
    return extract


def is_binary(fname):
    """Return true if the given file is binary.

    @param fname: name of file to check if binary
    @type fname: string
    @return: if file is binary
    @rtype: boolean
    """
    f = open(fname, 'rb')
    try:
        CHUNKSIZE = 1024
        while 1:
            chunk = f.read(CHUNKSIZE)
            if '\0' in chunk: # found null byte
                f.close()
                return 1
            if len(chunk) < CHUNKSIZE:
                break # done
    finally:
        f.close()

    return 0




def get_unparseable(fname):
    """Get data from unparseable files

    @param fname: filename to get data from
    @type fname: string
    @return: raw extract from unparseable file
    @rtype: dict with 'attribute' data
    """
    if zipfile.is_zipfile(fname):
        intro = 'ZIP archive'
        f = zipfile.ZipFile(fname)
        data = ', '.join(f.namelist())
        f.close()
    elif tarfile.is_tarfile(fname):
        intro = '(Zipped) TAR archive'
        f = tarfile.TarFile.open(fname)
        data = ', '.join(f.getnames())
        f.close()
    else:
        intro = 'Format not HDF5, zip or tar archive. Will parse the following text:'
        if is_binary(fname):
            data = ''
        else:
            f = open(fname, 'r')
            i = 0
            data = []
            for l in f:
                # make sure we have unicode strings afterwards
                # otherwise this might lead to exceptions
                data.append(l.decode('utf-8', 'replace'))
                i += 1
                if i > NUM_EXTRACT:
                    break
            f.close()
            data = "\n".join(data)

    return {'data': [[intro, data]]}


def get_uncompressed(fname):
    """Get name of uncompressed Data file.

    The given file must be a zipfile/tarball and
    must contain exactly 1 file to be uncompressed.
    The returned name represents a file actually created in the file
    system which the user of this method might have to os.remove().

    @param fname: (possibly) compressed filename
    @type fname: string
    @return: uncompressed filename if file is compressed, None otherwise
    @rtype: string
    """
    src = None
    path = os.path.dirname(fname)

    # archives
    if zipfile.is_zipfile(fname):
        src = zipfile.ZipFile(fname)
        fnames = src.namelist()
    elif tarfile.is_tarfile(fname):
        src = tarfile.open(fname)
        fnames = src.getnames()
    if src:
        if len(fnames) == 1:
            src.extract(fnames[0], path)
            return os.path.join(path, fnames[0])
        else:
            return None

    # gz/bz2
    try:
        src = bz2.BZ2File(fname)
    except:
        try:
            src = gzip.open(fname, 'r')
        except:
            return None
    try:
        base = os.path.basename(fname)
        name = os.path.join(path, '.'.join(base.split('.')[:-1]))
        dst = open(name, 'w')
        dst.write(src.read())
        dst.close()
    except:
        if os.path.exists(name) and not os.path.isdir(name):
            os.remove(name)
        name = None

    src.close()
    return name



def _find_dset(fname, output_variables):
    """Find the dataset in given contents that contains the
    output_variable(s).

    This would be easy if all the data was just in one blob, but it may be
    in several datasets as defined by contents['ordering'], e.g. in
    contents['label'] or contents['data'] or contents['nameofvariable'].

    @param output_variables: index of output_variables to look for
    @type output_variables: integer
    @return: dataset corresponding to output_variables
    @rtype: list of integer
    """

    if not h5py.is_hdf5(fname):
        return

    h5 = h5py.File(fname, 'r')
    dset = None

    ov = output_variables
    for name in h5['/data_descr/ordering']:
        path = '/data/' + name
        path_indptr = path+'_indptr'
        path_indices = path+'_indices'

        if path_indptr in h5['/data'] and path_indices in h5['/data']:
            A = csc_matrix(
                (h5['/data/data'], h5['/data/indices'], h5['/data/indptr'])
            )
            if ov == 0:
                for i in range(A.shape[0]):
                    if ov == 0:
                        dset = A[i].todense().tolist()
                        break # inner loop
                    else:
                        ov -= 1
        else:
            data = h5[path][...]

            if len(data.shape) == 1: # datasets with shape (x,)
                if ov == 0:
                    dset = data
                else:
                    ov -= 1
            else: # datasets with shape (x,y)
                for i in range(len(data)):
                    if ov == 0:
                        dset = data[i].tolist()
                        break # inner loop
                    else:
                        ov -= 1

            if dset is not None:
                break # outer loop

    h5.close()
    return dset

def get_attribute_types(fname):
    if not h5py.is_hdf5(fname):
        return ""

    types=set()
    dt = h5py.special_dtype(vlen=str)
    try:
        h5 = h5py.File(fname, 'r')
        have_type = '/data_descr/types' in h5
        all_types = set(h5['/data_descr/types'])
        for o in h5['/data_descr/ordering']:
            indptr_name='/data/' + o + '_indptr'
            indices_name='/data/' + o + '_indices'
            if indptr_name in h5 and indices_name in h5:
                types += 'Sparse Matrix'
            else:
                if have_type and o in all_types:
                    types += h5['/data_descr/types'][o]
                else:
                    t=h5['/data/' + o].dtype
                    if t==dt:
                        types.add("String")
                    elif t in (numpy.int64, numpy.int32):
                        types.add("Integer")
                    elif t in (numpy.float64, numpy.float32):
                        types.add("Floating Point")
                    else:
                        types.add(str(t))
        h5.close()
    except:
        pass

    return ','.join(list(types))

def get_correct(fname, test_idx, output_variables):
    """Get correct results from given file from given example indices at given attribute index.

    Used for solving tasks.

    @param fname: name of Data file
    @type fname: string
    @param test_idx: example indices for the test
    @type test_idx list of integer
    @param output_variables: index of attribute to get values for
    @type output_variables: integer
    @return: correct results
    @rtype: list of integer
    """
    dset = _find_dset(fname, output_variables)
    correct = []

    for idx in test_idx:
        correct.append(dset[idx])

    return correct
