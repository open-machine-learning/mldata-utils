"""
Handle Data objects and files.

This module heavily relies on the functionality required for http://mldata.org
"""

import h5py, numpy, os, tarfile, zipfile, bz2, gzip
from scipy.sparse import csc_matrix

import fileformat, converter
from . import NUM_EXTRACT


def get_num_instattr(fname):
    """Retrieve number of instances and number of attributes from given HDF5 file.

    @param fname: filename to retrieve data from
    @type fname: string
    @return: number of instances and number of attributes
    @rtype: tuple containing 2 integers
    """
    try:
        h5 = h5py.File(fname, 'r')
        if 'indptr' in h5['data']: # sparse
            A = csc_matrix(
                (h5['/data/data'], h5['/data/indices'], h5['/data/indptr'])
            )
            num_inst = A.shape[1]
            num_attr = A.shape[0]
        else:
            num_inst = 0
            num_attr = 0
            # this seems a bit non-intuitive aka magic...
            for name in h5['/data_descr/ordering']:
                if name == 'label': continue
                if len(h5['/data'][name].shape) == 2:
                    num_attr += h5['/data'][name].shape[0]
                    inst = h5['/data'][name].shape[1]
                    if inst > num_inst:
                        num_inst = inst
                else:
                    num_inst = h5['/data'][name].shape[0]
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
    extract = []
    try:
        if ('indptr' and 'indices') in h5['/data']:
            # taking all data takes to long for quick viewing, but having just
            # this extract may result in less columns displayed than indicated
            # by attributes_names
            data = h5['/data/data'][:h5['/data/indptr'][NUM_EXTRACT+1]]
            indices = h5['/data/indices'][:h5['/data/indptr'][NUM_EXTRACT+1]]
            indptr = h5['/data/indptr'][:NUM_EXTRACT+1]
            A=csc_matrix((data, indices, indptr)).todense().T
            extract = A[:NUM_EXTRACT].tolist()
        else:
            for dset in h5['/data_descr/ordering']:
                if dset == 'label': continue
                dset = 'data/' + dset
                if type(h5[dset][0]) == numpy.ndarray:
                    last = len(h5[dset][0])
                    if last > NUM_EXTRACT: last = NUM_EXTRACT
                    for i in xrange(len(h5[dset])):
                        extract.append(h5[dset][i][:last])
                else:
                    last = len(h5[dset])
                    if last > NUM_EXTRACT: last = NUM_EXTRACT
                    extract.append(h5[dset][:last])
            extract = numpy.matrix(extract).T

        # convert from numpy array to list, if necessary
        t = type(extract[0])
        if t == numpy.ndarray or t == numpy.matrix:
            extract = [y for x in extract for y in x.tolist()]

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
    if format != 'h5':
        fname_h5 = fileformat.get_filename(fname)
        try:
            sep = fileformat.infer_seperator(fname)
            c = converter.Converter(fname, fname_h5, format_in=format, seperator=sep)
            c.run()
        except converter.ConversionError:
            return get_unparseable(fname)
    else:
        fname_h5 = fname

    extract = {
        'mldata': 0,
        'name': '',
        'comment': '',
        'names': [],
        'types': [],
        'data': [],
    }
    try:
        h5 = h5py.File(fname, 'r')
    except:
        return extract

    attrs = ('mldata', 'name', 'comment')
    for a in attrs:
        if a in h5.attrs:
            extract[a] = h5.attrs[a]

    if '/data_descr/names' in h5:
        extract['names'] = h5['/data_descr/names'][:].tolist()
        if 'label' in extract['names']:
            extract['names'].remove('label')

    if '/data_descr/types' in h5:
        extract['types'] = h5['/data_descr/types'][:].tolist()

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
        intro = 'Unparseable Data'
        if is_binary(fname):
            data = ''
        else:
            f = open(fname, 'r')
            i = 0
            data = []
            for l in f:
                data.append(l)
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
    h5 = h5py.File(fname, 'r')
    dset = None

    if 'indptr' in h5['data']: # sparse
        A = csc_matrix(
            (h5['/data/data'], h5['/data/indices'], h5['/data/indptr'])
        ).todense()
        dset = A[output_variables].tolist()
    else:
        ov = output_variables
        for name in h5['/data_descr/ordering']:
            path = '/data/' + name
            if name == 'label': # labels need to be transposed
                data = h5[path][...].T
            else:
                data = h5[path][...]

            if len(data.shape) == 1: # datasets with shape (x,)
                if ov == 0:
                    dset = data
                else:
                    ov -= 1
            else: # datasets with shape (x,y)
                for i in xrange(len(data)):
                    if ov == 0:
                        dset = data[i].tolist()
                        break # inner loop
                    else:
                        ov -= 1

            if dset is not None:
                break # outer loop

    h5.close()
    return dset


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
