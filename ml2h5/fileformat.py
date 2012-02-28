import h5py
import os.path
import gzip
from ml2h5.converter.basehandler import ALLOWED_SEPERATORS
from ml2h5.converter import AUTODETECTION_MAXBUFLEN

def _try_suffix(fname):
    """Get format of given file by suffix.

    @param fname: name of file to determine format for
    @type fname: string
    """
    suffix = fname.split('.')[-1]
    # just assume libsvm if no proper suffix
    if suffix.find('/') != -1:
        return '', False

    if suffix in ('svm', 'libsvm', 'light', 'svmlight'):
        return 'libsvm', True
    elif suffix in ('arff'):
        return 'arff', True
    elif suffix in ('h5', 'hdf5'):
        return 'h5', True
    elif suffix in ('csv', 'tsv'):
        return 'csv', True
    #elif suffix in ('uci', 'data'):
    #    return 'uci', True
    elif suffix in ('zip', 'tgz'):
        return suffix, True
    elif suffix in ('bz2', 'gz'):
        try:
            presuffix = fname.split('.')[-2]
            if presuffix == 'tar':
                return presuffix + '.' + suffix, True
        except IndexError:
            pass
        return suffix, True
    elif suffix in ('mat'):
        return 'matlab', True
    elif suffix in ('octave'):
        return 'octave', True
    elif suffix in ('xml'):
        return 'xml', True
    elif suffix in ('RData', 'rdata'):
        return 'rdata', True
    else: # unknown
        return suffix, False


def _try_arff(fname):
    """Try if given file is in arff format

    @param fname: name of file to determine format for
    @type fname: string
    """
    try:
        import arff
        arff.ArffFile.load(fname)
        return True
    except:
        return False


def _try_csv(fname):
    """Try if given file is in csv format

    @param fname: name of file to determine format for
    @type fname: string
    """
    if infer_seperator(fname) == ',':
        return True
    else:
        return False


def _try_libsvm(fname):
    """Try if given file is in libsvm format

    @param fname: name of file to determine format for
    @type fname: string
    """
    try:
        fp = open(fname, 'r')
    except:
        return False

    line = fp.readline(AUTODETECTION_MAXBUFLEN)
    attributes = line.split()
    if len(attributes) > 1:
        # 0th might be label, so look at 1st
        if len(attributes[1].split(':')) == 2:
            fp.close()
            return True

    fp.close()
    return False


def _try_h5(fname):
    """Try if given file is in hdf5 format

    @param fname: name of file to determine format for
    @type fname: string
    """

    return h5py.is_hdf5(fname)

def _try_matlab(fname):
    try:
        return file(fname).read(6) == 'MATLAB'
    except:
        return False

def _try_octave(fname):
    try:
        return file(fname).read(13)==('# Created by ')
    except:
        return False

def _try_rdata(fname):
    try:
        return gzip.GzipFile(fname).read(4)==('RDX2')
    except:
        return False

def infer_seperator(fname):
    """Infer seperator for variables in given file.

    @param fname: filename to retrieve data from
    @type fname: string
    @return: inferred seperator
    @rtype: string
    """
    try:
        fp = open(fname, 'r')
    except:
        return None

    seperator = None
    minimum = 1

    for i in xrange(100): # try the first 100 lines
        line = fp.readline(AUTODETECTION_MAXBUFLEN)
        if not line:
            break
        for s in ALLOWED_SEPERATORS:
            l = len(line.split(s))
            if l > minimum:
                minimum = l
                seperator = s
        if seperator:
            break

        # stop processing if lines are too long anyways
        if not line.endswith('\n'):
            break

    fp.close()
    return seperator


def get(fname, skip_suffix=False):
    """Get format of given file.

    By suffix it detects: libsvm, arff, csv, h5, tar.gz, tar.bz2, zip,
    matlab, octave.
    By deeper inspection it detects: h5, matlab, octave, arff, csv, libsvm

    @param fname: name of file to determine format for
    @type fname: string
    @param skip_suffix: if detection by suffix (first priority) shall be skipped
    @type skip_suffix: boolean
    """
    if not skip_suffix:
        format, found = _try_suffix(fname)
        if found:
            return format

    if _try_h5(fname): return 'h5'
    elif _try_matlab(fname): return 'matlab'
    elif _try_rdata(fname): return 'rdata'
    elif _try_octave(fname): return 'octave'
    elif _try_libsvm(fname): return 'libsvm'
    elif _try_csv(fname): return 'csv'
    elif _try_arff(fname): return 'arff'

    return 'unknown'


def get_filename(orig):
    """Convert a given filename to something that indicates HDF5.

    @param orig: original filename
    @type orig: string
    @return: HDF5-ified filename
    @rtype: string
    """
    return os.path.splitext(orig)[0] + '.h5'


def can_convert_h5_to(dst_type, h5_filename=None):
    """Whether conversion from this particular h5 file to dst type is supported

    @param h5_filename: h5 filename or None
    @type h5_filename: string
    @param dst_type: name of type
    @type dst_type: string
    @return: True if possible
    """

    if dst_type in ('matlab', 'octave'):
        return True

    if h5_filename and h5_filename.endswith('.h5') and h5py.is_hdf5(h5_filename):
        try:
            h5 = h5py.File(h5_filename, 'r')

            if dst_type=='libsvm': # libsvm requires data/label
                ordering=set(('label','data'))
                if ordering.issubset(set(h5['data'].keys())):
                    return True # TODO check if this is sparse data / ndarray data
            elif dst_type in ('csv', 'arff', 'rdata'): # csv/arff/RData support everything except sparse data
                for k in h5['data'].keys():
                    if k.endswith('_indptr') or k.endswith('_indices'):
                        return False
                return True
        except:
            pass
    return False
