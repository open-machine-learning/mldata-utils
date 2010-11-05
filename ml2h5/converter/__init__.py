"""
Convert from and to HDF5 (spec of mldata.org)
"""

ALLOWED_SEPERATORS = (None, ',', ' ', '\t')

TO_H5 = ['libsvm', 'arff', 'csv', 'matlab', 'octave', 'uci']
FROM_H5 = ['libsvm','arff', 'csv', 'matlab', 'octave', 'xml']
EPSILON = 1e-15

#1 MB is maximum line len for autodetection
AUTODETECTION_MAXBUFLEN=1*1024*1024

import os, sys, numpy, h5py
from gettext import gettext as _
from scipy.sparse import csc_matrix
#from decimal import Decimal, InvalidOperation

import ml2h5.fileformat
from h5_arff import H5_ARFF
from h5_libsvm import H5_LibSVM
from h5_csv import H5_CSV
from h5_mat import H5_MAT
from h5_octave import H5_OCTAVE
from h5_uci import H5_UCI
from basehandler import BaseHandler

HANDLERS = {
    'libsvm': H5_LibSVM,
    'arff': H5_ARFF,
    'csv': H5_CSV,
    'matlab': H5_MAT,
    'octave' : H5_OCTAVE,
    'uci' : H5_UCI,
    'h5': BaseHandler
}

class ConversionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Converter(object):
    """Convert and verify conversion.

    @ivar fname_in: name of in-file
    @type fname_in: string
    @ivar fname_out: name of out-file
    @type fname_out: string
    @ivar format_in: format of in-file
    @type format_in: string
    @ivar format_out: format of out-file
    @type format_out: string
    @ivar handler_in: handler object to read data in a specific format
    @type handler_in: derivate of BaseHandler
    @ivar handler_out: handler object to write data in a specific format
    @type handler_out: derivate of BaseHandler
    """

    def __init__(self, fname_in, fname_out, format_in=None, format_out=None, seperator=None, attribute_names_first=False,merge=True):
        """
        @param fname_in: name of in-file
        @type fname_in: string
        @param fname_out: name of out-file
        @type fname_out: string
        @param format_in: format of in-file
        @type format_in: string
        @param format_out: format of out-file
        @type format_out: string
        @param seperator: seperator to seperate variables in examples
        @type seperator: string
        @param attribute_names_first: first line contains attributes (for CSV only)
        @type attribute_names_first: boolean
        """
        self.fname_in = fname_in
        self.fname_out = fname_out
        if format_in:
            self.format_in = format_in
        else:
            self.format_in = ml2h5.fileformat.get(fname_in)
        if format_out:
            self.format_out = format_out
        else:
            self.format_out = ml2h5.fileformat.get(fname_out)

        try:
            self.handler_in = HANDLERS[self.format_in](fname_in, seperator,merge=merge)

            if self.format_in == 'csv':
                self.handler_in.attribute_names_first = attribute_names_first
            self.handler_out = HANDLERS[self.format_out](fname_out, seperator,merge=merge)
            if self.format_out == 'csv':
                self.handler_out.attribute_names_first = attribute_names_first
        except KeyError:
            raise ConversionError(
                'Unknown conversion pair %s to %s!' % (self.format_in, self.format_out))
        except Exception, e: # reformat all other exceptions to ConversionError
            raise ConversionError, ConversionError(str(e)), sys.exc_info()[2]


    def run(self, verify=False, remove_out=True):
        """Convert to/from HDF5.

        @param verify: verify if data in output is same as data in input
        @type verify: boolean
        @param remove_out: if output file shall be removed before running.
        @type remove_out: boolean
        """
        # sometimes it seems files are not properly overwritten when opened by
        # 'w' during run().
        if remove_out and os.path.exists(self.fname_out):
            os.remove(self.fname_out)

        try:
                
            data = self.handler_in.read()
            self.handler_out.write(data)
#            if verify:
#                self.verify()
        except Exception, e: # reformat all exceptions to ConversionError
            raise ConversionError, ConversionError(str(e)), sys.exc_info()[2]


    def _compare(self, A, B):
        """Compare given matrices A and B.

        Used by verification process.

        @param A: list A to compare
        @type A: list of list
        @param B: list B to compare
        @type B: list of list
        """
        if type(A) == csc_matrix:
            try:
                if all(A.indices==B.indices) and all(A.indptr==B.indptr) and all(A.data==B.data):
                    return True
                else:
                    return False
            except:
                return False

        if type(A[0])==numpy.ndarray:
            xrange_A0 = xrange(len(A[0]))
            for i in xrange(len(A)):
                Ai = A[i]
                Bi = B[i]
                for j in xrange_A0:
                    try:
                        if abs(Ai[j] - Bi[j]) > EPSILON:
                            return False
                    except TypeError: #string
                        if str(Ai[j]) != str(Bi[j]):
                            return False
        else:
            for j in xrange(len(A)):
                try:
                    if abs(A[j] - B[j]) > EPSILON:
                        return False
                except TypeError: #string
                    if str(A[j]) != str(B[j]):
                        return False
        return True


    def verify(self):
        """Verify that data in given files is the same.

        @return: true if verification succeeds
        @rtype: boolean
        @raises: ConversionError
        """
        if self.format_in == 'uci':
            raise ConversionError('Cannot verify UCI data format, %s!' % self.fname_in)
        if self.format_out == 'uci':
            raise ConversionError('Cannot verify UCI data format, %s!' % self.fname_out)
        data_in = self.handler_in.read()
        data_out = self.handler_out.read()
        if 'label' in data_in:
            if not self._compare(data_in['label'], data_out['label']):
                raise ConversionError(
                    'Verification failed! Labels of %s != %s' % (self.fname_in, self.fname_out)
                )
        
        for i in xrange(len(data_in['ordering'])):
            name_in = data_in['ordering'][i]
            name_out = data_out['ordering'][i]
            if not self._compare(data_in['data'][name_in], data_out['data'][name_out]):
                raise ConversionError(
                    'Verification failed! Data of %s != %s' % (self.fname_in, self.fname_out)
            )

        return True
