"""
Utility functions around HDF5, mostly for mldata.org.
"""

__all__ = ['task', 'data', 'fileformat', 'converter']
VERSION_MLDATA = '0'
NUM_EXTRACT = 10
# maximum value length 
LEN_EXTRACT = 6
PROGRAM_NAME='ml2h5'
VERSION='0'


#workaround bugs
def import_h5py():
    h5py = __import__('h5py')
    h5file=h5py.File

    def h5py_file(name, mode=None, driver=None):#, **driver_kwds):
        if not mode:
            mode='r'
        file(name,mode)
        if mode == 'r':
            x=file(name).read(4)
            if not x.endswith('HDF'):
                raise Exception

        return h5file(name,mode,driver)#,driver_kwds)

    h5py.File=h5py_file
    return h5py

import __builtin__
__builtin__.h5py=import_h5py()
