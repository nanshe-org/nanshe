# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 14, 2014 10:19:59PM$"

import numpy
import h5py


# Need in order to have logging information no matter what.
import advanced_debugging


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)


@advanced_debugging.log_call(logger)
def write_numpy_structured_array_to_HDF5(fid, internalPath, data, overwrite = False):
    """
        Serializes a NumPy structure array to an HDF5 file by using the HDF5 compound data type.
        Also, will handle normal NumPy arrays and scalars, as well.
        
        Note:
            HDF5 does not support generic Python objects. So, serialization of objects to something
            else (perhaps strs of fixed size) must be performed first.
        
        Note:
            TODO: Write doctests.
        
        Args:
            fid(HDF5 file):         either an HDF5 file or an HDF5 filename.
            internalPath(str):      an internal path for the HDF5 file.
            data(numpy.ndarray):    the NumPy structure array to save (or normal NumPy array).
            overwrite(bool):        whether to overwrite what is already there (defaults to False).
    """

    close_fid = False

    if isinstance(fid, str):
        fid = h5py.File(fid, "a")
        close_fid = True

    try:
        fid.create_dataset(internalPath, shape = data.shape, dtype = data.dtype, data = data)
    except RuntimeError:
        if overwrite:
            del fid[internalPath]
            fid.create_dataset(internalPath, shape = data.shape, dtype = data.dtype, data = data)
        else:
            raise

    if close_fid:
        fid.close()


@advanced_debugging.log_call(logger)
def read_numpy_structured_array_from_HDF5(fid, internalPath):
    """
        Serializes a NumPy structure array from an HDF5 file by using the HDF5 compound data type.
        Also, it will handle normal NumPy arrays and scalars, as well.
        
        Note:
            HDF5 does not support generic Python objects. So, serialization of objects to something
            else (perhaps strs of fixed size) must be performed first.
        
        Args:
            fid(HDF5 file):         either an HDF5 file or an HDF5 filename.
            internalPath(str):      an internal path for the HDF5 file.
        
        Note:
            TODO: Write doctests.
        
        Returns:
            data(numpy.ndarray):    the NumPy structure array.
    """

    close_fid = False

    if isinstance(fid, str):
        fid = h5py.File(fid, "r")
        close_fid = True

    data = fid[internalPath].value

    if close_fid:
        fid.close()

    return(data)