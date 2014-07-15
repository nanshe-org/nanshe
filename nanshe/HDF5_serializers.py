# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 14, 2014 10:19:59PM$"

import numpy
import h5py


# Need in order to have logging information no matter what.
import debugging_tools


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)


@debugging_tools.log_call(logger)
def write_numpy_structured_array_to_HDF5(file_handle, internalPath, data, overwrite = False):
    """
        Serializes a NumPy structure array to an HDF5 file by using the HDF5 compound data type.
        Also, will handle normal NumPy arrays and scalars, as well.
        
        Note:
            HDF5 does not support generic Python objects. So, serialization of objects to something
            else (perhaps strs of fixed size) must be performed first.
        
        Note:
            TODO: Write doctests.
        
        Args:
            file_handle(HDF5 file):     either an HDF5 file or an HDF5 filename.
            internalPath(str):          an internal path for the HDF5 file.
            data(numpy.ndarray):        the NumPy structure array to save (or normal NumPy array).
            overwrite(bool):            whether to overwrite what is already there (defaults to False).
    """

    close_file_handle = False

    if isinstance(file_handle, str):
        file_handle = h5py.File(file_handle, "a")
        close_file_handle = True

    try:
        file_handle.create_dataset(internalPath, shape = data.shape, dtype = data.dtype, data = data)
    except RuntimeError:
        if overwrite:
            del file_handle[internalPath]
            file_handle.create_dataset(internalPath, shape = data.shape, dtype = data.dtype, data = data)
        else:
            raise

    if close_file_handle:
        file_handle.close()


@debugging_tools.log_call(logger)
def read_numpy_structured_array_from_HDF5(file_handle, internalPath):
    """
        Serializes a NumPy structure array from an HDF5 file by using the HDF5 compound data type.
        Also, it will handle normal NumPy arrays and scalars, as well.
        
        Note:
            HDF5 does not support generic Python objects. So, serialization of objects to something
            else (perhaps strs of fixed size) must be performed first.
        
        Args:
            file_handle(HDF5 file):     either an HDF5 file or an HDF5 filename.
            internalPath(str):          an internal path for the HDF5 file.
        
        Note:
            TODO: Write doctests.
        
        Returns:
            data(numpy.ndarray):    the NumPy structure array.
    """

    close_file_handle = False

    if isinstance(file_handle, str):
        file_handle = h5py.File(file_handle, "r")
        close_file_handle = True

    data = file_handle[internalPath].value

    if close_file_handle:
        file_handle.close()

    return(data)