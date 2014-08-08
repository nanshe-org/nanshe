# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 14, 2014 10:19:59PM$"


import os

import numpy
import h5py


# Need in order to have logging information no matter what.
import debugging_tools


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)


@debugging_tools.log_call(logger)
def create_numpy_structured_array_in_HDF5(file_handle, internalPath, data, overwrite = False):
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

    if isinstance(file_handle, str) or isinstance(file_handle, unicode):
        file_handle = h5py.File(file_handle, "a")
        close_file_handle = True

    data_array = data
    if not isinstance(data_array, numpy.ndarray):
        try:
            data_array = numpy.ndarray(data_array)
        except:
            if not data_array.dtype.names:
                raise TypeError("The argument provided for data is type: \"" + repr(type(data)) + "\" is not convertible to type \"" + repr(numpy.ndarray) + "\".")


    try:
        file_handle.create_dataset(internalPath, shape = data_array.shape, dtype = data_array.dtype, data = data_array)
    except RuntimeError:
        if overwrite:
            del file_handle[internalPath]
            file_handle.create_dataset(internalPath, shape = data_array.shape, dtype = data_array.dtype, data = data_array)
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

    if isinstance(file_handle, str) or isinstance(file_handle, unicode):
        file_handle = h5py.File(file_handle, "r")
        close_file_handle = True

    data = None

    data_object = file_handle[internalPath]
    data_file = data_object.file
    data_ref = data_object.value
    # data_ref = data_object[()]

    if isinstance(data_ref, (numpy.number, numpy.ndarray,)):
        data = data_object.value
    elif isinstance(data_ref, h5py.Reference):
        if ("filename" in data_object.attrs) and \
           (os.path.normpath(data_object.attrs["filename"]) != os.path.normpath(data_file.filename)):
            with h5py.File(data_object.attrs["filename"], "r") as external_file_handle:
                if isinstance(data_ref, h5py.RegionReference):
                    data = external_file_handle[data_ref][data_ref]
                else:
                    data = external_file_handle[data_ref].value
        else:
            if isinstance(data_ref, h5py.RegionReference):
                data = data_file[data_ref][data_ref]
            else:
                data = data_file[data_ref].value

    if close_file_handle:
        file_handle.close()

    return(data)