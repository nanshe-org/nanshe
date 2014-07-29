# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 4, 2014 11:10:55 AM$"

import copy

import h5py

import HDF5_serializers

# Need in order to have logging information no matter what.
import debugging_tools


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)


@debugging_tools.log_class(logger)
class EmptyArrayRecorder(object):
    def __init__(self):
        self.__recorders = set()

    def __nonzero__(self):
        return(False)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    def get(self, key, default=None):
        value = default

        try:
            value = self.__getitem__(key)
        except KeyError:
            pass

        return(value)

    def __contains__(self, key):
        return(key in self.__recorders)

    def __getitem__(self, key):
        if key in self.__recorders:
            return(EmptyArrayRecorder())
        else:
            raise(KeyError("unable to open object (Symbol table: Can't open object " + repr(key) + ")"))

    def __setitem__(self, key, value):
        # Exception will be thrown if value is empty or if key already exists (as intended).
        if (value is None) or (value is h5py.Group):
            self.__recorders.add(key)
        else:
            if value.size:
                pass
            else:
                raise ValueError("The array provided for output by the name: \"" + key + "\" is empty.")


@debugging_tools.log_class(logger)
class HDF5ArrayRecorder(object):
    def __init__(self, hdf5_handle, overwrite = False):
        self.hdf5_handle = hdf5_handle
        self.overwrite = overwrite

    def __nonzero__(self):
        return(True)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    def get(self, key, default=None):
        value = default

        try:
            value = self.__getitem__(key)
        except KeyError:
            pass

        return(value)

    def __contains__(self, key):
        return(key in self.hdf5_handle)

    def __getitem__(self, key):
        try:
            if isinstance(self.hdf5_handle[key], h5py.Group):
                return(HDF5ArrayRecorder(self.hdf5_handle[key], overwrite = self.overwrite))
            else:
                return(HDF5_serializers.read_numpy_structured_array_from_HDF5(self.hdf5_handle, key))
        except:
            raise(KeyError("unable to open object (Symbol table: Can't open object " + repr(key) + " in " + repr(self.hdf5_handle) + ")"))

    def __setitem__(self, key, value):
        if (value is None) or (value is h5py.Group):
            # Check to see if the output must go somewhere special.
            if key:
                # If so, check to see if it exists.
                if key in self.hdf5_handle:
                    # If it does and we want to overwrite it, do so.
                    if self.overwrite:
                        del self.hdf5_handle[key]

                        self.hdf5_handle.create_group(key)

                        self.hdf5_handle.file.flush()
                else:
                    # Create it if it doesn't, exist.
                    self.hdf5_handle.create_group(key)

                    self.hdf5_handle.file.flush()
        else:
            # Attempt to create a dataset in self.hdf5_handle named key with value and do not overwrite.
            # Exception will be thrown if value is empty or if key already exists (as intended).
            if value.size:
                HDF5_serializers.create_numpy_structured_array_in_HDF5(self.hdf5_handle,
                                                                      key,
                                                                      value,
                                                                      overwrite = self.overwrite)
                self.hdf5_handle.file.flush()

                return()
            else:
                raise ValueError("The array provided for output by the name: \"" + key + "\" is empty.")


@debugging_tools.log_call(logger)
def generate_HDF5_array_recorder(hdf5_handle, group_name = "", enable = True, overwrite_group = False, allow_overwrite_dataset = False):
    """
        Generates a function used for writing arrays (structured or otherwise)
        to a group in an HDF5 file.
        
        Args:
            hdf5_handle:        The HDF5 file group to place the debug contents into.

            group_name:         The name of the group within hdf5_handle to save the contents to.
                                (If set to the empty string, data will be saved to hdf5_handle directly)

            enable:             Whether to generate a real logger or a fake one.
            
            debug:              Whether to actually write the debug contents (True by default).
            
            overwrite_group:    Whether to replace the debug group if it already exists.

        Returns:
            A function, which will take a given array name and value and write them out.
    """

    if isinstance(hdf5_handle, str):
        hdf5_handle = h5py.File(hdf5_handle, "a")

    if (enable):
        hdf5_recording_handle = hdf5_handle

        # Check to if the output must go somewhere special.
        if group_name:
            # If so, check to see if it exists.
            if group_name in hdf5_handle:
                # If it does and we want to overwrite it, do so.
                if overwrite_group:
                    del hdf5_handle[group_name]

                    hdf5_handle.create_group(group_name)

                    hdf5_handle.file.flush()
            else:
                # Create it if it doesn't, exist.
                hdf5_handle.create_group(group_name)

                hdf5_handle.file.flush()

            hdf5_recording_handle = hdf5_handle[group_name]

        return(HDF5ArrayRecorder(hdf5_recording_handle, overwrite_dataset = allow_overwrite_dataset))
    else:
        return(EmptyArrayRecorder())


@debugging_tools.log_call(logger)
