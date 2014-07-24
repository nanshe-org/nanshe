# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 4, 2014 11:10:55 AM$"

import copy

import numpy
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


@debugging_tools.log_class(logger)
class HDF5EnumeratedArrayRecorder(object):
    def __init__(self, hdf5_handle):
        self.hdf5_handle = hdf5_handle

        # Must be a logger if it already exists.
        assert(self.hdf5_handle.attrs.get("is_logger", True))

        self.hdf5_handle.attrs["is_logger"] = True
        self.hdf5_handle.file.flush()

        self.hdf5_index_data_handles = {"." : -1}
        for each_index in self.hdf5_handle:
            each_index = int(each_index)
            self.hdf5_index_data_handles["."] = max(self.hdf5_index_data_handles["."], each_index)

        if self.hdf5_index_data_handles["."] != -1:
            hdf5_index_handle = self.hdf5_handle[str(self.hdf5_index_data_handles["."])]
            for each_key in hdf5_index_handle:
                if hdf5_index_handle[each_key].attrs.get("is_logger", False):
                    self.hdf5_index_data_handles[each_key] = None
                else:
                    self.hdf5_index_data_handles[each_key] = -1

                    for each_index in hdf5_index_handle[each_key]:
                        each_index = int(each_index)
                        self.hdf5_index_data_handles[each_key] = max(self.hdf5_index_data_handles[each_key], each_index)

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
            root_i = self.hdf5_index_data_handles.get(".", -1)
            key_i = self.hdf5_index_data_handles.get(key, -1)

            assert(root_i != -1)
            assert(key_i != -1)

            root_i_str = str(root_i)

            assert(isinstance(self.hdf5_handle[root_i_str], h5py.Group))
            assert(isinstance(self.hdf5_handle[root_i_str][key], h5py.Group))

            key_handle = self.hdf5_handle[root_i_str][key]
            if key_i is None:
                return(HDF5EnumeratedArrayRecorder(key_handle))
            else:
                key_i_str = str(key_i)
                return(HDF5_serializers.read_numpy_structured_array_from_HDF5(key_handle, key_i_str))
        except KeyError:
            raise(KeyError("unable to open object (Symbol table: Can't open object " + repr(key) + " in " + repr(self.hdf5_handle) + ")"))

    def __setitem__(self, key, value):
        if (key == "."):
            if not ( (value is None) or (value is h5py.Group) ):
                raise ValueError("Cannot store dataset in top level group ( " + self.hdf5_handle.name + " ).")

            self.hdf5_index_data_handles = { "." : self.hdf5_index_data_handles["."] + 1 }
            self.hdf5_handle.create_group(str(self.hdf5_index_data_handles["."]))
            self.hdf5_handle.file.flush()
        else:
            hdf5_index_handle = None
            try:
                hdf5_index_handle = self.hdf5_handle[str(self.hdf5_index_data_handles["."])]
            except KeyError:
                if self.hdf5_index_data_handles["."] == -1:
                    self.hdf5_index_data_handles = { "." : self.hdf5_index_data_handles["."] + 1 }

                self.hdf5_handle.create_group(str(self.hdf5_index_data_handles["."]))
                self.hdf5_handle.file.flush()

                hdf5_index_handle = self.hdf5_handle[str(self.hdf5_index_data_handles["."])]

            if (value is None) or (value is h5py.Group):
                # Create a group if it doesn't already exist.
                hdf5_index_handle.require_group(key)
                hdf5_index_handle.attrs["is_logger"] = True
                hdf5_index_handle.file.flush()
                self.hdf5_index_data_handles[key] = None
            else:
                # Attempt to create a dataset in self.hdf5_handle named key with value and do not overwrite.
                # Exception will be thrown if value is empty or if key already exists (as intended).

                # Index into a NumPy structured array can return a void type even though it is a valid array, which can
                # be stored. So, we must check.
                try:
                    assert(isinstance(value, numpy.ndarray))
                except AssertionError:
                    if not value.dtype.names:
                        raise
                if value.size:
                    # If so, check to see if it exists.
                    if key not in self.hdf5_index_data_handles:
                        hdf5_index_handle.create_group(key)
                        hdf5_index_handle[key].attrs["is_logger"] = False
                        hdf5_index_handle.file.flush()
                        self.hdf5_index_data_handles[key] = -1

                    self.hdf5_index_data_handles[key] += 1

                    HDF5_serializers.create_numpy_structured_array_in_HDF5(hdf5_index_handle[key],
                                                                           str(self.hdf5_index_data_handles[key]),
                                                                           value)

                    self.hdf5_handle.file.flush()
                else:
                    raise ValueError("The array provided for output by the name: \"" + key + "\" is empty.")


@debugging_tools.log_call(logger)
def generate_HDF5_array_recorder(hdf5_handle, group_name = "", enable = True, overwrite_group = False, recorder_constructor = HDF5ArrayRecorder, **kwargs):
    """
        Generates a function used for writing arrays (structured or otherwise)
        to a group in an HDF5 file.
        
        Args:
            hdf5_handle:            The HDF5 file group to place the debug contents into.

            group_name:             The name of the group within hdf5_handle to save the contents to.
                                    (If set to the empty string, data will be saved to hdf5_handle directly)

            enable:                 Whether to generate a real logger or a fake one.

            overwrite_group:        Whether to overwrite the group where data is stored.

            recorder_constructor:   Type of recorder to use if enable is True.

            **kwargs:               Other arguments to pass through to the recorder_constructor (won't pass through if
                                    enable is false).

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

        return(recorder_constructor(hdf5_recording_handle, **kwargs))
    else:
        return(EmptyArrayRecorder())


@debugging_tools.log_call(logger)
