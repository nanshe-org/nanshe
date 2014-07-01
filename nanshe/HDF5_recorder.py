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


class EmptyArrayRecorder(object):
    @debugging_tools.log_call(logger)
    def __init__(self):
        pass

    @debugging_tools.log_call(logger)
    def __nonzero__(self):
        return(False)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    @debugging_tools.log_call(logger)
    def __contains__(self, array_name):
        return(False)

    @debugging_tools.log_call(logger)
    def __getitem__(self, array_name):
        return(None)

    @debugging_tools.log_call(logger)
    def __call__(self, array_name, array_value):
        # Exception will be thrown if array_value is empty or if array_name already exists (as intended).
        if array_value.size:
            pass
        else:
            raise Exception("The array provided for output by the name: \"" + array_name + "\" is empty.")


class HDF5ArrayRecorder(object):
    @debugging_tools.log_call(logger)
    def __init__(self, hdf5_handle, overwrite_dataset = False):
        self.hdf5_handle = hdf5_handle
        self.overwrite_dataset = overwrite_dataset

    @debugging_tools.log_call(logger)
    def __nonzero__(self):
        return(True)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    @debugging_tools.log_call(logger)
    def __contains__(self, array_name):
        return(array_name in self.hdf5_handle)

    @debugging_tools.log_call(logger)
    def __getitem__(self, array_name):
        return(HDF5_serializers.read_numpy_structured_array_from_HDF5(self.hdf5_handle, array_name))

    @debugging_tools.log_call(logger)
    def __call__(self, array_name, array_value):
        # Attempt to create a dataset in self.hdf5_handle named array_name with array_value and do not overwrite.
        # Exception will be thrown if array_value is empty or if array_name already exists (as intended).
        if array_value.size:
            HDF5_serializers.write_numpy_structured_array_to_HDF5(self.hdf5_handle,
                                                                  array_name,
                                                                  array_value,
                                                                  overwrite = self.overwrite_dataset)
            self.hdf5_handle.file.flush()
        else:
            raise Exception("The array provided for output by the name: \"" + array_name + "\" is empty.")


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
def create_subgroup_HDF5_array_recorder(group_name, array_recorder, overwrite_group = False):
    """
        Generates a function used for writing arrays (structured or otherwise)
        to a group within the current group in an HDF5 file.
        
        Args:
            hdf5_handle:        The HDF5 file group to place the debug contents into.
            
            debug:              Whether to actually write the debug contents (True by default).
            
            group_name:         The name of the group within hdf5_handle to save the contents to.
                                (If set to the empty string, data will be saved to hdf5_handle directly)
            
            overwrite_group:    Whether to replace the debug group if it already exists.

        Returns:
            A function, which will take a given array name and value and write them out the new directory location.
    """

    # Must be a local import. Otherwise log_call will be undefined in HDF5_serializers.
    import HDF5_serializers

    if array_recorder:
        new_array_debug_recorder = copy.copy(array_recorder)

        # Check to if the output must go somewhere special.
        if group_name:
            # If so, check to see if it exists.
            if group_name in new_array_debug_recorder.hdf5_handle:
                # If it does and we want to overwrite it, do so.
                if overwrite_group:
                    del new_array_debug_recorder.hdf5_handle[group_name]

                    new_array_debug_recorder.hdf5_handle.create_group(group_name)

                    new_array_debug_recorder.hdf5_handle.file.flush()
            else:
                # Create it if it doesn't, exist.
                new_array_debug_recorder.hdf5_handle.create_group(group_name)

                new_array_debug_recorder.hdf5_handle.file.flush()

            new_array_debug_recorder.hdf5_handle = new_array_debug_recorder.hdf5_handle[group_name]

        return(new_array_debug_recorder)
    else:
        return(array_recorder)