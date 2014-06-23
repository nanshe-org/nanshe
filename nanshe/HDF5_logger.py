# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 4, 2014 11:10:55 AM$"

import copy

import h5py

import HDF5_serializers

# Need in order to have logging information no matter what.
import advanced_debugging


# Get the logger
logger = advanced_debugging.logging.getLogger(__name__)


class EmptyArrayLogger(object):
    @advanced_debugging.log_call(logger)
    def __init__(self):
        pass

    @advanced_debugging.log_call(logger)
    def __nonzero__(self):
        return(False)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    @advanced_debugging.log_call(logger)
    def __contains__(self, array_name):
        return(False)

    @advanced_debugging.log_call(logger)
    def __call__(self, array_name, array_value):
        # Exception will be thrown if array_value is empty or if array_name already exists (as intended).
        if array_value.size:
            pass
        else:
            raise Exception("The array provided for output by the name: \"" + array_name + "\" is empty.")


class HDF5ArrayLogger(object):
    @advanced_debugging.log_call(logger)
    def __init__(self, hdf5_handle):
        self.hdf5_handle = hdf5_handle

    @advanced_debugging.log_call(logger)
    def __nonzero__(self):
        return(True)

    # For forward compatibility with Python 3
    __bool__ = __nonzero__

    @advanced_debugging.log_call(logger)
    def __contains__(self, array_name):
        return(array_name in self.hdf5_handle)

    @advanced_debugging.log_call(logger)
    def __call__(self, array_name, array_value):
        # Attempt to create a dataset in self.hdf5_handle named array_name with array_value and do not overwrite.
        # Exception will be thrown if array_value is empty or if array_name already exists (as intended).
        if array_value.size:
            HDF5_serializers.write_numpy_structured_array_to_HDF5(self.hdf5_handle, array_name, array_value, overwrite = False)
            self.hdf5_handle.file.flush()
        else:
            raise Exception("The array provided for output by the name: \"" + array_name + "\" is empty.")


@advanced_debugging.log_call(logger)
def generate_HDF5_array_logger(hdf5_handle, group_name = "", enable = True, overwrite_group = False):
    """
        Generates a function used for writing arrays (structured or otherwise)
        to a group in an HDF5 file.
        
        Args:
            hdf5_handle:        The HDF5 file group to place the debug contents into.

            group_name:         The name of the group within hdf5_handle to save the contents to.
                                (If set to the empty string, data will be saved to hdf5_handle directly)
            
            debug:              Whether to actually write the debug contents (True by default).
            
            overwrite_group:    Whether to replace the debug group if it already exists.

        Returns:
            A function, which will take a given array name and value and write them out.
    """

    if isinstance(hdf5_handle, str):
        hdf5_handle = h5py.File(hdf5_handle, "a")

    if (enable):
        hdf5_logging_handle = hdf5_handle

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

            hdf5_logging_handle = hdf5_handle[group_name]

        return(HDF5ArrayLogger(hdf5_logging_handle))
    else:
        return(EmptyArrayLogger())


@advanced_debugging.log_call(logger)
def create_subgroup_HDF5_array_logger(group_name, array_debug_logger, overwrite_group = False):
    """
        Generates a function used for writing arrays (structured or otherwise)
        to a group within the current group in an HDF5 file.
        
        Args:
            hdf5_handle:                The HDF5 file group to place the debug contents into.
            
            debug:              Whether to actually write the debug contents (True by default).
            
            group_name:         The name of the group within hdf5_handle to save the contents to.
                                (If set to the empty string, data will be saved to hdf5_handle directly)
            
            overwrite_group:    Whether to replace the debug group if it already exists.

        Returns:
            A function, which will take a given array name and value and write them out the new directory location.
    """

    # Must be a local import. Otherwise log_call will be undefined in HDF5_serializers.
    import HDF5_serializers

    if array_debug_logger:
        new_array_debug_logger = copy.copy(array_debug_logger)

        # Check to if the output must go somewhere special.
        if group_name:
            # If so, check to see if it exists.
            if group_name in new_array_debug_logger.hdf5_handle:
                # If it does and we want to overwrite it, do so.
                if overwrite_group:
                    del new_array_debug_logger.hdf5_handle[group_name]

                    new_array_debug_logger.hdf5_handle.create_group(group_name)

                    new_array_debug_logger.hdf5_handle.file.flush()
            else:
                # Create it if it doesn't, exist.
                new_array_debug_logger.hdf5_handle.create_group(group_name)

                new_array_debug_logger.hdf5_handle.file.flush()

            new_array_debug_logger.hdf5_handle = new_array_debug_logger.hdf5_handle[group_name]

        return(new_array_debug_logger)
    else:
        return(array_debug_logger)