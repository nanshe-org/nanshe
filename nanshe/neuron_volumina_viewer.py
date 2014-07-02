__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 09, 2014 8:51:33AM$"


# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Copyright 2011-2014, the ilastik developers

"""High-level API.

"""


import debugging_tools

logger = debugging_tools.logging.getLogger(__name__)

import os
import collections
import itertools
import threading

import h5py
import numpy

from PyQt4.QtGui import QApplication
from PyQt4.QtCore import pyqtSignal
from PyQt4.QtCore import QObject

from volumina.multimethods import multimethod
from volumina.pixelpipeline.datasources import SourceABC, RequestABC
from volumina.pixelpipeline.datasources import is_pure_slicing
from volumina.layer import GrayscaleLayer, RGBALayer, ColortableLayer, ClickableColortableLayer, AlphaModulatedLayer
from volumina.viewer import Viewer

import HDF5_searchers
import expanded_numpy
import additional_generators



class HDF5DatasetNotFoundException( Exception ):
    """
        An exception raised when a dataset is not found in an HDF5 file.
    """
    pass

class HDF5DataSource( QObject ):
    """
        Creates a source that reads from an HDF5 dataset and shapes it in a way that Volumina can use.

        Attributes:
              file_handle(h5py.File or str):           A handle for reading the HDF5 file or the external file path.
              file_path(str):                          External path to the file
              dataset_path(str):                       Internal path to the dataset
              full_path(str):                          Both external and internal paths combined as one path
              dataset_shape(tuple of ints):            A tuple representing the shape of the dataset in each dimension
              dataset_dtype(numpy.dtype or type):      The type of the underlying dataset.
              axis_order(tuple of ints):               A tuple representing how to reshape the array before returning a request

    """

    #TODO: Reshaping should probably be some sort of lazyflow operator and thus removed from this directly.

    isDirty = pyqtSignal( object )
    numberOfChannelsChanged = pyqtSignal(int) # Never emitted

    @debugging_tools.log_call(logger)
    def __init__( self, file_handle, internal_path, shape = None, dtype = None):
        """
            Constructs an HDF5DataSource using a given file and path to the dataset. Optionally, the shape and dtype
            can be specified.

            Args:
                file_handle(h5py.File or str):          A file handle for the HDF5 file..
                internal_path(str):                     path to the dataset inside of the HDF5 file.
                shape(tuple of ints):                   shape of underlying dataset if not specified defaults to that of the dataset.
                dtype(numpy.dtype or type):             type of underlying dataset if not specified defaults to that of the dataset.
        """
        #TODO: Get rid of shape and dtype as arguments.

        super(HDF5DataSource, self).__init__()

        self.file_handle = None

        self.file_path = ""
        self.dataset_path = ""

        self.full_path = ""

        # Constructed standard shape and dtype if provided
        self.dataset_shape = tuple(shape) if shape is not None else None
        self.dataset_dtype = numpy.dtype(dtype).type if dtype is not None else None

        self.axis_order = [-1, -1, -1, -1, -1]

        # If it is a filename, get the file handle.
        if isinstance(file_handle, str):
            file_handle.rstrip("/")
            file_handle = h5py.File(file_handle)

        self.file_handle = file_handle

        self.file_path = self.file_handle.filename
        self.dataset_path = "/" + internal_path.strip("/")

        self.full_path = self.file_path + self.dataset_path

        # Check to see if the dataset exists in the file. Otherwise throw an Exception for it.
        if self.dataset_path not in self.file_handle:
            raise(HDF5DatasetNotFoundException("Could not find the path \"" + self.dataset_path + "\" in filename " + "\"" + self.file_path + "\"."))

        # Fill in the shape and or dtype information if it doesn't already exist.
        if ( (self.dataset_shape is None) and (self.dataset_dtype is None) ):
            dataset = self.file_handle[self.dataset_path]
            #print dataset.name
            self.dataset_shape = dataset.shape
            self.dataset_dtype = dataset.dtype.type
        elif (self.dataset_shape is None):
            dataset = self.file_handle[self.dataset_path]
            self.dataset_shape = dataset.shape
        elif (self.dataset_dtype is None):
            dataset = self.file_handle[self.dataset_path]
            self.dataset_dtype = dataset.dtype.type

        # Using the shape information, determine how to reshape the axes to present the data as we wish.
        if len(self.dataset_shape) == 1:
            self.axis_order = [-1, 0, -1, -1, -1]
        if len(self.dataset_shape) == 2:
            # self.axis_order = [-1, 0, 1, -1, -1]
            self.axis_order = [-1, 1, 0, -1, -1]
        elif (len(self.dataset_shape) == 3) and (self.dataset_shape[2] <= 4):
            # self.axis_order = [-1, 0, 1, -1, 2]
            self.axis_order = [-1, 1, 0, -1, 2]
        elif len(self.dataset_shape) == 3:
            # self.axis_order = [-1, 1, 2, -1, 0]
            self.axis_order = [-1, 2, 1, -1, 0]
        elif len(self.dataset_shape) == 4:
            # self.axis_order = [-1, 1, 2, 3, 0]
            self.axis_order = [-1, 3, 2, 1, 0]
        elif len(self.dataset_shape) == 5:
            # self.axis_order = [0, 1, 2, 3, 4]
            self.axis_order = [0, 3, 2, 1, 4]
        else:
            raise Exception("Unacceptable shape provided for display. Found shape to be \"" + self.dataset_shape + "\".")

        # Construct the shape to be singleton if the axis order is irrelevant or the appropriate shape for the reordered axis.
        self.dataset_shape = tuple([1 if _ == -1 else self.dataset_shape[_] for _ in self.axis_order])

    @debugging_tools.log_call(logger)
    def numberOfChannels(self):
        return(self.dataset_shape[-1])

    @debugging_tools.log_call(logger)
    def clean_up(self):
        # Close file
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

        self.file_path = None
        self.dataset_path = None
        self.full_path = None
        self.dataset_dtype = None
        self.dataset_shape = None

    @debugging_tools.log_call(logger)
    def dtype(self):
        return(self.dataset_dtype)

    @debugging_tools.log_call(logger)
    def shape(self):
        return(self.dataset_shape)

    @debugging_tools.log_call(logger)
    def request( self, slicing ):
        if not is_pure_slicing(slicing):
            raise Exception('HDF5DataSource: slicing is not pure')

        assert(len(slicing) == len(self.dataset_shape), "Expect a slicing for a txyzc array.")

        additional_generators.reformat_slices(slicing, self.dataset_shape)

        return(HDF5DataRequest(self.file_handle, self.dataset_path, self.axis_order, self.dataset_dtype, slicing))

    @debugging_tools.log_call(logger)
    def setDirty( self, slicing):
        if not is_pure_slicing(slicing):
            raise Exception('dirty region: slicing is not pure')
        self.isDirty.emit( slicing )

    @debugging_tools.log_call(logger)
    def __eq__( self, other ):
        if other is None:
            return False

        return(self.full_path == other.full_path)

    @debugging_tools.log_call(logger)
    def __ne__( self, other ):
        if other is None:
            return True

        return(self.full_path != other.full_path)

assert issubclass(HDF5DataSource, SourceABC)


class HDF5DataRequest( object ):
    """
        Created by an HDF5DataSource to provide a way to request slices of the HDF5 file in a nice way.

        Attributes:
          file_handle(h5py.File or str):           A handle for reading the HDF5 file or the external file path.
          dataset_path(str):                       Internal path to the dataset
          axis_order(tuple of ints):               A tuple representing how to reshape the array before returning a request.
          dataset_dtype(numpy.dtype or type):      The type of the underlying dataset.
          throw_on_not_found(bool):                Whether to throw an exception if the dataset is not found.
          slicing(tuple of slices):                The slicing request by Volumina.
          actual_slicing(tuple of slices):         The actual slicing that will be performed on the dataset.
          throw_on_not_found(bool):                   Whether to throw an exception if the dataset is not found.

        Note:
             Before returning the result to Volumina the axes will likely need to be transposed. Also, singleton axes
             will need to be inserted to ensure the dimensionality is 5 as Volumina expects. This result will be cached
             inside the request instance. So, if this request instance is kept, this won't need to be repeated.

    """

    #TODO: Try to remove throw_on_not_found. This basically would have been thrown earlier. So, we would rather not have this as it is a bit hacky.
    #TODO: Try to remove dataset_dtype as this should be readily available information from the dataset.

    @debugging_tools.log_call(logger)
    def __init__( self, file_handle, dataset_path, axis_order, dataset_dtype, slicing, throw_on_not_found = False ):
        """
            Constructs an HDF5DataRequest using a given file and path to the dataset. Optionally, throwing can be
            suppressed if the source is not found.

            Args:
                file_handle(h5py.File or str):              A file handle for the HDF5 file.
                dataset_path(str):                          Internal path to the dataset
                axis_order(tuple of ints):                  A tuple representing how to reshape the array before returning a request.
                dataset_dtype(numpy.dtype or type):         The type of the underlying dataset.
                slicing(tuple of ints):                     The slicing to extract from the HDF5 file.
                throw_on_not_found(bool):                   Whether to throw an exception if the dataset is not found.
        """

        # TODO: Look at adding assertion check on slices.

        self.file_handle = file_handle
        self.dataset_path = dataset_path
        self.axis_order = axis_order
        self.dataset_dtype = dataset_dtype
        self.throw_on_not_found = throw_on_not_found

        self._result = None

        # Clean up slicing. Here self.slicing is the requested slicing.
        # actual_slicing_dict includes a key for each_axis.
        # To construct the list requires a second pass either way.
        self.slicing = list()
        actual_slicing_dict = dict()
        for i, (each_slice, each_axis) in enumerate(itertools.izip(slicing, self.axis_order)):
            self.slicing.append(each_slice)
            if each_axis != -1:
                actual_slicing_dict[each_axis] = each_slice

        self.slicing = tuple(self.slicing)

        # As the dictionary sorts by keys, we are ensured to have the slices in the order of the axes.
        self.actual_slicing = actual_slicing_dict.values()

        # Convert to tuple as it is expected.
        self.actual_slicing = tuple(self.actual_slicing)

    @debugging_tools.log_call(logger)
    def wait( self ):
        if self._result is None:
            # Construct a result the size of the slicing
            slicing_shape = additional_generators.len_slices(self.slicing)
            self._result = numpy.zeros(slicing_shape, dtype = self.dataset_dtype)

            try:
                dataset = self.file_handle[self.dataset_path]

                a_result = dataset[self.actual_slicing]
                a_result = numpy.array(a_result)

                # Get the axis order without the singleton axes
                the_axis_order = numpy.array(self.axis_order)
                the_axis_order = the_axis_order[the_axis_order != -1]

                # Reorder the axes for Volumina
                a_result = a_result.transpose(the_axis_order)

                # Insert singleton axes to make 5D for Volumina
                for i, each_axis_order in enumerate(self.axis_order):
                    if each_axis_order == -1:
                        a_result = expanded_numpy.add_singleton_axis_pos(a_result, i)

                self._result[:] = a_result
            except KeyError:
                if self.throw_on_not_found:
                   raise

            logger.debug("Found the result.")

        return self._result

    @debugging_tools.log_call(logger)
    def getResult(self):
        return self._result

    @debugging_tools.log_call(logger)
    def cancel( self ):
        pass

    @debugging_tools.log_call(logger)
    def submit( self ):
        pass

    # callback( result = result, **kwargs )
    @debugging_tools.log_call(logger)
    def notify( self, callback, **kwargs ):
        t = threading.Thread(target=self._doNotify, args=( callback, kwargs ))
        t.start()

    @debugging_tools.log_call(logger)
    def _doNotify( self, callback, kwargs ):
        result = self.wait()
        callback(result, **kwargs)
assert issubclass(HDF5DataRequest, RequestABC)


class HDF5Viewer(Viewer):
    """
        Extends the Viewer from Volumina so that it provides some additional features that are nice for HDF5 sources.

        Attributes:
              file_handle(h5py.File or str):           A handle for reading the HDF5 file or the external file path.
              file_path(str):                          External path to the file
              dataset_path(str):                       Internal path to the dataset
              full_path(str):                          Both external and internal paths combined as one path
              dataset_shape(tuple of ints):            A tuple representing the shape of the dataset in each dimension
              dataset_dtype(numpy.dtype or type):      The type of the underlying dataset.
              axis_order(tuple of ints):               A tuple representing how to reshape the array before returning a request

    """

    @debugging_tools.log_call(logger)
    def __init__(self, parent=None):
        super(HDF5Viewer, self).__init__(parent)

    @debugging_tools.log_call(logger)
    def addGrayscaleHDF5Source(self, source, shape, name=None, direct=False):
        self.dataShape = shape
        layer = GrayscaleLayer(source, direct=direct)
        layer.numberOfChannels = self.dataShape[-1]

        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

    @debugging_tools.log_call(logger)
    def addGrayscaleHDF5Layer(self, a, name=None, direct=False):
        source, self.dataShape = createHDF5DataSource(a, True)
        layer = GrayscaleLayer(source, direct=direct)
        layer.numberOfChannels = self.dataShape[-1]

        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

    @debugging_tools.log_call(logger)
    def addAlphaModulatedHDF5Layer(self, a, name=None):
        source,self.dataShape = createHDF5DataSource(a, True)
        layer = AlphaModulatedLayer(source)
        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

    @debugging_tools.log_call(logger)
    def addRGBAHDF5Layer(self, a, name=None):
        # TODO: Avoid this array indexing as it is a filename.
        assert(False)
        assert a.shape[2] >= 3
        sources = [None, None, None,None]
        for i in range(3):
            sources[i], self.dataShape = createHDF5DataSource(a[...,i], True)
        if(a.shape[3] >= 4):
            sources[3], self.dataShape = createHDF5DataSource(a[...,3], True)
        layer = RGBALayer(sources[0],sources[1],sources[2], sources[3])
        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

    @debugging_tools.log_call(logger)
    def addRandomColorsHDF5Layer(self, a, name=None, direct=False):
        layer = self.addColorTableLayer(a, name, colortable=None, direct=direct)
        layer.colortableIsRandom = True
        layer.zeroIsTransparent = True
        return layer

    @debugging_tools.log_call(logger)
    def addColorTableHDF5Source(self, source, shape, name=None, colortable=None, direct=False, clickFunctor=None):
        self.dataShape = shape

        if colortable is None:
            colortable = self._randomColors()

        layer = None
        if clickFunctor is None:
            layer = ColortableLayer(source, colortable, direct=direct)
        else:
            layer = ClickableColortableLayer(self.editor, clickFunctor, source, colortable, direct=direct)
        if name:
            layer.name = name

        layer.numberOfChannels = self.dataShape[-1]
        self.layerstack.append(layer)
        return layer

    @debugging_tools.log_call(logger)
    def addColorTableHDF5Layer(self, a, name=None, colortable=None, direct=False, clickFunctor=None):
        source, self.dataShape = createHDF5DataSource(a,True)

        if colortable is None:
            colortable = self._randomColors()

        layer = None
        if clickFunctor is None:
            layer = ColortableLayer(source, colortable, direct=direct)
        else:
            layer = ClickableColortableLayer(self.editor, clickFunctor, source, colortable, direct=direct)
        if name:
            layer.name = name

        layer.numberOfChannels = self.dataShape[-1]
        self.layerstack.append(layer)
        return layer




@multimethod(str, bool)
@debugging_tools.log_call(logger)
def createHDF5DataSource(full_path, withShape = False):
    # Get a source for the HDF5 file.
    src = HDF5DataSource(full_path)

    if withShape:
        return src, src.shape()
    else:
        return src

@multimethod(str)
@debugging_tools.log_call(logger)
def createHDF5DataSource(full_path):
    return createHDF5DataSource(full_path, False)


class HDF5NoFusedSourceException( Exception ):
    pass

class HDF5UndefinedShapeDtypeException( Exception ):
    pass

class HDF5DataFusedSource( QObject ):
    isDirty = pyqtSignal( object )
    numberOfChannelsChanged = pyqtSignal(int) # Never emitted

    @debugging_tools.log_call(logger)
    def __init__( self, fuse_axis, *data_sources, **kwargs):
        super(HDF5DataFusedSource, self).__init__()

        if len(data_sources) == 0:
            raise HDF5NoFusedSourceException("Have no data sources to fuse.")

        self.fuse_axis = fuse_axis

        self.data_sources = data_sources

        self.data_sources_defined = numpy.array(self.data_sources)
        self.data_sources_defined = self.data_sources_defined[self.data_sources_defined != numpy.array(None)]
        self.data_sources_defined = list(self.data_sources_defined)

        if len(self.data_sources_defined) == 0:
            if (("dtype" in kwargs) and ("shape" in kwargs)):
                self.data_dtype = kwargs["dtype"]
                self.data_shape = kwargs["shape"]
            else:
                raise HDF5UndefinedShapeDtypeException("Have no defined data sources to fuse and no shape or dtype to fallback on.")
        else:
            self.data_dtype = self.data_sources_defined[0].dtype()
            self.data_shape = -numpy.ones((5,), dtype = int)
            for each in self.data_sources_defined[1:]:
                try:
                    each_shape = each.shape()
                    each_shape = numpy.array(each_shape)

                    self.data_shape = numpy.array([self.data_shape, each_shape.astype(int)]).max(axis = 0)
                except AttributeError:
                    pass

                each_dtype = each.dtype()
                if not numpy.can_cast(each_dtype, self.data_dtype):
                    if numpy.can_cast(self.data_dtype, each_dtype):
                        self.data_dtype = each_dtype
                    else:
                        raise Exception("Cannot find safe conversion between self.data_dtype = " + repr(self.data_dtype) + " and each_dtype = " + repr(each_dtype) + ".")

        self.data_shape[self.fuse_axis] = len(self.data_sources)
        self.data_shape = tuple(self.data_shape)

        self.fuse_axis %= len(self.data_shape)
        if self.fuse_axis < 0:
            self.fuse_axis += len(self.data_shape)

    @debugging_tools.log_call(logger)
    def numberOfChannels(self):
        return self.dataset_shape[-1]

    @debugging_tools.log_call(logger)
    def clean_up(self):
        self.fuse_axis = None
        self.data_sources = None
        self.data_dtype = None
        self.data_shape = None

    @debugging_tools.log_call(logger)
    def dtype(self):
        return self.data_dtype

    @debugging_tools.log_call(logger)
    def shape(self):
        return self.data_shape

    @debugging_tools.log_call(logger)
    def request( self, slicing ):
        if not is_pure_slicing(slicing):
            raise Exception('HDF5DataFusedSource: slicing is not pure')


        slicing_formatted = []
        slicing_shape = []

        fuse_slicing = None
        non_fuse_slicing = []
        for i, (each_slicing, each_len) in enumerate(itertools.izip(slicing, self.data_shape)):
            each_slicing_formatted = None
            if i == self.fuse_axis:
                each_len = len(self.data_sources)
                fuse_slicing = each_slicing_formatted = additional_generators.reformat_slice(each_slicing, each_len)
                non_fuse_slicing.append(slice(0, 1, 1))
            else:
                each_slicing_formatted = additional_generators.reformat_slice(each_slicing, each_len)
                non_fuse_slicing.append(each_slicing_formatted)

            each_slicing_len = additional_generators.len_slice(each_slicing_formatted, each_len)

            slicing_formatted.append(each_slicing_formatted)
            slicing_shape.append(each_slicing_len)

        slicing_formatted = tuple(slicing_formatted)
        slicing_shape = tuple(slicing_shape)
        non_fuse_slicing = tuple(non_fuse_slicing)

        selected_data_sources = self.data_sources[fuse_slicing]

        selected_data_requests = []
        for each_data_source in selected_data_sources:
            each_data_request = None
            if each_data_source is not None:
                each_data_request = each_data_source.request(non_fuse_slicing)

            selected_data_requests.append(each_data_request)

        request = HDF5DataFusedRequest( self.fuse_axis, slicing_shape, self.data_dtype, *selected_data_requests )

        return(request)

    @debugging_tools.log_call(logger)
    def setDirty( self, slicing):
        if not is_pure_slicing(slicing):
            raise Exception('dirty region: slicing is not pure')
        self.isDirty.emit( slicing )

    @debugging_tools.log_call(logger)
    def __eq__( self, other ):
        if other is None:
            return False

        return(self.full_path == other.full_path)

    @debugging_tools.log_call(logger)
    def __ne__( self, other ):
        if other is None:
            return True

        return(self.full_path != other.full_path)

assert issubclass(HDF5DataFusedSource, SourceABC)


class HDF5DataFusedRequest( object ):

    @debugging_tools.log_call(logger)
    def __init__( self, fuse_axis, data_shape, data_dtype, *data_requests ):
        # TODO: Look at adding assertion check on slices.

        self.fuse_axis = fuse_axis
        self.data_shape = data_shape
        self.data_dtype = data_dtype
        self.data_requests = data_requests

        self._result = None

    @debugging_tools.log_call(logger)
    def wait( self ):
        if self._result is None:
            if True:
                self._result = numpy.zeros(self.data_shape, dtype = self.data_dtype)

                for i, each_data_request in enumerate(self.data_requests):
                    if each_data_request is not None:
                        each_result = each_data_request.wait()

                        result_view = expanded_numpy.index_axis_at_pos(self._result, self.fuse_axis, i)
                        each_result_view = expanded_numpy.index_axis_at_pos(each_result, self.fuse_axis, i)
                        result_view[:] = each_result_view

                logger.debug("Found the result.")

        return self._result

    @debugging_tools.log_call(logger)
    def getResult(self):
        return self._result

    @debugging_tools.log_call(logger)
    def cancel( self ):
        pass

    @debugging_tools.log_call(logger)
    def submit( self ):
        pass

    # callback( result = result, **kwargs )
    @debugging_tools.log_call(logger)
    def notify( self, callback, **kwargs ):
        t = threading.Thread(target=self._doNotify, args=( callback, kwargs ))
        t.start()

    @debugging_tools.log_call(logger)
    def _doNotify( self, callback, kwargs ):
        result = self.wait()
        callback(result, **kwargs)
assert issubclass(HDF5DataFusedRequest, RequestABC)


class SyncedChannelLayers(object):
    def __init__(self, *layers):
        self.layers = list(layers)
        self.currently_syncing_list = False

        for each_layer in self.layers:
            each_layer.channelChanged.connect(self)


    def __call__(self, channel):
        if not self.currently_syncing_list:
            self.currently_syncing_list = True

            for each_layer in self.layers:
                each_layer.channel = channel

            self.currently_syncing_list = False



@debugging_tools.log_call(logger)
def main(*argv):
    # TODO: Try to extract code for viewing each file with each viewer. This way multiple files generates multiple viewers.

    # Only necessary if running main (normally if calling command line). No point in importing otherwise.
    import read_config
    import argparse

    argv = list(argv)

    # Creates command line parser
    parser = argparse.ArgumentParser(description = "Parses input from the command line for a batch job.")

    # Takes a config file and then a series of one or more HDF5 files.
    parser.add_argument("config_filename", metavar = "CONFIG_FILE", type = str,
                        help = "JSON file that provides groups of items to be displayed together with the groups to keep in sync, layer names, and data locations.")
    parser.add_argument("input_files", metavar = "INPUT_FILE", type = str, nargs = '+',
                        help = "HDF5 file(s) to use for viewing. Must all have the same internal structure as specified by the JSON file.")

    # Results of parsing arguments (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Go ahead and stuff in parameters with the other parsed_args
    # A little risky if parsed_args may later contain a parameters variable due to changing the main file
    # or argparse changing behavior; however, this keeps all arguments in the same place.
    parsed_args.parameters = read_config.read_parameters(parsed_args.config_filename, maintain_order = True)

    # Open all of the files and store their handles
    parsed_args.file_handles = []
    for i in xrange(len(parsed_args.input_files)):
        parsed_args.input_files[i] = parsed_args.input_files[i].rstrip("/")
        parsed_args.input_files[i] = os.path.abspath(parsed_args.input_files[i])

        parsed_args.file_handles.append(h5py.File(parsed_args.input_files[i], "r"))

    # Make all each_layer_source_location_list is a lists whether they were or not before
    for i in xrange(len(parsed_args.parameters)):
        for (each_layer_name, each_layer_source_location_list) in parsed_args.parameters[i].items():
            if isinstance(each_layer_source_location_list, str):
                parsed_args.parameters[i][each_layer_name] = [ each_layer_source_location_list ]

    # Find all possible matches and whether they exist or not
    parsed_args.parameters_expanded = list()
    for each_layer_names_locations_group in parsed_args.parameters:

        # As we go through all files, we don't want to include the same layer more than once.
        # Also, we want to preserve the layer order in the configure file.
        parsed_args.parameters_expanded.append(collections.OrderedDict())
        for (each_layer_name, each_layer_source_location_list) in each_layer_names_locations_group.items():

            # Ensure the order of files is preserved (probably not an issue if were sorted) and each is unique.
            # This way we know they get fused in the right sequential order if necessary.
            parsed_args.parameters_expanded[-1][each_layer_name] = collections.OrderedDict()
            for each_layer_source_location in each_layer_source_location_list:

                # TODO: See if we can't move this loop out. (Could change this parsed_args.parameters_expanded to an collections.OrderedDict and store none for values (ordered set).)
                for each_file in parsed_args.file_handles:
                    new_matches = HDF5_searchers.get_matching_grouped_paths(each_file, each_layer_source_location)
                    new_matches_ldict = itertools.izip(new_matches, itertools.repeat(None))
                    parsed_args.parameters_expanded[-1][each_layer_name].update(new_matches_ldict)


    layer_names_locations_groups = parsed_args.parameters_expanded

    app = QApplication([""])#argv)
    viewer = HDF5Viewer()
    viewer.show()

    # Must reverse as Volumina puts the last items near the top.
    for each_file in reversed(parsed_args.file_handles):
        for each_layer_names_locations_group in reversed(layer_names_locations_groups):
            layer_sync_list = []

            for (each_layer_name, each_layer_source_dict_location_found) in reversed(each_layer_names_locations_group.items()):
                each_source = []

                # Ignore whether the file exists as that may differ for different files
                for each_layer_source_location in each_layer_source_dict_location_found.keys():
                    each_layer_source_location = each_layer_source_location.lstrip("/")

                    each_file_source = None

                    # Try to make the source. If it fails, we take no source.
                    try:
                        each_file_source = HDF5DataSource(each_file, each_layer_source_location)
                    except HDF5DatasetNotFoundException:
                        each_file_source = None

                    each_source.append(each_file_source)

                if len(each_source) > 1:
                    try:
                        each_source = HDF5DataFusedSource(-1, *each_source)
                    except HDF5UndefinedShapeDtypeException:
                        each_source = None
                elif len(each_source) == 1:
                    each_source = each_source[0]
                else:
                    each_source = None


                if each_source is not None:
                    each_layer = None
                    if issubclass(each_source.dtype(), numpy.integer):
                        each_layer = viewer.addColorTableHDF5Source(each_source, each_source.shape(), each_layer_name)
                    elif issubclass(each_source.dtype(), numpy.floating):
                        each_layer = viewer.addGrayscaleHDF5Source(each_source, each_source.shape(), each_layer_name)
                    elif issubclass(each_source.dtype(), numpy.bool_) or issubclass(each_source.dtype(), numpy.bool):
                        each_layer = viewer.addColorTableHDF5Source(each_source, each_source.shape(), each_layer_name)

                    each_layer.visible = False


                    layer_sync_list.append(each_layer)

            SyncedChannelLayers(*layer_sync_list)

    exit_code = app.exec_()

    # Close and clean up files
    parsed_args.file_handles = []
    for i in xrange(len(parsed_args.file_handles)):
        parsed_args.file_handles[i].close()
        parsed_args.file_handles[i] = None

    parsed_args.file_handles = None
    del parsed_args.file_handles

    return(exit_code)



if __name__ == "__main__":
    # only necessary if running main (normally if calling command line). no point in importing otherwise.
    import sys

    # call main if the script is loaded from command line. otherwise, user can import package without main being called.
    sys.exit(main(*sys.argv))

