import multiprocessing

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 09, 2014 8:51:33AM$"


import volumina
import volumina.pixelpipeline


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


import advanced_debugging

logger = advanced_debugging.logging.getLogger(__name__)
advanced_debugging.logging.getLogger().setLevel(advanced_debugging.logging.WARN)

import volumina
from volumina.multimethods import multimethod
from volumina.viewer import Viewer
from volumina.pixelpipeline.datasources import *
from volumina.pixelpipeline.datasourcefactories import *
from volumina.layer import *
from volumina.layerstack import LayerStackModel
from volumina.navigationControler import NavigationInterpreter
from volumina import colortables



from PyQt4.QtCore import QTimer, pyqtSignal
from PyQt4.QtGui import QMainWindow, QApplication, QIcon, QAction, qApp
from PyQt4.uic import loadUi


#advanced_debugging.logging.getLogger().setLevel(advanced_debugging.logging.WARN)

import os
import random
import itertools


import pathHelpers
import h5py


import advanced_numpy
import advanced_iterators



class HDF5DatasetNotFoundException( Exception ):
    pass

class HDF5DataSource( QObject ):
    isDirty = pyqtSignal( object )
    numberOfChannelsChanged = pyqtSignal(int) # Never emitted

    @advanced_debugging.log_call(logger)
    def __init__( self, file_handle, internal_path, shape = None, dtype = None):
        super(HDF5DataSource, self).__init__()

        self.file_handle = None

        self.file_path = ""
        self.dataset_path = ""

        self.full_path = ""

        self.dataset_shape = shape
        self.dataset_dtype = dtype

        self.axis_order = [-1, -1, -1, -1, -1]


        if isinstance(file_handle, str):
            file_handle.rstrip("/")
            file_handle = h5py.File(file_handle)

        self.file_handle = file_handle

        self.file_path = self.file_handle.filename
        self.dataset_path = "/" + internal_path.strip("/")

        self.full_path = self.file_path + self.dataset_path


        if self.dataset_path not in self.file_handle:
            raise(HDF5DatasetNotFoundException("Could not find the path \"" + self.dataset_path + "\" in filename " + "\"" + self.file_path + "\"."))

        if ( (self.dataset_shape is None) and (self.dataset_dtype is None) ):
            dataset = self.file_handle[self.dataset_path]
            #print dataset.name
            self.dataset_shape = list(dataset.shape)
            self.dataset_dtype = dataset.dtype
        elif (self.dataset_shape is None):
            dataset = self.file_handle[self.dataset_path]
            self.dataset_shape = list(dataset.shape)
        elif (self.dataset_dtype is None):
            dataset = self.file_handle[self.dataset_path]
            self.dataset_dtype = dataset.dtype
        else:
            self.dataset_shape = list(self.dataset_shape)

        if len(self.dataset_shape) == 2:
            # Pretend that the shape is ( (1,) + self.dataset_shape + (1,1) )
            self.dataset_shape = [1, self.dataset_shape[0], self.dataset_shape[1], 1, 1]
            self.axis_order = [-1, 0, 1, -1, -1]
        elif len(self.dataset_shape) == 3 and self.dataset_shape[2] <= 4:
            # Pretend that the shape is ( (1,) + self.dataset_shape[0:2] + (1,) + (self.dataset_shape[2],) )
            self.dataset_shape = [1, self.dataset_shape[0], self.dataset_shape[1], 1, self.dataset_shape[2]]
            self.axis_order = [-1, 0, 1, -1, 2]
        elif len(self.dataset_shape) == 3:
            # Pretend that the shape is ( (1,) + self.dataset_shape + (1,) )
            #self.dataset_shape = [self.dataset_shape[0], self.dataset_shape[1], self.dataset_shape[2], 1, 1]
            self.dataset_shape = [1, self.dataset_shape[1], self.dataset_shape[2], 1, self.dataset_shape[0]]
            self.axis_order = [-1, 1, 2, -1, 0]
        elif len(self.dataset_shape) == 4:
            # Pretend that the shape is ( (1,) + self.dataset_shape )
            #self.dataset_shape = [self.dataset_shape[0], self.dataset_shape[1], self.dataset_shape[2], self.dataset_shape[3], 1]
            self.dataset_shape = [1, self.dataset_shape[1], self.dataset_shape[2], self.dataset_shape[3], self.dataset_shape[0]]
            self.axis_order = [-1, 1, 2, 3, 0]
        # elif len(self.dataset_shape) == 1:
        #     # Pretend that the shape is ( (1,) + self.dataset_shape + (1,1) )
        #     self.dataset_shape = [1, self.dataset_shape[0], 1, 1, 1]
        else:
            pass
            # assert(False, \
            # "slicing into an array of shape=%r requested, but slicing is %r" \
            # % (self.dataset_shape, slicing) )

        self.dataset_shape = tuple(self.dataset_shape)

    @advanced_debugging.log_call(logger)
    def numberOfChannels(self):
        return self.dataset_shape[-1]

    @advanced_debugging.log_call(logger)
    def clean_up(self):
        self.full_path = None
        self.file_path = None
        self.dataset_path = None
        self.dataset_dtype = None
        self.dataset_shape = None

    @advanced_debugging.log_call(logger)
    def dtype(self):
        return self.dataset_dtype

    @advanced_debugging.log_call(logger)
    def shape(self):
        return self.dataset_shape

    @advanced_debugging.log_call(logger)
    def request( self, slicing ):
        if not is_pure_slicing(slicing):
            raise Exception('HDF5DataSource: slicing is not pure')

        slicing = list(slicing)

        for i, (each_slicing, each_shape) in enumerate(itertools.izip(slicing, self.dataset_shape)):
            slicing[i] = advanced_iterators.reformat_slice(each_slicing, each_shape)

        slicing = tuple(slicing)

        assert(len(slicing) == len(self.dataset_shape), "Expect a slicing for a txyzc array.")

        return HDF5DataRequest(self.file_handle, self.dataset_path, self.axis_order, self.dataset_dtype, slicing)

    @advanced_debugging.log_call(logger)
    def setDirty( self, slicing):
        if not is_pure_slicing(slicing):
            raise Exception('dirty region: slicing is not pure')
        self.isDirty.emit( slicing )

    @advanced_debugging.log_call(logger)
    def __eq__( self, other ):
        if other is None:
            return False

        return(self.full_path == other.full_path)

    @advanced_debugging.log_call(logger)
    def __ne__( self, other ):
        if other is None:
            return True

        return(self.full_path != other.full_path)

assert issubclass(HDF5DataSource, SourceABC)


class HDF5DataRequest( object ):
    @advanced_debugging.log_call(logger)
    def __init__( self, file_handle, dataset_path, axis_order, dataset_dtype, slicing, throw_on_not_found = False ):
        # TODO: Look at adding assertion check on slices.

        self.file_handle = file_handle
        self.dataset_path = dataset_path
        self.axis_order = axis_order
        self.dataset_dtype = dataset_dtype
        self.throw_on_not_found = throw_on_not_found

        self._result = None

        self.slicing = list()
        actual_slicing_dict = dict()
        for i, (each_slice, each_axis) in enumerate(itertools.izip(slicing, self.axis_order)):
            self.slicing.append(each_slice)
            if each_axis != -1:
                actual_slicing_dict[each_axis] = each_slice

        self.slicing = tuple(self.slicing)

        self.actual_slicing = numpy.zeros((len(actual_slicing_dict),), dtype = slice)
        for each_axis in sorted(actual_slicing_dict.keys()):
            self.actual_slicing[each_axis] = actual_slicing_dict[each_axis]

        self.actual_slicing = tuple(self.actual_slicing)

    @advanced_debugging.log_call(logger)
    def wait( self ):
        if self._result is None:
            if True:
                slicing_shape = advanced_iterators.len_slices(self.slicing)
                self._result = numpy.zeros(slicing_shape, dtype = self.dataset_dtype)

                try:
                    #print self.file_handle
                    #print self.dataset_path
                    dataset = self.file_handle[self.dataset_path]
                    a_result = dataset[self.actual_slicing]
                    a_result = numpy.array(a_result)

                    the_axis_order = numpy.array(self.axis_order)
                    a_result = a_result.transpose(the_axis_order[the_axis_order != -1])

                    for i, each_axis_order in enumerate(self.axis_order):
                        if each_axis_order == -1:
                            a_result = advanced_numpy.add_singleton_axis_pos(a_result, i)

                    self._result[:] = a_result
                except KeyError:
                    if self.throw_on_not_found:
                       raise

                logger.debug("Found the result.")

        return self._result

    @advanced_debugging.log_call(logger)
    def getResult(self):
        return self._result

    @advanced_debugging.log_call(logger)
    def cancel( self ):
        pass

    @advanced_debugging.log_call(logger)
    def submit( self ):
        pass

    # callback( result = result, **kwargs )
    @advanced_debugging.log_call(logger)
    def notify( self, callback, **kwargs ):
        t = threading.Thread(target=self._doNotify, args=( callback, kwargs ))
        t.start()

    @advanced_debugging.log_call(logger)
    def _doNotify( self, callback, kwargs ):
        result = self.wait()
        callback(result, **kwargs)
assert issubclass(HDF5DataRequest, RequestABC)


class HDF5Viewer(Viewer):
    @advanced_debugging.log_call(logger)
    def __init__(self, parent=None):
        super(HDF5Viewer, self).__init__(parent)

    @advanced_debugging.log_call(logger)
    def addGrayscaleHDF5Source(self, source, shape, name=None, direct=False):
        self.dataShape = shape
        layer = GrayscaleLayer(source, direct=direct)
        layer.numberOfChannels = self.dataShape[-1]

        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

    @advanced_debugging.log_call(logger)
    def addGrayscaleHDF5Layer(self, a, name=None, direct=False):
        source, self.dataShape = createHDF5DataSource(a, True)
        layer = GrayscaleLayer(source, direct=direct)
        layer.numberOfChannels = self.dataShape[-1]

        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

    @advanced_debugging.log_call(logger)
    def addAlphaModulatedHDF5Layer(self, a, name=None):
        source,self.dataShape = createHDF5DataSource(a, True)
        layer = AlphaModulatedLayer(source)
        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

    @advanced_debugging.log_call(logger)
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

    @advanced_debugging.log_call(logger)
    def addRandomColorsHDF5Layer(self, a, name=None, direct=False):
        layer = self.addColorTableLayer(a, name, colortable=None, direct=direct)
        layer.colortableIsRandom = True
        layer.zeroIsTransparent = True
        return layer

    @advanced_debugging.log_call(logger)
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

    @advanced_debugging.log_call(logger)
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
@advanced_debugging.log_call(logger)
def createHDF5DataSource(full_path, withShape = False):
    # Get a source for the HDF5 file.
    src = HDF5DataSource(full_path)

    if withShape:
        return src, src.shape()
    else:
        return src

@multimethod(str)
@advanced_debugging.log_call(logger)
def createHDF5DataSource(full_path):
    return createHDF5DataSource(full_path, False)


class HDF5NoFusedSourceException( Exception ):
    pass

class HDF5UndefinedShapeDtypeException( Exception ):
    pass

class HDF5DataFusedSource( QObject ):
    isDirty = pyqtSignal( object )
    numberOfChannelsChanged = pyqtSignal(int) # Never emitted

    @advanced_debugging.log_call(logger)
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

    @advanced_debugging.log_call(logger)
    def numberOfChannels(self):
        return self.dataset_shape[-1]

    @advanced_debugging.log_call(logger)
    def clean_up(self):
        self.fuse_axis = None
        self.data_sources = None
        self.data_dtype = None
        self.data_shape = None

    @advanced_debugging.log_call(logger)
    def dtype(self):
        return self.data_dtype

    @advanced_debugging.log_call(logger)
    def shape(self):
        return self.data_shape

    @advanced_debugging.log_call(logger)
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
                fuse_slicing = each_slicing_formatted = advanced_iterators.reformat_slice(each_slicing, each_len)
                non_fuse_slicing.append(slice(0, 1, 1))
            else:
                each_slicing_formatted = advanced_iterators.reformat_slice(each_slicing, each_len)
                non_fuse_slicing.append(each_slicing_formatted)

            each_slicing_len = advanced_iterators.len_slice(each_slicing_formatted, each_len)

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

    @advanced_debugging.log_call(logger)
    def setDirty( self, slicing):
        if not is_pure_slicing(slicing):
            raise Exception('dirty region: slicing is not pure')
        self.isDirty.emit( slicing )

    @advanced_debugging.log_call(logger)
    def __eq__( self, other ):
        if other is None:
            return False

        return(self.full_path == other.full_path)

    @advanced_debugging.log_call(logger)
    def __ne__( self, other ):
        if other is None:
            return True

        return(self.full_path != other.full_path)

assert issubclass(HDF5DataFusedSource, SourceABC)


class HDF5DataFusedRequest( object ):

    @advanced_debugging.log_call(logger)
    def __init__( self, fuse_axis, data_shape, data_dtype, *data_requests ):
        # TODO: Look at adding assertion check on slices.

        self.fuse_axis = fuse_axis
        self.data_shape = data_shape
        self.data_dtype = data_dtype
        self.data_requests = data_requests

        self._result = None

    @advanced_debugging.log_call(logger)
    def wait( self ):
        if self._result is None:
            if True:
                self._result = numpy.zeros(self.data_shape, dtype = self.data_dtype)

                for i, each_data_request in enumerate(self.data_requests):
                    if each_data_request is not None:
                        each_result = each_data_request.wait()

                        result_view = advanced_numpy.index_axis_at_pos(self._result, self.fuse_axis, i)
                        each_result_view = advanced_numpy.index_axis_at_pos(each_result, self.fuse_axis, i)
                        result_view[:] = each_result_view

                logger.debug("Found the result.")

        return self._result

    @advanced_debugging.log_call(logger)
    def getResult(self):
        return self._result

    @advanced_debugging.log_call(logger)
    def cancel( self ):
        pass

    @advanced_debugging.log_call(logger)
    def submit( self ):
        pass

    # callback( result = result, **kwargs )
    @advanced_debugging.log_call(logger)
    def notify( self, callback, **kwargs ):
        t = threading.Thread(target=self._doNotify, args=( callback, kwargs ))
        t.start()

    @advanced_debugging.log_call(logger)
    def _doNotify( self, callback, kwargs ):
        result = self.wait()
        callback(result, **kwargs)
assert issubclass(HDF5DataFusedRequest, RequestABC)


class SyncedChannelLayers(object):
    def __init__(self, *layers):
        self.layers = list(layers)
        self.currently_syncing_list = False

        #logger.warning(repr([_.name for _ in self.layers]))

        for each_layer in self.layers:
            each_layer.channelChanged.connect(self)


    def __call__(self, channel):
        if not self.currently_syncing_list:
            self.currently_syncing_list = True

            for each_layer in self.layers:
                #logger.warning( each_layer.name )
                each_layer.channel = channel

            self.currently_syncing_list = False



@advanced_debugging.log_call(logger)
def main(*argv):
    # Only necessary if running main (normally if calling command line). No point in importing otherwise.
    import read_config
    import argparse
    import os

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

    parsed_args.file_handles = []
    for i in xrange(len(parsed_args.input_files)):
        parsed_args.input_files[i] = parsed_args.input_files[i].rstrip("/")
        parsed_args.input_files[i] = os.path.abspath(parsed_args.input_files[i])

        parsed_args.file_handles.append(h5py.File(parsed_args.input_files[i], "r"))

    app = QApplication([""])#argv)
    viewer = HDF5Viewer()
    viewer.show()


    layer_names_locations_groups = parsed_args.parameters

    for (each_input_filename, each_file) in reversed(zip(parsed_args.input_files, parsed_args.file_handles)):
        for each_layer_names_locations_group in reversed(layer_names_locations_groups):
            layer_sync_list = []

            for (each_layer_name, each_layer_source_location_list) in reversed(each_layer_names_locations_group.items()):
                each_source = None

                #print each_layer_name

                if isinstance(each_layer_source_location_list, str):
                    # Non-Fuse source
                    each_source_loc = each_layer_source_location_list
                    each_source_loc = each_source_loc.lstrip("/")

                    each_source = HDF5DataSource(each_file, each_source_loc)
                elif isinstance(each_layer_source_location_list, list):
                    # Fuse source
                    each_source = list()

                    for each_source_id, each_source_loc in enumerate(each_layer_source_location_list):
                        #print each_source_loc

                        each_source_loc = each_source_loc.lstrip("/")

                        each_file_source = None
                        try:
                            each_file_source = HDF5DataSource(each_file, each_source_loc)
                        except HDF5DatasetNotFoundException:
                            each_file_source = None

                        each_source.append(each_file_source)


                    try:
                        each_source = HDF5DataFusedSource(-1, *each_source)
                    except HDF5UndefinedShapeDtypeException:
                        each_source = None

                else:
                    raise Exception("Unknown value.")


                if each_source is not None:
                    each_layer = None
                    if issubclass(each_source.dtype().type, numpy.integer):
                        each_layer = viewer.addColorTableHDF5Source(each_source, each_source.shape(), each_layer_name)
                    elif issubclass(each_source.dtype().type, numpy.floating):
                        each_layer = viewer.addGrayscaleHDF5Source(each_source, each_source.shape(), each_layer_name)
                    elif issubclass(each_source.dtype().type, numpy.bool_) or issubclass(each_source.dtype().type, numpy.bool):
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

