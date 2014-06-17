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





class HDF5DataSource( QObject ):
    isDirty = pyqtSignal( object )
    numberOfChannelsChanged = pyqtSignal(int) # Never emitted

    @advanced_debugging.log_call(logger)
    def __init__( self, full_path, shape = None, dtype = None):
        super(HDF5DataSource, self).__init__()

        self.full_path = full_path

        data_path_comps = pathHelpers.PathComponents(self.full_path)

        self.file_path = data_path_comps.externalPath
        self.dataset_path = data_path_comps.internalPath

        self.dataset_shape = shape
        self.dataset_dtype = dtype

        self.axis_order = [-1, -1, -1, -1, -1]

        with h5py.File(self.file_path, "r") as fid:
            if ( (self.dataset_shape is None) and (self.dataset_dtype is None) ):
                dataset = fid[self.dataset_path]
                self.dataset_shape = list(dataset.shape)
                self.dataset_dtype = dataset.dtype
            elif (self.dataset_shape is None):
                dataset = fid[self.dataset_path]
                self.dataset_shape = list(dataset.shape)
            elif (self.dataset_dtype is None):
                dataset = fid[self.dataset_path]
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

        return HDF5DataRequest(self.file_path, self.dataset_path, self.axis_order, self.dataset_dtype, slicing)

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
    def __init__( self, file_path, dataset_path, axis_order, dataset_dtype, slicing, throw_on_not_found = False ):
        # TODO: Look at adding assertion check on slices.

        self.file_path = file_path
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

                with h5py.File(self.file_path, "r") as fid:
                    try:
                        dataset = fid[self.dataset_path]
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


class HDF5DataFusedSource( QObject ):
    isDirty = pyqtSignal( object )
    numberOfChannelsChanged = pyqtSignal(int) # Never emitted

    @advanced_debugging.log_call(logger)
    def __init__( self, fuse_axis, *data_sources):
        super(HDF5DataFusedSource, self).__init__()

        if len(data_sources) == 0:
            raise Exception("Have no data sources to fuse.")

        self.fuse_axis = fuse_axis

        self.data_sources = data_sources

        self.data_dtype = data_sources[0].dtype()
        self.data_shape = -numpy.ones((5,), dtype = int)
        for each in self.data_sources[1:]:
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

        self.data_shape[self.fuse_axis] = len(data_sources)
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


@advanced_debugging.log_call(logger)
def main(*argv):
    argv = list(argv)

    app = QApplication(argv)
    viewer = HDF5Viewer()
    viewer.show()

    original_images = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/ADINA_results/images/original_data"

    viewer.addGrayscaleHDF5Layer(original_images, "original_data").visible = False

    images_max_projection = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/ADINA_results/images/debug/images_max_projection"

    viewer.addGrayscaleHDF5Layer(images_max_projection, "images_max_projection").visible = False

    original_images = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/ADINA_results/images/original_data"

    viewer.addGrayscaleHDF5Layer(original_images, "original_data").visible = False

    dictionary_images_max_projection = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/ADINA_results/images/debug/dictionary_images_max_projection"

    viewer.addGrayscaleHDF5Layer(dictionary_images_max_projection, "dictionary_images_max_projection").visible = False


    #neurons = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/ADINA_results/images/neurons"

    #viewer.addGrayscaleHDF5Layer(dictionary, "neurons")

    unmerged_neuron_set_contours = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/ADINA_results/images/debug/unmerged_neuron_set_contours"
    viewer.addColorTableHDF5Layer(unmerged_neuron_set_contours, "unmerged_neuron_set_contours").visible = False

    neurons_set_contours = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/ADINA_results/images/debug/new_neurons_set_contours"
    viewer.addColorTableHDF5Layer(neurons_set_contours, "neurons_set_contours").visible = False

    neuron_sets = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/ADINA_results/images/debug/neuron_sets"


    neuron_sets_path_comps = pathHelpers.PathComponents(neuron_sets)
    neuron_sets_file_path, neuron_sets_group_path = neuron_sets_path_comps.externalPath, neuron_sets_path_comps.internalPath

    # group_data = {}
    group_data_shape_dtype = dict()
    with h5py.File(neuron_sets_file_path, "r") as fid:
        logger.debug("Opened HDF5 file: \"" + neuron_sets_file_path + "\".")

        group = fid[neuron_sets_group_path]

        for each_group_id, (each_group_name, each_group) in enumerate(group.items()):
            for each_dataset_name, each_dataset in each_group.items():
                if isinstance(each_dataset, h5py.Dataset):
                    if each_dataset_name not in group_data_shape_dtype:
                        # group_data[each_dataset_name] = {each_group_id : each_dataset}
                        group_data_shape_dtype[each_dataset_name] = (each_dataset.shape, each_dataset.dtype)
                    else:
                        assert(group_data_shape_dtype[each_dataset_name][0] == each_dataset.shape, "Shape mismatch.")
                        assert(group_data_shape_dtype[each_dataset_name][1] == each_dataset.dtype, "Type mismatch.")

        logger.debug("Determined properties of all datasets in : \"" + neuron_sets + "\"." )


        group_data = dict()
        for each_dataset_name, (each_shape, each_dtype) in group_data_shape_dtype.items():
            if isinstance(each_dtype, numpy.dtype):
                if each_dtype.names:
                    continue

            if len(each_shape) < 2:
                continue

            if each_dataset_name.endswith("flattened_mask") or each_dataset_name.endswith("flattened"):
                continue

            group_data[each_dataset_name] = list()

            for each_group_id, each_group_name in enumerate(group.keys()):
                each_dataset_source = HDF5DataSource( neuron_sets + "/" + each_group_name + "/" + each_dataset_name + "/", shape = each_shape, dtype = each_dtype)

                group_data[each_dataset_name].append(each_dataset_source)

            group_data[each_dataset_name] = HDF5DataFusedSource(-1, *group_data[each_dataset_name])

            each_layer = None
            if issubclass(group_data[each_dataset_name].dtype().type, numpy.integer):
                each_layer = viewer.addColorTableHDF5Source(group_data[each_dataset_name], group_data[each_dataset_name].shape(), each_dataset_name)
            elif issubclass(group_data[each_dataset_name].dtype().type, numpy.floating):
                each_layer = viewer.addGrayscaleHDF5Source(group_data[each_dataset_name], group_data[each_dataset_name].shape(), each_dataset_name)
            elif issubclass(group_data[each_dataset_name].dtype().type, numpy.bool_) or issubclass(group_data[each_dataset_name].dtype().type, numpy.bool):
                each_layer = viewer.addColorTableHDF5Source(group_data[each_dataset_name], group_data[each_dataset_name].shape(), each_dataset_name)

            each_layer.visible = False

        logger.debug("Added all datasets as layers for : \"" + neuron_sets + "\"." )

    return(app.exec_())



if __name__ == "__main__":
    # only necessary if running main (normally if calling command line). no point in importing otherwise.
    import sys

    # call main if the script is loaded from command line. otherwise, user can import package without main being called.
    sys.exit(main(*sys.argv))

