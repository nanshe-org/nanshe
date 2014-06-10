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

import os
import random
import itertools


import pathHelpers
import h5py


import advanced_numpy


class HDF5DataSource( QObject ):
    isDirty = pyqtSignal( object )
    numberOfChannelsChanged = pyqtSignal(int) # Never emitted

    def __init__( self, full_path):
        super(HDF5DataSource, self).__init__()

        self.full_path = full_path

        data_path_comps = pathHelpers.PathComponents(self.full_path)

        self.file_path = data_path_comps.externalPath
        self.dataset_path = data_path_comps.internalPath

        self.dataset_shape = None
        self.dataset_dtype = None

        with h5py.File(self.file_path, "r") as fid:
            dataset = fid[self.dataset_path]
            self.dataset_dtype = dataset.dtype
            self.dataset_shape = list(dataset.shape)

            if len(self.dataset_shape) == 2:
                # Pretend that the shape is ( (1,) + self.dataset_shape + (1,1) )
                self.dataset_shape = [1, self.dataset_shape[0], self.dataset_shape[1], 1, 1]
            elif len(self.dataset_shape) == 3 and self.dataset_shape[2] <= 4:
                # Pretend that the shape is ( (1,) + self.dataset_shape[0:2] + (1,) + (self.dataset_shape[2],) )
                #self.dataset_shape = [1, self.dataset_shape[0], self.dataset_shape[1], 1, self.dataset_shape[2]]
                self.dataset_shape = [1, self.dataset_shape[0], self.dataset_shape[1], 1, self.dataset_shape[2]]
            elif len(self.dataset_shape) == 3:
                # Pretend that the shape is ( (1,) + self.dataset_shape + (1,) )
                #self.dataset_shape = [1, self.dataset_shape[0], self.dataset_shape[1], self.dataset_shape[2], 1]
                self.dataset_shape = [self.dataset_shape[0], 1, self.dataset_shape[1], self.dataset_shape[2], 1]
            elif len(self.dataset_shape) == 4:
                # Pretend that the shape is ( (1,) + self.dataset_shape )
                #self.dataset_shape = [1, self.dataset_shape[0], self.dataset_shape[1], self.dataset_shape[2], self.dataset_shape[3]]
                self.dataset_shape = [self.dataset_shape[0], self.dataset_shape[1], self.dataset_shape[2], 1, self.dataset_shape[3]]
            else:
                assert(False, \
                "slicing into an array of shape=%r requested, but slicing is %r" \
                % (self.dataset_shape, slicing) )

        self.dataset_shape = tuple(self.dataset_shape)

    def numberOfChannels(self):
        return self.dataset_shape[-1]

    def clean_up(self):
        self.full_path = None
        self.file_path = None
        self.dataset_path = None
        self.dataset_dtype = None
        self.dataset_shape = None

    def dtype(self):
        return self.dataset_dtype

    def shape(self):
        return self.dataset_shape

    def request( self, slicing ):
        if not is_pure_slicing(slicing):
            raise Exception('HDF5DataSource: slicing is not pure')

        relevant_slicing = slicing

        with h5py.File(self.file_path, "r") as fid:
            dataset = fid[self.dataset_path]

            assert(len(slicing) == len(self.dataset_shape), "Expect a slicing for a txyzc array.")

            return HDF5DataRequest(self.file_path, self.dataset_path, self.dataset_shape, relevant_slicing)

    def setDirty( self, slicing):
        if not is_pure_slicing(slicing):
            raise Exception('dirty region: slicing is not pure')
        self.isDirty.emit( slicing )

    def __eq__( self, other ):
        if other is None:
            return False

        return(self.full_path == other.full_path)

    def __ne__( self, other ):
        if other is None:
            return True

        return(self.full_path != other.full_path)

assert issubclass(HDF5DataSource, SourceABC)



class HDF5DataRequest( object ):
    def __init__( self, file_path, dataset_path, dataset_shape, slicing ):
        # TODO: Look at adding assertion check on slices.

        self.file_path = file_path
        self.dataset_path = dataset_path
        self.dataset_shape = dataset_shape

        self._result = None

        self.slicing = list()
        for i, (each_slice, each_shape) in enumerate(itertools.izip(slicing, self.dataset_shape)):
            if each_shape != 1:
                self.slicing.append(each_slice)

        self.slicing = tuple(self.slicing)

    def wait( self ):
        if self._result is None:
            with h5py.File(self.file_path, "r") as fid:
                dataset = fid[self.dataset_path]
                print dataset
                print self.slicing
                self._result = dataset[self.slicing]

                for i, each_dim in enumerate(self.dataset_shape):
                    if each_dim == 1:
                        self._result = advanced_numpy.add_singleton_axis_pos(self._result, i)

                print self.dataset_shape
                print self._result.shape

        return self._result

    def getResult(self):
        return self._result

    def cancel( self ):
        pass

    def submit( self ):
        pass

    # callback( result = result, **kwargs )
    def notify( self, callback, **kwargs ):
        t = threading.Thread(target=self._doNotify, args=( callback, kwargs ))
        t.start()

    def _doNotify( self, callback, kwargs ):
        result = self.wait()
        callback(result, **kwargs)
assert issubclass(HDF5DataRequest, RequestABC)


class HDF5Viewer(Viewer):
    def __init__(self, parent=None):
        super(HDF5Viewer, self).__init__(parent)

    def addGrayscaleHDF5Layer(self, a, name=None, direct=False):
        source, self.dataShape = createHDF5DataSource(a, True)
        layer = GrayscaleLayer(source, direct=direct)
        layer.numberOfChannels = self.dataShape[-1]

        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

    def addAlphaModulatedHDF5Layer(self, a, name=None):
        source,self.dataShape = createHDF5DataSource(a, True)
        layer = AlphaModulatedLayer(source)
        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer

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

    def addRandomColorsHDF5Layer(self, a, name=None, direct=False):
        layer = self.addColorTableLayer(a, name, colortable=None, direct=direct)
        layer.colortableIsRandom = True
        layer.zeroIsTransparent = True
        return layer

    def addColorTableHDF5Layer(self, a, name=None, colortable=None, direct=False, clickFunctor=None):
        if colortable is None:
            colortable = self._randomColors()
        source,self.dataShape = createHDF5DataSource(a,True)
        if clickFunctor is None:
            layer = ColortableLayer(source, colortable, direct=direct)
        else:
            layer = ClickableColortableLayer(self.editor, clickFunctor, source, colortable, direct=direct)
        if name:
            layer.name = name
        self.layerstack.append(layer)
        return layer




@multimethod(str, bool)
def createHDF5DataSource(full_path, withShape = False):
    #has to handle NumpyArray
    #check if the array is 5d, if not so embed it in a canonical way

    src = HDF5DataSource(full_path)

    if withShape:
        return src, src.shape()
    else:
        return src

@multimethod(str)
def createHDF5DataSource(full_path):
    return createHDF5DataSource(full_path, False)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    data1 = (numpy.random.random((50, 100, 3))) * 255
    #source1, source1_shape = volumina.pixelpipeline.datasourcefactories.createDataSource(data1, True)

    #viewer.addGrayscaleSource(source1, source1_shape)
    #viewer.addGrayscaleLayer(data1)

    data2 = "/Users/kirkhamj/Developer/PyCharmCE/nanshe/nanshe/data_test/data_invitro_susanne.h5/images"

    viewer.addGrayscaleHDF5Layer(data2)

#    class MyInterpreter(NavigationInterpreter):
#
#        def __init__(self, navigationcontroler):
#            NavigationInterpreter.__init__(self,navigationcontroler)
#
#        def onMouseMove_default( self, imageview, event ):
#            if imageview._ticker.isActive():
#                #the view is still scrolling
#                #do nothing until it comes to a complete stop
#                return
#
#            imageview.mousePos = mousePos = imageview.mapScene2Data(imageview.mapToScene(event.pos()))
#            imageview.oldX, imageview.oldY = imageview.x, imageview.y
#            x = imageview.x = mousePos.y()
#            y = imageview.y = mousePos.x()
#            self._navCtrl.positionCursor( x, y, self._navCtrl._views.index(imageview))
#
#    #like this
#    myInt = MyInterpreter
#    viewer.editor.navigationInterpreterType = myInt
#
#    #or like this
#    tmpInt = viewer.editor.navigationInterpreterType
#    tmpInt.onMouseMove_default = myInt.onMouseMove_default
#    viewer.editor.navigationInterpreterType = tmpInt

    app.exec_()
