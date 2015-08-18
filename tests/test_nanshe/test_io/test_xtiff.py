__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 04, 2014 14:48:56 EDT$"


import collections
import itertools
import json
import os
import os.path
import shutil
import tempfile

import numpy
import h5py

import vigra
import vigra.impex

import nanshe.util.iters
import nanshe.util.xnumpy

import nanshe.io.xtiff
import nanshe.converter


class TestXTiff(object):
    def setup(self):
        self.temp_dir = ""
        self.filedata = collections.OrderedDict()
        self.offsets = None
        self.data = None

        self.data = numpy.random.random_integers(0, 255, (1000, 1, 102, 101, 1)).astype(numpy.uint8)

        self.offsets = list(xrange(0, self.data.shape[0] + 100 - 1, 100))

        self.temp_dir = tempfile.mkdtemp()
        for i, i_str, (a_b, a_e) in nanshe.util.iters.filled_stringify_enumerate(
                                        itertools.izip(
                                                *nanshe.util.iters.lagged_generators(
                                                    self.offsets
                                                )
                                        )
                                    ):
            each_filename = os.path.join(self.temp_dir, "test_tiff_" + str(i) + ".tif")
            each_data = self.data[a_b:a_e]

            self.filedata[each_filename] = each_data

            vigra.impex.writeVolume(nanshe.util.xnumpy.tagging_reorder_array(each_data, to_axis_order="czyxt")[0, 0],
                                    os.path.join(self.temp_dir, "test_tiff_" + str(i) + ".tif"), "")

        self.offsets = self.offsets[:-1]

    def test_get_multipage_tiff_shape_dtype(self):
        for each_filename, each_filedata in self.filedata.items():
            each_shape_dtype = nanshe.io.xtiff.get_multipage_tiff_shape_dtype(each_filename)

            each_filedata = nanshe.util.xnumpy.tagging_reorder_array(each_filedata, to_axis_order="zyxtc")[0]

            assert (each_shape_dtype["shape"] == each_filedata.shape)
            assert (each_shape_dtype["dtype"] == each_filedata.dtype.type)

    def test_get_multipage_tiff_shape_dtype_transformed(self):
        for each_filename, each_filedata in self.filedata.items():
            each_shape_dtype = nanshe.io.xtiff.get_multipage_tiff_shape_dtype_transformed(
                each_filename,
                axis_order="tzyxc"
            )

            assert (each_shape_dtype["shape"] == each_filedata.shape)
            assert (each_shape_dtype["dtype"] == each_filedata.dtype.type)

    def test_get_standard_tiff_array(self):
        for each_filename, each_filedata in self.filedata.items():
            each_data = nanshe.io.xtiff.get_standard_tiff_array(each_filename)

            assert (each_data.shape == each_filedata.shape)
            assert (each_data.dtype == each_filedata.dtype)

            assert (each_data == each_filedata).all()

    def test_convert_tiffs(self):
        hdf5_filename = os.path.join(self.temp_dir, "test.h5")
        hdf5_filepath = hdf5_filename + "/data"

        nanshe.io.xtiff.convert_tiffs(self.filedata.keys(), hdf5_filepath)

        assert os.path.exists(hdf5_filename)

        filenames = None
        offsets = None
        descriptions = None
        data = None
        with h5py.File(hdf5_filename, "r") as hdf5_handle:
            assert "filenames" in hdf5_handle["data"].attrs
            assert "offsets" in hdf5_handle["data"].attrs
            assert "descriptions" in hdf5_handle["data"].attrs

            filenames = hdf5_handle["data"].attrs["filenames"]
            offsets = hdf5_handle["data"].attrs["offsets"]
            descriptions = hdf5_handle["data"].attrs["descriptions"]

            data = hdf5_handle["data"].value

        self_data_h5 = nanshe.util.xnumpy.tagging_reorder_array(
            self.data, to_axis_order="cztyx"
        )[0, 0]
        self_filenames = numpy.array(self.filedata.keys())

        assert len(filenames) == len(self_filenames)
        assert (filenames == self_filenames).all()

        assert len(offsets) == len(self.offsets)
        assert numpy.equal(offsets, self.offsets).all()

        assert len(descriptions) == len(self.data)
        assert all(_ == u"" for _ in descriptions)

        assert (data == self_data_h5).all()

        os.remove(hdf5_filename)

    def teardown(self):
        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
        self.filedata = collections.OrderedDict()
        self.data = None
