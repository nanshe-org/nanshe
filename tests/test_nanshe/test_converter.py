__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 30, 2015 08:25:33 EDT$"


import collections
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


class TestConverter(object):
    def setup(self):
        self.temp_dir = ""
        self.filedata = collections.OrderedDict()
        self.data = None

        self.data = numpy.random.random_integers(0, 255, (1000, 1, 102, 101, 1)).astype(numpy.uint8)

        self.temp_dir = tempfile.mkdtemp()
        for i, i_str, (a_b, a_e) in nanshe.util.iters.filled_stringify_enumerate(
                                        nanshe.util.iters.izip(
                                                *nanshe.util.iters.lagged_generators(
                                                    nanshe.util.iters.irange(
                                                        0,
                                                        self.data.shape[0] + 100 - 1,
                                                        100
                                                    )
                                                )
                                        )
                                    ):
            each_filename = os.path.join(self.temp_dir, "test_tiff_" + str(i) + ".tif")
            each_data = self.data[a_b:a_e]

            self.filedata[each_filename] = each_data

            vigra.impex.writeVolume(nanshe.util.xnumpy.tagging_reorder_array(each_data, to_axis_order="czyxt")[0, 0],
                                    os.path.join(self.temp_dir, "test_tiff_" + str(i) + ".tif"), "")


    def test_main(self):
        params = {
            "axis" : 0,
            "channel" : 0,
            "z_index" : 0,
            "pages_to_channel" : 1
        }

        config_filename = os.path.join(self.temp_dir, "config.json")

        hdf5_filename = os.path.join(self.temp_dir, "test.h5")
        hdf5_filepath = hdf5_filename + "/data"

        with open(config_filename, "w") as fid:
            json.dump(params, fid)
            fid.write("\n")

        main_args = ["./converter.py"] + ["tiff"] + [config_filename] + list(self.filedata.keys()) + [hdf5_filepath]

        assert (nanshe.converter.main(*main_args) == 0)

        assert os.path.exists(hdf5_filename)

        data = None
        with h5py.File(hdf5_filename, "r") as hdf5_handle:
            data = hdf5_handle["data"].value

        self_data_h5 = nanshe.util.xnumpy.tagging_reorder_array(self.data, to_axis_order="cztyx")[0, 0]

        assert (data == self_data_h5).all()

        os.remove(hdf5_filename)


    def teardown(self):
        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
        self.filedata = collections.OrderedDict()
        self.data = None
