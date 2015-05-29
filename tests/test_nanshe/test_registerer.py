__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 12, 2015 09:40:16 EST$"


import nose
import nose.plugins
import nose.plugins.attrib

import json
import os
import shutil
import tempfile

import h5py
import numpy

import nanshe.util.xnumpy
import nanshe.registerer


class TestRegisterer(object):
    def setup(self):
        self.temp_dirname = ""
        self.config_filename = ""
        self.data_filename = ""
        self.result_filename = ""


        self.temp_dirname = os.path.abspath(tempfile.mkdtemp())

        self.config_filename = os.path.join(self.temp_dirname, "config.json")
        with open(self.config_filename, "w") as config_file:
            pass

        self.data_filename = os.path.join(self.temp_dirname, "in.h5")
        with h5py.File(self.data_filename, "w") as data_file:
            pass

        self.result_filename = os.path.join(self.temp_dirname, "out.h5")


    def teardown(self):
        if self.temp_dirname:
            shutil.rmtree(self.temp_dirname)

        self.temp_dirname = ""
        self.config_filename = ""
        self.data_filename = ""
        self.result_filename = ""


    def test_main_0a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())
        b = nanshe.util.xnumpy.truncate_masked_frames(b)


        with open(self.config_filename, "a") as config_file:
            json.dump({}, config_file)

        with h5py.File(self.data_filename, "a") as data_file:
            data_file["images"] = a
            data_file["images"].attrs["attr"] = "test"

        self.data_filepath = self.data_filename + "/" + "images"
        self.result_filepath = self.result_filename + "/" + "images"

        nanshe.registerer.main(
            nanshe.registerer.__file__,
            self.config_filename,
            self.data_filepath,
            self.result_filepath
        )

        b2 = None
        with h5py.File(self.result_filename, "a") as result_file:
            assert "attr" in result_file["images"].attrs
            assert "test" == result_file["images"].attrs["attr"]

            b2 = result_file["images"][...]

        assert (b2 == b).all()


    def test_main_1a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked

        b = nanshe.util.xnumpy.truncate_masked_frames(b)


        with open(self.config_filename, "a") as config_file:
            json.dump({}, config_file)

        with h5py.File(self.data_filename, "a") as data_file:
            data_file["images"] = a
            data_file["images"].attrs["attr"] = "test"

        self.data_filepath = self.data_filename + "/" + "images"
        self.result_filepath = self.result_filename + "/" + "images"

        nanshe.registerer.main(
            nanshe.registerer.__file__,
            self.config_filename,
            self.data_filepath,
            self.result_filepath
        )

        b2 = None
        with h5py.File(self.result_filename, "a") as result_file:
            assert "attr" in result_file["images"].attrs
            assert "test" == result_file["images"].attrs["attr"]

            b2 = result_file["images"][...]

        assert (b2 == b).all()


    def test_main_2a(self):
        a = numpy.zeros((20,11,12), dtype=int)

        a[:, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked

        b = nanshe.util.xnumpy.truncate_masked_frames(b)


        with open(self.config_filename, "a") as config_file:
            json.dump({}, config_file)

        with h5py.File(self.data_filename, "a") as data_file:
            data_file["images"] = a
            data_file["images"].attrs["attr"] = "test"

        self.data_filepath = self.data_filename + "/" + "images"
        self.result_filepath = self.result_filename + "/" + "images"

        nanshe.registerer.main(
            nanshe.registerer.__file__,
            self.config_filename,
            self.data_filepath,
            self.result_filepath
        )

        b2 = None
        with h5py.File(self.result_filename, "a") as result_file:
            assert "attr" in result_file["images"].attrs
            assert "test" == result_file["images"].attrs["attr"]

            b2 = result_file["images"][...]

        assert (b2 == b).all()


    @nose.plugins.attrib.attr("3D")
    def test_main_0b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())
        b = nanshe.util.xnumpy.truncate_masked_frames(b)


        with open(self.config_filename, "a") as config_file:
            json.dump({}, config_file)

        with h5py.File(self.data_filename, "a") as data_file:
            data_file["images"] = a
            data_file["images"].attrs["attr"] = "test"

        self.data_filepath = self.data_filename + "/" + "images"
        self.result_filepath = self.result_filename + "/" + "images"

        nanshe.registerer.main(
            nanshe.registerer.__file__,
            self.config_filename,
            self.data_filepath,
            self.result_filepath
        )

        b2 = None
        with h5py.File(self.result_filename, "a") as result_file:
            assert "attr" in result_file["images"].attrs
            assert "test" == result_file["images"].attrs["attr"]

            b2 = result_file["images"][...]

        assert (b2 == b).all()


    @nose.plugins.attrib.attr("3D")
    def test_main_1b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6, :-6] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked

        b = nanshe.util.xnumpy.truncate_masked_frames(b)


        with open(self.config_filename, "a") as config_file:
            json.dump({}, config_file)

        with h5py.File(self.data_filename, "a") as data_file:
            data_file["images"] = a
            data_file["images"].attrs["attr"] = "test"

        self.data_filepath = self.data_filename + "/" + "images"
        self.result_filepath = self.result_filename + "/" + "images"

        nanshe.registerer.main(
            nanshe.registerer.__file__,
            self.config_filename,
            self.data_filepath,
            self.result_filepath
        )

        b2 = None
        with h5py.File(self.result_filename, "a") as result_file:
            assert "attr" in result_file["images"].attrs
            assert "test" == result_file["images"].attrs["attr"]

            b2 = result_file["images"][...]

        assert (b2 == b).all()


    @nose.plugins.attrib.attr("3D")
    def test_main_2b(self):
        a = numpy.zeros((20,11,12,13), dtype=int)

        a[:, 3:-4, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7, :-7] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked

        b = nanshe.util.xnumpy.truncate_masked_frames(b)


        with open(self.config_filename, "a") as config_file:
            json.dump({}, config_file)

        with h5py.File(self.data_filename, "a") as data_file:
            data_file["images"] = a
            data_file["images"].attrs["attr"] = "test"

        self.data_filepath = self.data_filename + "/" + "images"
        self.result_filepath = self.result_filename + "/" + "images"

        nanshe.registerer.main(
            nanshe.registerer.__file__,
            self.config_filename,
            self.data_filepath,
            self.result_filepath
        )

        b2 = None
        with h5py.File(self.result_filename, "a") as result_file:
            assert "attr" in result_file["images"].attrs
            assert "test" == result_file["images"].attrs["attr"]

            b2 = result_file["images"][...]

        assert (b2 == b).all()
