__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 28, 2014 11:50:37 EDT$"


import os
import shutil
import tempfile

import numpy
import h5py

import nanshe.HDF5_serializers


class TestHDF5Searchers(object):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()

        self.temp_hdf5_file = h5py.File(os.path.join(self.temp_dir, "test.h5"), "w")


    def test_create_numpy_structured_array_in_HDF5_1(self):
        data1 = numpy.random.random((10, 10))
        data2 = numpy.random.random((10, 10))

        while (data1 == data2).all():
            data2 = numpy.random.random((10, 10))

        nanshe.HDF5_serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data1)

        assert("data" in self.temp_hdf5_file)
        assert((data1 == self.temp_hdf5_file["data"].value).all())

        try:
            nanshe.HDF5_serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2)
        except:
            assert(True)
        else:
            assert(False)

        assert("data" in self.temp_hdf5_file)
        assert((data1 == self.temp_hdf5_file["data"].value).all())

        try:
            nanshe.HDF5_serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2, overwrite=True)
        except:
            assert(False)
        else:
            assert(True)

        assert("data" in self.temp_hdf5_file)
        assert((data2 == self.temp_hdf5_file["data"].value).all())


    def test_create_numpy_structured_array_in_HDF5_2(self):
        data1 = numpy.zeros((10, 10), dtype=[("a", float, 2), ("b", int, 3)])
        data1["a"] = numpy.random.random((10, 10, 2))
        data1["b"] = numpy.random.random_integers(0, 10, (10, 10, 3))

        data2 = numpy.zeros((10, 10), dtype=[("a", float, 2), ("b", int, 3)])
        data2["a"] = numpy.random.random((10, 10, 2))
        data2["b"] = numpy.random.random_integers(0, 10, (10, 10, 3))
        while (data1 == data2).all():
            data2["a"] = numpy.random.random((10, 10, 2))
            data2["b"] = numpy.random.random_integers(0, 10, (10, 10, 3))

        nanshe.HDF5_serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data1)

        assert("data" in self.temp_hdf5_file)
        assert(data1.dtype == self.temp_hdf5_file["data"].dtype)
        assert(data1.shape == self.temp_hdf5_file["data"].shape)
        assert((data1 == self.temp_hdf5_file["data"].value).all())

        try:
            nanshe.HDF5_serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2)
        except:
            assert(True)
        else:
            assert(False)

        assert("data" in self.temp_hdf5_file)
        assert(data1.dtype == self.temp_hdf5_file["data"].dtype)
        assert(data1.shape == self.temp_hdf5_file["data"].shape)
        assert((data1 == self.temp_hdf5_file["data"].value).all())

        try:
            nanshe.HDF5_serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2, overwrite=True)
        except:
            assert(False)
        else:
            assert(True)

        assert("data" in self.temp_hdf5_file)
        assert(data2.dtype == self.temp_hdf5_file["data"].dtype)
        assert(data2.shape == self.temp_hdf5_file["data"].shape)
        assert((data2 == self.temp_hdf5_file["data"].value).all())


    def test_read_numpy_structured_array_from_HDF5(self):
        pass


    def teardown(self):
        self.temp_hdf5_file.close()

        self.temp_hdf5_file = None

        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""