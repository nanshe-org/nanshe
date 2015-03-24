__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 24, 2015 07:35:20 EDT$"



import os
import tempfile
import shutil

import h5py
import numpy

import nanshe.nanshe.HDF5_recorder


class TestHDF5Recorder(object):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()


    def test_EmptyArrayRecorder(self):
        recorder = nanshe.nanshe.HDF5_recorder.EmptyArrayRecorder()


        # Check if this stores results.

        assert not recorder


        # Check for missing key.

        assert recorder.get("key") is None

        assert recorder.get("key", True)

        assert "key" not in recorder

        got_key_error = False
        try:
            recorder["key"]
        except KeyError:
            got_key_error = True

        assert got_key_error


        # Add subgroup key and check for it.

        recorder["key"] = None

        assert recorder.get("key") is not None

        assert "key" in recorder

        got_key_error = False
        try:
            recorder["key"]
        except KeyError:
            got_key_error = True

        assert not got_key_error


        # Add data

        got_value_error = False
        try:
            recorder["value"] = numpy.array([])
        except ValueError:
            got_value_error = True
        assert got_value_error

        recorder["value"] = numpy.array(0)

        got_key_error = False
        try:
            recorder["value"]
        except KeyError:
            got_key_error = True
        assert got_key_error

        recorder["key"]["value"] = numpy.array(0)

        got_key_error = False
        try:
            recorder["key"]["value"]
        except KeyError:
            got_key_error = True
        assert got_key_error


    def test_HDF5ArrayRecorder(self):
        hdf5_filename = os.path.join(self.temp_dir, "test.h5")

        with h5py.File(hdf5_filename, "w") as hdf5_file:
            recorder = nanshe.nanshe.HDF5_recorder.HDF5ArrayRecorder(hdf5_file)


            # Check if this stores results.

            assert recorder


            # Check for missing key.

            assert recorder.get("key") is None

            assert recorder.get("key", True)

            assert "key" not in recorder

            got_key_error = False
            try:
                recorder["key"]
            except KeyError:
                got_key_error = True

            assert got_key_error


            # Add subgroup key and check for it.

            recorder["key"] = None

            assert recorder.get("key") is not None

            assert "key" in recorder

            got_key_error = False
            try:
                recorder["key"]
            except KeyError:
                got_key_error = True

            assert not got_key_error

            assert "key" in hdf5_file


            # Add data

            got_value_error = False
            try:
                recorder["value"] = numpy.array([])
            except ValueError:
                got_value_error = True
            assert got_value_error

            recorder["value"] = numpy.array(0)

            got_key_error = False
            try:
                recorder["value"]
            except KeyError:
                got_key_error = True
            assert not got_key_error

            assert "value" in hdf5_file

            recorder["key"]["value"] = numpy.array(0)

            got_key_error = False
            try:
                recorder["key"]["value"]
            except KeyError:
                got_key_error = True
            assert not got_key_error

            assert "key/value" in hdf5_file

            # Overwrite group test

            recorder.overwrite = True

            recorder["key"] = None

            got_key_error = False
            try:
                recorder["key"]["value"]
            except KeyError:
                got_key_error = True
            assert got_key_error

            assert "key/value" not in hdf5_file


    def teardown(self):
        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
