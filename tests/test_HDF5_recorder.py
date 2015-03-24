__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 24, 2015 07:35:20 EDT$"



import tempfile
import shutil

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


    def teardown(self):
        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
