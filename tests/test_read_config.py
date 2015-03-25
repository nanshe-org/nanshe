__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 24, 2015 22:04:14 EDT$"



import shutil
import tempfile

import nanshe.nanshe.read_config


class TestReadConfig(object):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()


    def teardown(self):
        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
