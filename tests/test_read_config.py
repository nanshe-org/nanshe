__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 24, 2015 22:04:14 EDT$"



import json
import os
import shutil
import tempfile

import nanshe.nanshe.read_config


class TestReadConfig(object):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()


    def test0(self):
        dict_type = dict

        params = dict_type()
        params["b"] = range(3)
        params["c"] = "test"
        params["a"] = 5
        params["d"] = dict_type(params)
        params["h"] = [dict_type(params["d"])]
        params["g"] = [[_k, _v] for _k, _v in params["d"].items()]

        config_filename = os.path.join(self.temp_dir, "config.json")
        with open(config_filename, "w") as config_file:
            json.dump(params, config_file)

        with open(config_filename, "r") as config_file:
            params_raw_out = json.load(config_file)
            assert params == params_raw_out

        params_out = nanshe.nanshe.read_config.read_parameters(
            config_filename
        )

        assert params == params_out


    def teardown(self):
        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
