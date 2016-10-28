__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 24, 2015 22:04:14 EDT$"


import collections
import json
import os
import shutil
import tempfile

import nanshe.io.xjson


from past.builtins import unicode

from builtins import range as irange


class TestXJson(object):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()


    def test0a(self):
        dict_type = dict

        params = collections.OrderedDict()
        params["b"] = list(irange(3))
        params["c"] = "test"
        params["a"] = 5
        params["d"] = collections.OrderedDict(params)
        params["h"] = [collections.OrderedDict(params["d"])]
        params["g"] = [[_k, _v] for _k, _v in params["d"].items()]

        params["d"] = dict_type(params["d"])
        params["h"][0] = dict_type(params["h"][0])
        params = dict_type(params)

        config_filename = os.path.join(self.temp_dir, "config.json")
        with open(config_filename, "w") as config_file:
            json.dump(params, config_file)

        with open(config_filename, "r") as config_file:
            params_raw_out = json.load(config_file)
            assert params == params_raw_out

        params_out = nanshe.io.xjson.read_parameters(
            config_filename
        )

        assert params == params_out


    def test0b(self):
        dict_type = dict

        params = collections.OrderedDict()
        params["b"] = list(irange(3))
        params["b"].append("__comment__ to drop")
        params["c"] = "test"
        params["a"] = 5
        params["d"] = collections.OrderedDict(params)
        params["h"] = [collections.OrderedDict(params["d"])]
        params["g"] = [[_k, _v] for _k, _v in params["d"].items()]
        params["e"] = "__comment__ will be removed"
        params["__comment__ e"] = "also will be removed"
        params["f"] = u"will not be unicode"

        params["d"] = dict_type(params["d"])
        params["h"][0] = dict_type(params["h"][0])
        params = dict_type(params)

        config_filename = os.path.join(self.temp_dir, "config.json")
        with open(config_filename, "w") as config_file:
            json.dump(params, config_file)

        with open(config_filename, "r") as config_file:
            params_raw_out = json.load(config_file)
            assert params == params_raw_out

        params_out = nanshe.io.xjson.read_parameters(
            config_filename
        )

        params["b"] = params["b"][:-1]
        params["d"]["b"] = params["d"]["b"][:-1]
        params["h"][0]["b"] = params["h"][0]["b"][:-1]
        params["g"][0][-1] = params["g"][0][-1][:-1]
        del params["e"]
        del params["__comment__ e"]

        if str != unicode:
            params["f"] = params["f"].encode("utf-8")

        assert params == params_out


    def test1a(self):
        dict_type = collections.OrderedDict

        params = collections.OrderedDict()
        params["b"] = list(irange(3))
        params["c"] = "test"
        params["a"] = 5
        params["d"] = collections.OrderedDict(params)
        params["h"] = [collections.OrderedDict(params["d"])]
        params["g"] = [[_k, _v] for _k, _v in params["d"].items()]

        params["d"] = dict_type(params["d"])
        params["h"][0] = dict_type(params["h"][0])
        params = dict_type(params)

        config_filename = os.path.join(self.temp_dir, "config.json")
        with open(config_filename, "w") as config_file:
            json.dump(params, config_file)

        with open(config_filename, "r") as config_file:
            params_raw_out = json.load(config_file)
            assert params == params_raw_out

        params_out = nanshe.io.xjson.read_parameters(
            config_filename, maintain_order=True
        )

        assert params == params_out


    def test1b(self):
        dict_type = collections.OrderedDict

        params = collections.OrderedDict()
        params["b"] = list(irange(3))
        params["b"].append("__comment__ to drop")
        params["c"] = "test"
        params["a"] = 5
        params["d"] = collections.OrderedDict(params)
        params["h"] = [collections.OrderedDict(params["d"])]
        params["g"] = [[_k, _v] for _k, _v in params["d"].items()]
        params["e"] = "__comment__ will be removed"
        params["__comment__ e"] = "also will be removed"
        params["f"] = u"will not be unicode"

        params["d"] = dict_type(params["d"])
        params["h"][0] = dict_type(params["h"][0])
        params = dict_type(params)

        config_filename = os.path.join(self.temp_dir, "config.json")
        with open(config_filename, "w") as config_file:
            json.dump(params, config_file)

        with open(config_filename, "r") as config_file:
            params_raw_out = json.load(config_file)
            assert params == params_raw_out

        params_out = nanshe.io.xjson.read_parameters(
            config_filename, maintain_order=True
        )

        params["b"] = params["b"][:-1]
        params["d"]["b"] = params["d"]["b"][:-1]
        params["h"][0]["b"] = params["h"][0]["b"][:-1]
        params["g"][0][1] = params["g"][0][1][:-1]
        del params["e"]
        del params["__comment__ e"]

        if str != unicode:
            params["f"] = params["f"].encode("utf-8")

        assert params == params_out


    def teardown(self):
        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
