__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 28, 2014 11:50:37 EDT$"


import itertools
import os
import shutil
import tempfile

import h5py

import nanshe.nanshe.HDF5_searchers


class TestHDF5Searchers(object):
    groups_0 = [u"000", u"005", u"010", u"015", u"020", u"025", u"030", u"035", u"040", u"045", u"050", u"055", u"060",
                u"065", u"070", u"075", u"080", u"085", u"090", u"095", u"100"]

    groups_1 = [u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9"]

    @staticmethod
    def setup_paths_generator():
        for _ in TestHDF5Searchers.groups_0:
            for __ in TestHDF5Searchers.groups_1:
                if not ((_ == TestHDF5Searchers.groups_0[0]) and (__ == TestHDF5Searchers.groups_1[0])):
                    yield( u"/" + u"/".join([u"test", _, u"group", __, u"data"]) )
                else:
                    yield( u"/" + u"/".join([u"test", _, u"group", __]) )

    @staticmethod
    def get_matching_paths_generator():
        for _ in TestHDF5Searchers.groups_0:
            for __ in TestHDF5Searchers.groups_1:
                if not ((_ == TestHDF5Searchers.groups_0[0]) and (__ == TestHDF5Searchers.groups_1[0])):
                    yield( u"/" + u"/".join([u"test", _, u"group", __, u"data"]) )

    @staticmethod
    def get_matching_paths_groups_generator():
        yield([u"test"])
        yield(TestHDF5Searchers.groups_0)
        yield([u"group"])
        yield(TestHDF5Searchers.groups_1)
        yield([u"data"])

    @staticmethod
    def match_path_groups_gen(group_matches):
        for _ in itertools.product(*group_matches):
            yield( u"/" + u"/".join(_) )

    @staticmethod
    def get_matching_grouped_paths_gen():
        return(TestHDF5Searchers.match_path_groups_gen(list(TestHDF5Searchers.get_matching_paths_groups_generator())))

    def setup(self):
        self.temp_dir = tempfile.mkdtemp()

        self.temp_hdf5_file = h5py.File(os.path.join(self.temp_dir, "test.h5"), "w")

        for _ in TestHDF5Searchers.setup_paths_generator():
            self.temp_hdf5_file.create_group(_)

    def test_get_matching_paths(self):
        all_matched = nanshe.nanshe.HDF5_searchers.get_matching_paths(self.temp_hdf5_file, u"/test/[0-9]{3}/group/[0-9]/data")

        assert(len(all_matched) == (len(TestHDF5Searchers.groups_0) * len(TestHDF5Searchers.groups_1) - 1))

        for _1, _2 in itertools.izip(TestHDF5Searchers.get_matching_paths_generator(), all_matched):
            assert(_1 == _2)

    def test_get_matching_paths_groups(self):
        all_matched = nanshe.nanshe.HDF5_searchers.get_matching_paths_groups(self.temp_hdf5_file, u"/test/[0-9]{3}/group/[0-9]/data")

        num_permutations = 1
        for each_group_match_list in all_matched:
            num_permutations *= len(each_group_match_list)

        assert(num_permutations == (len(TestHDF5Searchers.groups_0) * len(TestHDF5Searchers.groups_1)))

        for _1, _2 in itertools.izip(TestHDF5Searchers.get_matching_paths_groups_generator(), all_matched):
            assert(_1 == _2)

        for _1, _2 in itertools.izip(TestHDF5Searchers.get_matching_grouped_paths_gen(), TestHDF5Searchers.match_path_groups_gen(all_matched)):
            assert(_1 == _2)

    def test_get_matching_grouped_paths(self):
        all_matched = nanshe.nanshe.HDF5_searchers.get_matching_grouped_paths(self.temp_hdf5_file, u"/test/[0-9]{3}/group/[0-9]/data")

        assert(len(all_matched) == (len(TestHDF5Searchers.groups_0) * len(TestHDF5Searchers.groups_1)))

        for _1, _2 in itertools.izip(TestHDF5Searchers.get_matching_grouped_paths_gen(), all_matched):
            assert(_1 == _2)

    def test_get_matching_grouped_paths_found(self):
        all_matched = nanshe.nanshe.HDF5_searchers.get_matching_grouped_paths_found(self.temp_hdf5_file, u"/test/[0-9]{3}/group/[0-9]/data")

        assert(len(all_matched) == (len(TestHDF5Searchers.groups_0) * len(TestHDF5Searchers.groups_1)))

        for _1, _2, _3 in itertools.izip(TestHDF5Searchers.get_matching_grouped_paths_gen(), all_matched.iterkeys(), all_matched.itervalues()):
            assert(_1 == _2)
            assert((_2 in self.temp_hdf5_file) == _3)

    def teardown(self):
        self.temp_hdf5_file.close()

        self.temp_hdf5_file = None

        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""

