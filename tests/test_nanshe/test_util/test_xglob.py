__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 28, 2014 11:50:37 EDT$"


import nanshe.util.xglob


class TestXGlob(object):
    num_files = 10

    def setup(self):
        import tempfile

        self.temp_dir = tempfile.mkdtemp()

        self.temp_files = []
        for i in xrange(TestXGlob.num_files):
            self.temp_files.append(tempfile.NamedTemporaryFile(suffix = ".tif", dir = self.temp_dir))

        self.temp_files.sort(cmp = lambda a, b: 2*(a.name > b.name) - 1)


    def test_expand_pathname_list(self):
        import itertools

        matched_filenames = nanshe.util.xglob.expand_pathname_list(self.temp_dir + "/*.tif")
        matched_filenames.sort(cmp = lambda a, b: 2*(a > b) - 1)

        assert (len(matched_filenames) == len(self.temp_files))

        for each_l, each_f in itertools.izip(matched_filenames, self.temp_files):
            assert (each_l == each_f.name)

    def teardown(self):
        import shutil

        for i in xrange(len(self.temp_files)):
            self.temp_files[i].close()

        self.temp_files = []

        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
