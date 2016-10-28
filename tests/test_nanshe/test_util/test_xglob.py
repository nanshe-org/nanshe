__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 28, 2014 11:50:37 EDT$"


import nanshe.util.xglob


from builtins import range as irange


class TestXGlob(object):
    num_files = 10

    def setup(self):
        import tempfile

        self.temp_dir = tempfile.mkdtemp()

        self.temp_files = []
        temp_files_dict = dict()
        for i in irange(TestXGlob.num_files):
            each_tempfile = tempfile.NamedTemporaryFile(
                suffix=".tif", dir=self.temp_dir
            )
            temp_files_dict[each_tempfile.name] = each_tempfile

        for each_filename in sorted(temp_files_dict.keys()):
            self.temp_files.append(temp_files_dict[each_filename])


    def test_expand_pathname_list(self):
        import nanshe.util.iters

        matched_filenames = nanshe.util.xglob.expand_pathname_list(self.temp_dir + "/*.tif")
        matched_filenames = sorted(matched_filenames)

        assert (len(matched_filenames) == len(self.temp_files))

        for each_l, each_f in nanshe.util.iters.izip(matched_filenames, self.temp_files):
            assert (each_l == each_f.name)

    def teardown(self):
        import shutil

        for i in irange(len(self.temp_files)):
            self.temp_files[i].close()

        self.temp_files = []

        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
