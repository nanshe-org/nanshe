__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Feb 20, 2015 10:40:15 EST$"


import nose
import nose.plugins
import nose.plugins.attrib

import os
import shutil
import tempfile

import h5py
import numpy

import nanshe.io.hdf5.serializers
import nanshe.imp.registration


class TestRegisterMeanOffsets(object):
    def test0a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        b2 = nanshe.imp.registration.register_mean_offsets(a)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test1a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked


        b2 = nanshe.imp.registration.register_mean_offsets(a)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test2a(self):
        a = numpy.zeros((20,11,12), dtype=int)

        a[:, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked


        b2 = nanshe.imp.registration.register_mean_offsets(a)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test3a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 10
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test4a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 10
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test5a(self):
        a = numpy.zeros((20,11,12), dtype=int)

        a[:, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 10
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test6a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 7
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test7a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 7
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test8a(self):
        a = numpy.zeros((20,11,12), dtype=int)

        a[:, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 7
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test9a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 30
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test10a(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 30
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test11a(self):
        a = numpy.zeros((20,11,12), dtype=int)

        a[:, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 30
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test12a(self):
        cwd = os.getcwd()
        temp_dir = ""

        try:
            temp_dir = tempfile.mkdtemp()
            os.chdir(temp_dir)

            with h5py.File("in.h5", "w") as f:
                a = f.create_dataset(
                    "a", shape=(20, 11, 12), dtype=int, chunks=True
                )
                a[:, 3:-4, 3:-4] = 1

                b = numpy.ma.masked_array(a[...])

                a[10] = 0
                a[10, :-7, :-7] = 1


                b[10, :, :3] = numpy.ma.masked
                b[10, :3, :] = numpy.ma.masked

                fn = nanshe.imp.registration.register_mean_offsets(
                    a, block_frame_length=7
                )

            assert isinstance(fn, basestring)
            assert os.path.exists(fn)

            b2 = None
            with h5py.File(fn, "r") as f:
                b2g = f["reg_frames"]
                b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
                b2 = b2d[...]

            os.remove(fn)

            assert (b2.dtype == b.dtype)
            assert (b2.data == b.data).all()
            assert (b2.mask == b.mask).all()
        finally:
            os.chdir(cwd)
            if temp_dir:
                shutil.rmtree(temp_dir)

    @nose.plugins.attrib.attr("3D")
    def test0b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        b2 = nanshe.imp.registration.register_mean_offsets(a)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test1b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6, :-6] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked


        b2 = nanshe.imp.registration.register_mean_offsets(a)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test2b(self):
        a = numpy.zeros((20,11,12,13), dtype=int)

        a[:, 3:-4, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7, :-7] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked


        b2 = nanshe.imp.registration.register_mean_offsets(a)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test3b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 10
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test4b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6, :-6] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 10
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test5b(self):
        a = numpy.zeros((20,11,12,13), dtype=int)

        a[:, 3:-4, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7, :-7] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 10
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test6b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 7
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test7b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6, :-6] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 7
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test8b(self):
        a = numpy.zeros((20,11,12,13), dtype=int)

        a[:, 3:-4, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7, :-7] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 7
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test9b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 30
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test10b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6, :-6] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 30
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test11b(self):
        a = numpy.zeros((20,11,12,13), dtype=int)

        a[:, 3:-4, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7, :-7] = 1

        b[10, :3, :, :] = numpy.ma.masked
        b[10, :, :3, :] = numpy.ma.masked
        b[10, :, :, :3] = numpy.ma.masked

        fn = nanshe.imp.registration.register_mean_offsets(
            a, block_frame_length = 30
        )

        b2 = None
        with h5py.File(fn, "r") as f:
            b2g = f["reg_frames"]
            b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
            b2 = b2d[...]

        os.remove(fn)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    @nose.plugins.attrib.attr("3D")
    def test12b(self):
        cwd = os.getcwd()
        temp_dir = ""

        try:
            temp_dir = tempfile.mkdtemp()
            os.chdir(temp_dir)

            with h5py.File("in.h5", "w") as f:
                a = f.create_dataset(
                    "a", shape=(20, 11, 12, 13), dtype=int, chunks=True
                )
                a[:, 3:-4, 3:-4, 3:-4] = 1

                b = numpy.ma.masked_array(a[...])

                a[10] = 0
                a[10, :-7, :-7, :-7] = 1

                b[10, :3, :, :] = numpy.ma.masked
                b[10, :, :3, :] = numpy.ma.masked
                b[10, :, :, :3] = numpy.ma.masked

                fn = nanshe.imp.registration.register_mean_offsets(
                    a, block_frame_length=7
                )

            assert isinstance(fn, basestring)
            assert os.path.exists(fn)

            b2 = None
            with h5py.File(fn, "r") as f:
                b2g = f["reg_frames"]
                b2d = nanshe.io.hdf5.serializers.HDF5MaskedDataset(b2g)
                b2 = b2d[...]

            os.remove(fn)

            assert (b2.dtype == b.dtype)
            assert (b2.data == b.data).all()
            assert (b2.mask == b.mask).all()
        finally:
            os.chdir(cwd)
            if temp_dir:
                shutil.rmtree(temp_dir)


class TestFindOffsets(object):
    def test0a(self):
        a = numpy.zeros((20,10,11), dtype=int)
        a_off = numpy.zeros((len(a), a.ndim-1), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        am = a.mean(axis=0)

        af = numpy.fft.fftn(a, axes=range(1, a.ndim))
        amf = numpy.fft.fftn(am, axes=range(am.ndim))


        a_off2 = nanshe.imp.registration.find_offsets(af, amf)

        assert (a_off2.dtype == a_off.dtype)
        assert (a_off2 == a_off).all()

    def test1a(self):
        a = numpy.zeros((20,10,11), dtype=int)
        a_off = numpy.zeros((len(a), a.ndim-1), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        a[10] = 0
        a[10, :-6, :-6] = 1

        a_off[10] = a.shape[1:]
        a_off[10] -= 3
        numpy.negative(a_off, out=a_off)

        am = a.mean(axis=0)

        af = numpy.fft.fftn(a, axes=range(1, a.ndim))
        amf = numpy.fft.fftn(am, axes=range(am.ndim))


        a_off2 = nanshe.imp.registration.find_offsets(af, amf)

        assert (a_off2.dtype == a_off.dtype)
        assert (a_off2 == a_off).all()

    def test2a(self):
        a = numpy.zeros((20,11,12), dtype=int)
        a_off = numpy.zeros((len(a), a.ndim-1), dtype=int)

        a[:, 3:-4, 3:-4] = 1

        a[10] = 0
        a[10, :-7, :-7] = 1

        a_off[10] = a.shape[1:]
        a_off[10] -= 3
        numpy.negative(a_off, out=a_off)

        am = a.mean(axis=0)

        af = numpy.fft.fftn(a, axes=range(1, a.ndim))
        amf = numpy.fft.fftn(am, axes=range(am.ndim))


        a_off2 = nanshe.imp.registration.find_offsets(af, amf)

        assert (a_off2.dtype == a_off.dtype)
        assert (a_off2 == a_off).all()

    @nose.plugins.attrib.attr("3D")
    def test0b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)
        a_off = numpy.zeros((len(a), a.ndim-1), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        am = a.mean(axis=0)

        af = numpy.fft.fftn(a, axes=range(1, a.ndim))
        amf = numpy.fft.fftn(am, axes=range(am.ndim))


        a_off2 = nanshe.imp.registration.find_offsets(af, amf)

        assert (a_off2.dtype == a_off.dtype)
        assert (a_off2 == a_off).all()

    @nose.plugins.attrib.attr("3D")
    def test1b(self):
        a = numpy.zeros((20,10,11,12), dtype=int)
        a_off = numpy.zeros((len(a), a.ndim-1), dtype=int)

        a[:, 3:-3, 3:-3, 3:-3] = 1

        a[10] = 0
        a[10, :-6, :-6, :-6] = 1

        a_off[10] = a.shape[1:]
        a_off[10] -= 3
        numpy.negative(a_off, out=a_off)

        am = a.mean(axis=0)

        af = numpy.fft.fftn(a, axes=range(1, a.ndim))
        amf = numpy.fft.fftn(am, axes=range(am.ndim))


        a_off2 = nanshe.imp.registration.find_offsets(af, amf)

        assert (a_off2.dtype == a_off.dtype)
        assert (a_off2 == a_off).all()

    @nose.plugins.attrib.attr("3D")
    def test2b(self):
        a = numpy.zeros((20,11,12,13), dtype=int)
        a_off = numpy.zeros((len(a), a.ndim-1), dtype=int)

        a[:, 3:-4, 3:-4, 3:-4] = 1

        a[10] = 0
        a[10, :-7, :-7, :-7] = 1

        a_off[10] = a.shape[1:]
        a_off[10] -= 3
        numpy.negative(a_off, out=a_off)

        am = a.mean(axis=0)

        af = numpy.fft.fftn(a, axes=range(1, a.ndim))
        amf = numpy.fft.fftn(am, axes=range(am.ndim))


        a_off2 = nanshe.imp.registration.find_offsets(af, amf)

        assert (a_off2.dtype == a_off.dtype)
        assert (a_off2 == a_off).all()
