__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Feb 20, 2015 10:40:15 EST$"



import numpy

import nanshe.nanshe.registration


class TestRegisterMeanOffsets(object):
    def test1(self):
        a = numpy.zeros((20,10,11), dtype=int)

        a[:, 3:-3, 3:-3] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-6, :-6] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked


        b2 = nanshe.nanshe.registration.register_mean_offsets(a)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()

    def test2(self):
        a = numpy.zeros((20,11,12), dtype=int)

        a[:, 3:-4, 3:-4] = 1

        b = numpy.ma.masked_array(a.copy())

        a[10] = 0
        a[10, :-7, :-7] = 1

        b[10, :, :3] = numpy.ma.masked
        b[10, :3, :] = numpy.ma.masked


        b2 = nanshe.nanshe.registration.register_mean_offsets(a)

        assert (b2.dtype == b.dtype)
        assert (b2.data == b.data).all()
        assert (b2.mask == b.mask).all()
