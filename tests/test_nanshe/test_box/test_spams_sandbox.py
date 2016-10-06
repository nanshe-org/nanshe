from __future__ import print_function

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 05, 2014 17:38:58 EDT$"


import imp

import nose
import nose.plugins
import nose.plugins.attrib

import ctypes
import multiprocessing

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import numpy
import npctypes

import nanshe.box.spams_sandbox

import nanshe.syn.data


has_spams = False
try:
    imp.find_module("spams")
    has_spams = True
except ImportError:
    pass


try:
    xrange
except NameError:
    xrange = range


class TestSpamsSandbox(object):
    def setup(self):
        self.p = numpy.array([[27, 51],
                              [66, 85],
                              [77, 45]])

        self.p3 = numpy.array([[27, 51, 37],
                               [66, 85, 25],
                               [77, 45, 73]])

        self.space = numpy.array((100, 100))
        self.space3 = numpy.array((100, 100, 100))
        self.radii = numpy.array((5, 6, 7))

        self.g = nanshe.syn.data.generate_hypersphere_masks(
            self.space, self.p, self.radii
        )

        self.g = self.g.reshape((self.g.shape[0], -1))
        self.g = self.g.transpose()
        self.g = numpy.asmatrix(self.g)
        self.g = numpy.asfortranarray(self.g)

        self.g3 = nanshe.syn.data.generate_hypersphere_masks(
            self.space3, self.p3, self.radii
        )

        self.g3 = self.g3.reshape((self.g3.shape[0], -1))
        self.g3 = self.g3.transpose()
        self.g3 = numpy.asmatrix(self.g3)
        self.g3 = numpy.asfortranarray(self.g3)

    def test_run_multiprocessing_queue_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        out_queue = Queue()

        nanshe.box.spams_sandbox.run_multiprocessing_queue_spams_trainDL(
            out_queue,
            self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = out_queue.get()

        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_run_multiprocessing_queue_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        out_queue = Queue()

        nanshe.box.spams_sandbox.run_multiprocessing_queue_spams_trainDL(
            out_queue,
            self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = out_queue.get()

        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_run_multiprocessing_queue_spams_trainDL_3(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        out_queue = Queue()

        nanshe.box.spams_sandbox.run_multiprocessing_queue_spams_trainDL(
            out_queue,
            self.g.astype(float),
            D=self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = out_queue.get()

        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (self.g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_run_multiprocessing_queue_spams_trainDL_4(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        out_queue = Queue()

        nanshe.box.spams_sandbox.run_multiprocessing_queue_spams_trainDL(
            out_queue,
            self.g3.astype(float),
            D=self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = out_queue.get()

        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

        assert (self.g3.astype(bool) == d3.astype(bool)).all()

    def test_call_multiprocessing_queue_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_multiprocessing_queue_spams_trainDL(
            self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_call_multiprocessing_queue_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_multiprocessing_queue_spams_trainDL(
            self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_call_multiprocessing_queue_spams_trainDL_3(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_multiprocessing_queue_spams_trainDL(
            self.g.astype(float),
            D=self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (self.g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_call_multiprocessing_queue_spams_trainDL_4(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_multiprocessing_queue_spams_trainDL(
            self.g3.astype(float),
            D=self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

        assert (self.g3.astype(bool) == d3.astype(bool)).all()

    def test_run_multiprocessing_array_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        float_type = numpy.float64

        g_array = npctypes.shared.ndarray(self.g.shape, float_type, "F")
        with npctypes.shared.as_ndarray(g_array) as g_array_numpy:
            g_array_numpy[...] = self.g
        del g_array_numpy

        result_array = npctypes.shared.ndarray((self.g.shape[0], self.g.shape[1]), float_type, "F")

        nanshe.box.spams_sandbox.run_multiprocessing_array_spams_trainDL(
            result_array,
            g_array,
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )

        d = None
        with npctypes.shared.as_ndarray(result_array) as d:
            d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_run_multiprocessing_array_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        float_type = numpy.float64

        g3_array = npctypes.shared.ndarray(self.g3.shape, float_type, "F")
        with npctypes.shared.as_ndarray(g3_array) as g3_array_numpy:
            g3_array_numpy[...] = self.g3
        del g3_array_numpy

        result_array = npctypes.shared.ndarray((self.g3.shape[0], self.g3.shape[1]), float_type, "F")

        nanshe.box.spams_sandbox.run_multiprocessing_array_spams_trainDL(
            result_array,
            g3_array,
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )

        d3 = None
        with npctypes.shared.as_ndarray(result_array) as d3:
            d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_call_multiprocessing_array_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_multiprocessing_array_spams_trainDL(
            self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_call_multiprocessing_array_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_multiprocessing_array_spams_trainDL(
            self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_run_multiprocessing_array_spams_trainDL_3(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        float_type = numpy.float64

        g_array = npctypes.shared.ndarray(self.g.shape, float_type, "F")
        with npctypes.shared.as_ndarray(g_array) as g_array_numpy:
            g_array_numpy[...] = self.g
        del g_array_numpy

        result_array = npctypes.shared.ndarray((self.g.shape[0], self.g.shape[1]), float_type, "F")

        nanshe.box.spams_sandbox.run_multiprocessing_array_spams_trainDL(
            result_array,
            g_array,
            False,
            g_array,
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )

        d = None
        with npctypes.shared.as_ndarray(result_array) as d:
            d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (self.g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_run_multiprocessing_array_spams_trainDL_4(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        float_type = numpy.float64

        g3_array = npctypes.shared.ndarray(self.g3.shape, float_type, "F")
        with npctypes.shared.as_ndarray(g3_array) as g3_array_numpy:
            g3_array_numpy[...] = self.g3
        del g3_array_numpy

        result_array = npctypes.shared.ndarray((self.g3.shape[0], self.g3.shape[1]), float_type, "F")

        nanshe.box.spams_sandbox.run_multiprocessing_array_spams_trainDL(
            result_array,
            g3_array,
            False,
            g3_array,
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )

        d3 = None
        with npctypes.shared.as_ndarray(result_array) as d3:
            d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

        assert (self.g3.astype(bool) == d3.astype(bool)).all()

    def test_call_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_spams_trainDL(
            self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_call_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_spams_trainDL(
            self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_call_spams_trainDL_3(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_spams_trainDL(
            self.g.astype(float),
            D=self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (self.g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_call_spams_trainDL_4(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_spams_trainDL(
            self.g3.astype(float),
            D=self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

        assert (self.g3.astype(bool) == d3.astype(bool)).all()

    def teardown(self):
        self.p = None
        self.space = None
        self.radii = None
        self.g = None

        self.p3 = None
        self.g3 = None
