__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 05, 2014 17:38:58 EDT$"


import ctypes
import multiprocessing

import numpy

import spams_sandbox
import spams_sandbox.spams_sandbox

import synthetic_data


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

        self.g = synthetic_data.generate_hypersphere_masks(self.space, self.p, self.radii)

        self.g = self.g.reshape((self.g.shape[0], -1))
        self.g = self.g.transpose()
        self.g = numpy.asmatrix(self.g)
        self.g = numpy.asfortranarray(self.g)

        self.g3 = synthetic_data.generate_hypersphere_masks(self.space3, self.p3, self.radii)

        self.g3 = self.g3.reshape((self.g3.shape[0], -1))
        self.g3 = self.g3.transpose()
        self.g3 = numpy.asmatrix(self.g3)
        self.g3 = numpy.asfortranarray(self.g3)

    def test_run_multiprocessing_queue_spams_trainDL_1(self):
        out_queue = multiprocessing.Queue()

        spams_sandbox.spams_sandbox.run_multiprocessing_queue_spams_trainDL(out_queue,
                                                                            self.g.astype(float),
                                                                            **{
                                                                                    "gamma2" : 0,
                                                                                    "gamma1" : 0,
                                                                                     "numThreads" : -1,
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

        assert(self.g.shape == d.shape)

        assert((self.g.astype(bool).max(axis = 0) == d.astype(bool).max(axis = 0)).all())

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

        print unmatched_g

        assert(len(unmatched_g) == 0)

    def test_run_multiprocessing_queue_spams_trainDL_2(self):
        out_queue = multiprocessing.Queue()

        spams_sandbox.spams_sandbox.run_multiprocessing_queue_spams_trainDL(out_queue,
                                                                            self.g3.astype(float),
                                                                            **{
                                                                                    "gamma2" : 0,
                                                                                    "gamma1" : 0,
                                                                                     "numThreads" : -1,
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

        assert(self.g3.shape == d3.shape)

        assert((self.g3.astype(bool).max(axis = 0) == d3.astype(bool).max(axis = 0)).all())

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

        print unmatched_g3

        assert(len(unmatched_g3) == 0)

    def test_call_multiprocessing_queue_spams_trainDL_1(self):
        d = spams_sandbox.spams_sandbox.call_multiprocessing_queue_spams_trainDL(self.g.astype(float),
                                                                                 **{
                                                                                        "gamma2" : 0,
                                                                                        "gamma1" : 0,
                                                                                         "numThreads" : -1,
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

        assert(self.g.shape == d.shape)

        assert((self.g.astype(bool).max(axis = 0) == d.astype(bool).max(axis = 0)).all())

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

        print unmatched_g

        assert(len(unmatched_g) == 0)

    def test_call_multiprocessing_queue_spams_trainDL_2(self):
        d3 = spams_sandbox.spams_sandbox.call_multiprocessing_queue_spams_trainDL(self.g3.astype(float),
                                                                                 **{
                                                                                        "gamma2" : 0,
                                                                                        "gamma1" : 0,
                                                                                         "numThreads" : -1,
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

        assert(self.g3.shape == d3.shape)

        assert((self.g3.astype(bool).max(axis = 0) == d3.astype(bool).max(axis = 0)).all())

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

        print unmatched_g3

        assert(len(unmatched_g3) == 0)

    def test_run_multiprocessing_array_spams_trainDL_1(self):
        output_array_size = self.g.shape[0] * self.g.shape[1]
        output_array = multiprocessing.Array(ctypes.c_double, output_array_size)

        spams_sandbox.spams_sandbox.run_multiprocessing_array_spams_trainDL(output_array,
                                                                            self.g.astype(float),
                                                                            **{
                                                                                    "gamma2" : 0,
                                                                                    "gamma1" : 0,
                                                                                     "numThreads" : -1,
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
        d = numpy.frombuffer(output_array.get_obj(), dtype = ctypes.c_double).reshape((-1, self.g.shape[1])).copy()
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert(self.g.shape == d.shape)

        assert((self.g.astype(bool).max(axis = 0) == d.astype(bool).max(axis = 0)).all())

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

        print unmatched_g

        assert(len(unmatched_g) == 0)

    def test_run_multiprocessing_array_spams_trainDL_2(self):
        output_array_size = self.g3.shape[0] * self.g3.shape[1]
        output_array = multiprocessing.Array(ctypes.c_double, output_array_size)

        spams_sandbox.spams_sandbox.run_multiprocessing_array_spams_trainDL(output_array,
                                                                            self.g3.astype(float),
                                                                            **{
                                                                                    "gamma2" : 0,
                                                                                    "gamma1" : 0,
                                                                                     "numThreads" : -1,
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
        d3 = numpy.frombuffer(output_array.get_obj(), dtype = ctypes.c_double).reshape((-1, self.g3.shape[1])).copy()
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert(self.g3.shape == d3.shape)

        assert((self.g3.astype(bool).max(axis = 0) == d3.astype(bool).max(axis = 0)).all())

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

        print unmatched_g3

        assert(len(unmatched_g3) == 0)

    def test_call_multiprocessing_array_spams_trainDL(self):
        d = spams_sandbox.spams_sandbox.call_multiprocessing_array_spams_trainDL(self.g.astype(float),
                                                                                 **{
                                                                                        "gamma2" : 0,
                                                                                        "gamma1" : 0,
                                                                                         "numThreads" : -1,
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

        assert(self.g.shape == d.shape)

        assert((self.g.astype(bool).max(axis = 0) == d.astype(bool).max(axis = 0)).all())

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

        print unmatched_g

        assert(len(unmatched_g) == 0)

    def test_call_spams_trainDL(self):
        d = spams_sandbox.spams_sandbox.call_spams_trainDL(self.g.astype(float),
                                                           **{
                                                                "gamma2" : 0,
                                                                "gamma1" : 0,
                                                                "numThreads" : -1,
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

        assert(self.g.shape == d.shape)

        assert((self.g.astype(bool).max(axis = 0) == d.astype(bool).max(axis = 0)).all())

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

        print unmatched_g

        assert(len(unmatched_g) == 0)

    def teardown(self):
        self.p = None
        self.space = None
        self.radii = None
        self.g = None

        self.p3 = None
        self.g3 = None
