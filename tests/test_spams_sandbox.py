__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 05, 2014 17:38:58 EDT$"


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

        self.space = numpy.array((100, 100))
        self.radii = numpy.array((5, 6, 7))

        self.g = synthetic_data.generate_hypersphere_masks(self.space, self.p, self.radii)

        self.g = self.g.reshape((self.g.shape[0], -1))
        self.g = self.g.transpose()
        self.g = numpy.asmatrix(self.g)
        self.g = numpy.asfortranarray(self.g)

    def test_run_multiprocessing_queue_spams_trainDL(self):
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

    def test_call_multiprocessing_queue_spams_trainDL(self):
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