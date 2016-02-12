from __future__ import print_function


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 30, 2014 19:35:11 EDT$"


import nose
import nose.plugins
import nose.plugins.attrib

import numpy
import scipy

import scipy.spatial
import scipy.spatial.distance

import scipy.stats

import nanshe.util.xnumpy

import nanshe.imp.segment

import nanshe.syn.data


class TestSegment(object):
    def test_remove_zeroed_lines_1(self):
        a = numpy.ones((1, 100, 101))
        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        r = numpy.array([[0, 0, 0], [a.shape[1]-2, 3, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.imp.segment.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape)

        assert (a == b).all()

    def test_remove_zeroed_lines_2(self):
        a = numpy.ones((1, 100, 101))
        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        r = numpy.array([[0, 0, 0], [1, 3, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.imp.segment.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape)

        assert (a == b).all()


    def test_remove_zeroed_lines_3(self):
        a = numpy.ones((1, 100, 101))
        p = 0.2
        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        nr = numpy.random.geometric(p)

        r = numpy.array([numpy.repeat(0, nr), numpy.random.random_integers(1, a.shape[1] - 2, nr)]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.imp.segment.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape)

        assert (a == b).all()

    def test_remove_zeroed_lines_4(self):
        a = numpy.ones((1, 100, 101))
        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        r = numpy.array([[0, 0, 0], [a.shape[1], 0, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.imp.segment.remove_zeroed_lines(ar, dilation_shape=dilation_shape, erosion_shape=erosion_shape)

        assert (a == b).all()

    def test_remove_zeroed_lines_5(self):
        a = numpy.ones((1, 100, 101))
        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        r = numpy.array([[0, 0, 0, 0], [a.shape[1], a.shape[1]-1, 0, 1]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.imp.segment.remove_zeroed_lines(ar, dilation_shape=dilation_shape, erosion_shape=erosion_shape)

        assert (a == b).all()

    def test_remove_zeroed_lines_6(self):
        a = numpy.repeat(numpy.arange(100)[None].T, 101, axis=1)[None].astype(float)
        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        r = numpy.array([[0, 0, 0], [1, 3, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.imp.segment.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape)

        assert numpy.allclose(a, b, rtol=0, atol=1e-13)

    def test_remove_zeroed_lines_7(self):
        a = numpy.repeat(numpy.arange(100)[None], 101, axis=0)[None].astype(float)
        a[0, :, 0] = 1
        nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(a, 0, 0), -1, 0)[:] = 1

        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        r = numpy.array([[0, 0, 0, 0], [0, 2, 3, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.imp.segment.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape)

        assert numpy.allclose(a, b, rtol=0, atol=1e-13)

    def test_remove_zeroed_lines_8(self):
        a = numpy.ones((1, 100, 101))
        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        r = numpy.array([[0, 0, 0], [a.shape[1]-2, 3, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = numpy.zeros_like(a)
        nanshe.imp.segment.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape, out=b)

        assert (a == b).all()

    def test_remove_zeroed_lines_9(self):
        a = numpy.ones((1, 100, 101))
        erosion_shape = [21, 1]
        dilation_shape = [1, 3]

        r = numpy.array([[0, 0, 0], [a.shape[1]-2, 3, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = ar
        nanshe.imp.segment.remove_zeroed_lines(b, erosion_shape=erosion_shape, dilation_shape=dilation_shape, out=b)

        assert (a == b).all()

    @nose.plugins.attrib.attr("3D")
    def test_remove_zeroed_lines_10(self):
        a = numpy.ones((1, 100, 101, 102))
        erosion_shape = [21, 1, 1]
        dilation_shape = [1, 3, 1]

        r = numpy.array([[0, 0, 0], [a.shape[1]-2, 3, 4], [0, 0, 0]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(nanshe.util.xnumpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1]), -1, each_r[-2])[:] = 0

        b = ar
        nanshe.imp.segment.remove_zeroed_lines(b, erosion_shape=erosion_shape, dilation_shape=dilation_shape, out=b)

        assert (a == b).all()

    def test_estimate_f0_1(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 20

        a = numpy.ones((100, 101, 102))

        b = nanshe.imp.segment.estimate_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size
        )

        assert (b == a).all()

    def test_estimate_f0_1b(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 20

        a = numpy.ones((100, 101, 102))

        b = a.copy()
        nanshe.imp.segment.estimate_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size,
            out=b
        )

        assert (b == a).all()

    def test_estimate_f0_1c(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 20

        a = numpy.ones((100, 101, 102))

        b = a.copy()
        nanshe.imp.segment.estimate_f0(
            b,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size,
            out=b
        )

        assert (b == a).all()

    def test_estimate_f0_2(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 49

        mean = 0.0
        stdev = 1.0

        a = numpy.random.normal(mean, stdev, (100, 101, 102))

        b = nanshe.imp.segment.estimate_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size
        )

        # Seems to be basically 2 orders of magnitude in reduction. However, it may be a little above exactly two.
        # Hence, multiplication by 99 instead of 100.
        assert ((99.0*b.std()) < a.std())

    @nose.plugins.attrib.attr("3D")
    def test_estimate_f0_3(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 20

        a = numpy.ones((100, 101, 102, 103))

        b = nanshe.imp.segment.estimate_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size
        )

        assert (b == a).all()

    @nose.plugins.attrib.attr("3D")
    def test_estimate_f0_4(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 49

        mean = 0.0
        stdev = 1.0

        a = numpy.random.normal(mean, stdev, (100, 101, 102, 103))

        b = nanshe.imp.segment.estimate_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size
        )

        # Seems to be basically 2 orders of magnitude in reduction. However, it may be a little above exactly two.
        # Hence, multiplication by 99 instead of 100.
        assert ((99.0*b.std()) < a.std())

    def test_extract_f0_1(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 20
        bias = 100

        a = numpy.ones((100, 101, 102))

        b = nanshe.imp.segment.extract_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size,
            bias=bias
        )

        assert (b == 0).all()

    def test_extract_f0_1b(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 20
        bias = 100

        a = numpy.ones((100, 101, 102))

        b = a.copy()
        nanshe.imp.segment.extract_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size,
            bias=bias,
            out=b
        )

        assert (b == 0).all()

    def test_extract_f0_1c(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 20
        bias = 100

        a = numpy.ones((100, 101, 102))

        b = a.copy()
        nanshe.imp.segment.extract_f0(
            b,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size,
            bias=bias,
            out=b
        )

        assert (b == 0).all()

    def test_extract_f0_2(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 49
        bias = 100

        mean = 0.0
        stdev = 1.0

        a = numpy.random.normal(mean, stdev, (100, 101, 102))
        b = nanshe.imp.segment.extract_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size,
            bias=bias
        )

        # Seems to be basically 2 orders of magnitude in reduction. However, it may be a little above exactly two.
        # Hence, multiplication by 99 instead of 100.
        assert ((99.0*b.std()) < a.std())

        # Turns out that a difference greater than 0.1 will be over 10 standard deviations away.
        assert (((a - 100.0*b) < 0.1).all())

    @nose.plugins.attrib.attr("3D")
    def test_extract_f0_3(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 20
        bias = 100

        a = numpy.ones((100, 101, 102, 103))

        b = nanshe.imp.segment.extract_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size,
            bias=bias
        )

        assert (b == 0).all()

    @nose.plugins.attrib.attr("3D")
    def test_extract_f0_4(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        spatial_smoothing_gaussian_filter_window_size = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        temporal_smoothing_gaussian_filter_window_size = 5.0
        half_window_size = 49
        bias = 100

        mean = 0.0
        stdev = 1.0

        a = numpy.random.normal(mean, stdev, (100, 101, 102, 103))

        b = nanshe.imp.segment.extract_f0(
            a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            spatial_smoothing_gaussian_filter_window_size=spatial_smoothing_gaussian_filter_window_size,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            temporal_smoothing_gaussian_filter_window_size=temporal_smoothing_gaussian_filter_window_size,
            half_window_size=half_window_size,
            bias=bias
        )

        # Seems to be basically 2 orders of magnitude in reduction. However, it may be a little above exactly two.
        # Hence, multiplication by 99 instead of 100.
        assert ((99.0*b.std()) < a.std())

        # Turns out that a difference greater than 0.1 will be over 10 standard deviations away.
        assert (((a - 100.0*b) < 0.1).all())

    def test_preprocess_data_1(self):
        ## Does NOT test accuracy.

        config = {
            "normalize_data" : {
                "renormalized_images" : {
                    "ord" : 2
                }
            },
            "extract_f0" : {
                "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                "spatial_smoothing_gaussian_filter_window_size" : 5.0,
                "which_quantile" : 0.5,
                "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                "temporal_smoothing_gaussian_filter_window_size" : 5.0,
                "half_window_size" : 20,
                "bias" : 100
            },
            "remove_zeroed_lines" : {
                "erosion_shape" : [
                    21,
                    1
                ],
                "dilation_shape" : [
                    1,
                    3
                ]
            },
            "wavelet.transform" : {
                "scale" : [
                    3,
                    4,
                    4
                ]
            }
        }

        space = numpy.array([100, 100, 100])
        radii = numpy.array([5, 6])
        magnitudes = numpy.array([15, 16])
        points = numpy.array([[20, 30, 24],
                              [70, 59, 65]])

        masks = nanshe.syn.data.generate_hypersphere_masks(space, points, radii)
        images = nanshe.syn.data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        image_stack = images.max(axis=0)

        nanshe.imp.segment.preprocess_data(image_stack, **config)

    def test_preprocess_data_2(self):
        ## Does NOT test accuracy.

        config = {
            "normalize_data" : {
                "renormalized_images" : {
                    "ord" : 2
                }
            },
            "remove_zeroed_lines" : {
                "erosion_shape" : [
                    21,
                    1
                ],
                "dilation_shape" : [
                    1,
                    3
                ]
            },
            "wavelet.transform" : {
                "scale" : [
                    3,
                    4,
                    4
                ]
            }
        }

        space = numpy.array([100, 100, 100])
        radii = numpy.array([5, 6])
        magnitudes = numpy.array([15, 16])
        points = numpy.array([[20, 30, 24],
                              [70, 59, 65]])

        masks = nanshe.syn.data.generate_hypersphere_masks(space, points, radii)
        images = nanshe.syn.data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        image_stack = images.max(axis=0)

        nanshe.imp.segment.preprocess_data(image_stack, **config)

    def test_preprocess_data_3(self):
        ## Does NOT test accuracy.

        config = {
            "normalize_data" : {
                "renormalized_images" : {
                    "ord" : 2
                }
            },
            "extract_f0" : {
                "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                "spatial_smoothing_gaussian_filter_window_size" : 5.0,
                "which_quantile" : 0.5,
                "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                "temporal_smoothing_gaussian_filter_window_size" : 5.0,
                "half_window_size" : 20,
                "bias" : 100
            },
            "wavelet.transform" : {
                "scale" : [
                    3,
                    4,
                    4
                ]
            }
        }

        space = numpy.array([100, 100, 100])
        radii = numpy.array([5, 6])
        magnitudes = numpy.array([15, 16])
        points = numpy.array([[20, 30, 24],
                              [70, 59, 65]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        image_stack = images.max(axis=0)

        nanshe.imp.segment.preprocess_data(image_stack, **config)

    def test_preprocess_data_4(self):
        ## Does NOT test accuracy.

        config = {
            "normalize_data" : {
                "renormalized_images" : {
                    "ord" : 2
                }
            },
            "extract_f0" : {
                "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                "spatial_smoothing_gaussian_filter_window_size" : 5.0,
                "which_quantile" : 0.5,
                "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                "temporal_smoothing_gaussian_filter_window_size" : 5.0,
                "half_window_size" : 20,
                "bias" : 100
            },
            "remove_zeroed_lines" : {
                "erosion_shape" : [
                    21,
                    1
                ],
                "dilation_shape" : [
                    1,
                    3
                ]
            }
        }

        space = numpy.array([100, 100, 100])
        radii = numpy.array([5, 6])
        magnitudes = numpy.array([15, 16])
        points = numpy.array([[20, 30, 24],
                              [70, 59, 65]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        image_stack = images.max(axis=0)

        nanshe.imp.segment.preprocess_data(image_stack, **config)

    @nose.plugins.attrib.attr("3D")
    def test_preprocess_data_5(self):
        ## Does NOT test accuracy.

        config = {
            "normalize_data" : {
                "renormalized_images" : {
                    "ord" : 2
                }
            },
            "extract_f0" : {
                "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                "spatial_smoothing_gaussian_filter_window_size" : 5.0,
                "which_quantile" : 0.5,
                "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                "temporal_smoothing_gaussian_filter_window_size" : 5.0,
                "half_window_size" : 20,
                "bias" : 100
            },
            "wavelet.transform" : {
                "scale" : [
                    3,
                    4,
                    4,
                    4
                ]
            }
        }

        space = numpy.array([100, 100, 100, 100])
        radii = numpy.array([5, 6])
        magnitudes = numpy.array([15, 16])
        points = numpy.array([[20, 30, 24, 85],
                              [70, 59, 65, 17]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        image_stack = images.max(axis=0)

        nanshe.imp.segment.preprocess_data(image_stack, **config)

    @nose.plugins.attrib.attr("3D")
    def test_preprocess_data_6(self):
        ## Does NOT test accuracy.

        config = {
            "normalize_data" : {
                "renormalized_images" : {
                    "ord" : 2
                }
            },
            "wavelet.transform" : {
                "scale" : [
                    3,
                    4,
                    4,
                    4
                ]
            }
        }

        space = numpy.array([100, 100, 100, 100])
        radii = numpy.array([5, 6])
        magnitudes = numpy.array([15, 16])
        points = numpy.array([[20, 30, 24, 85],
                              [70, 59, 65, 17]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        image_stack = images.max(axis=0)

        nanshe.imp.segment.preprocess_data(image_stack, **config)

    @nose.plugins.attrib.attr("3D")
    def test_preprocess_data_7(self):
        ## Does NOT test accuracy.

        config = {
            "normalize_data" : {
                "renormalized_images" : {
                    "ord" : 2
                }
            },
            "extract_f0" : {
                "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                "spatial_smoothing_gaussian_filter_window_size" : 5.0,
                "which_quantile" : 0.5,
                "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                "temporal_smoothing_gaussian_filter_window_size" : 5.0,
                "half_window_size" : 20,
                "bias" : 100
            }
        }

        space = numpy.array([100, 100, 100, 100])
        radii = numpy.array([5, 6])
        magnitudes = numpy.array([15, 16])
        points = numpy.array([[20, 30, 24, 85],
                              [70, 59, 65, 17]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        image_stack = images.max(axis=0)

        nanshe.imp.segment.preprocess_data(image_stack, **config)

    def test_generate_dictionary_00(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(numpy.float32),
            **{
                "spams.trainDL" : {
                    "gamma2" : 0,
                    "gamma1" : 0,
                    "numThreads" : 1,
                    "K" : len(g),
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
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    def test_generate_dictionary_01(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(float),
            **{
                "spams.trainDL" : {
                    "gamma2" : 0,
                    "gamma1" : 0,
                    "numThreads" : 1,
                    "K" : len(g),
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
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_generate_dictionary_02(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(numpy.float32),
            **{
                "spams.trainDL" : {
                    "gamma2" : 0,
                    "gamma1" : 0,
                    "numThreads" : 1,
                    "K" : len(g),
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
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_generate_dictionary_03(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(float),
            **{
                "spams.trainDL" : {
                    "gamma2" : 0,
                    "gamma1" : 0,
                    "numThreads" : 1,
                    "K" : len(g),
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
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    def test_generate_dictionary_04(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(numpy.float32),
            g.astype(numpy.float32),
            **{
                "spams.trainDL" : {
                    "gamma2" : 0,
                    "gamma1" : 0,
                    "numThreads" : 1,
                    "K" : len(g),
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
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (g.astype(bool) == d.astype(bool)).all()

    def test_generate_dictionary_05(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(float),
            g.astype(float),
            **{
                "spams.trainDL" : {
                    "gamma2" : 0,
                    "gamma1" : 0,
                    "numThreads" : 1,
                    "K" : len(g),
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
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_generate_dictionary_06(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(numpy.float32),
            g.astype(numpy.float32),
            **{
                "spams.trainDL" : {
                    "gamma2" : 0,
                    "gamma1" : 0,
                    "numThreads" : 1,
                    "K" : len(g),
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
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_generate_dictionary_07(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(float),
            g.astype(float),
            **{
                "spams.trainDL" : {
                    "gamma2" : 0,
                    "gamma1" : 0,
                    "numThreads" : 1,
                    "K" : len(g),
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
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (g.astype(bool) == d.astype(bool)).all()

    def test_generate_dictionary_08(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(float),
            **{
                "sklearn.decomposition.dict_learning_online" : {
                    "n_jobs" : 1,
                    "n_components" : len(g),
                    "n_iter" : 20,
                    "batch_size" : 256,
                    "alpha" : 0.2,
                }
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_generate_dictionary_09(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(float),
            **{
                "sklearn.decomposition.dict_learning_online" : {
                    "n_jobs" : 1,
                    "n_components" : len(g),
                    "n_iter" : 20,
                    "batch_size" : 256,
                    "alpha" : 0.2,
                }
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    def test_generate_dictionary_10(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(float),
            g.astype(float),
            **{
                "sklearn.decomposition.dict_learning_online" : {
                    "n_jobs" : 1,
                    "n_components" : len(g),
                    "n_iter" : 20,
                    "batch_size" : 256,
                    "alpha" : 0.2,
                }
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_generate_dictionary_11(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))

        g = nanshe.syn.data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.imp.segment.generate_dictionary(
            g.astype(float),
            g.astype(float),
            **{
                "sklearn.decomposition.dict_learning_online" : {
                    "n_jobs" : 1,
                    "n_components" : len(g),
                    "n_iter" : 20,
                    "batch_size" : 256,
                    "alpha" : 0.2,
                }
            }
        )
        d = (d != 0)

        assert (g.shape == d.shape)

        assert (g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (g.astype(bool) == d.astype(bool)).all()

    def test_generate_local_maxima_vigra_1(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = nanshe.imp.segment.generate_local_maxima_vigra(g.max(axis=0))

        assert (numpy.array(m.nonzero()) == p.T).all()

    @nose.plugins.attrib.attr("3D")
    def test_generate_local_maxima_vigra_2(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = nanshe.imp.segment.generate_local_maxima_vigra(g.max(axis=0))

        assert (numpy.array(m.nonzero()) == p.T).all()

    def test_generate_local_maxima_scikit_image_1(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = nanshe.imp.segment.generate_local_maxima_scikit_image(g.max(axis=0))

    @nose.plugins.attrib.attr("3D")
    def test_generate_local_maxima_scikit_image_2(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = nanshe.imp.segment.generate_local_maxima_scikit_image(g.max(axis=0))

        assert (numpy.array(m.nonzero()) == p.T).all()

    def test_generate_local_maxima_1(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = nanshe.imp.segment.generate_local_maxima(g.max(axis=0))

        assert (numpy.array(m.nonzero()) == p.T).all()

    @nose.plugins.attrib.attr("3D")
    def test_generate_local_maxima_2(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = nanshe.imp.segment.generate_local_maxima(g.max(axis=0))

        assert (numpy.array(m.nonzero()) == p.T).all()

    def test_extended_region_local_maxima_properties_1(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = (g > 0.00065)
        g *= m

        e = nanshe.imp.segment.extended_region_local_maxima_properties(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        assert (numpy.bincount(e["label"])[1:]  == 1).all()

        assert (len(e) == len(p))

        assert (e["local_max"] == p).all()

        assert (e["area"] == numpy.apply_over_axes(numpy.sum, m, axes=range(1, m.ndim)).squeeze().astype(float)).all()

        assert (e["centroid"] == e["local_max"]).all()

        assert (e["intensity"] == g.max(axis=0)[tuple(p.T)]).all()

    def test_extended_region_local_maxima_properties_2(self):
        p = numpy.array([[27, 51],
                         [32, 53],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        g = numpy.array([g[0] + g[1], g[2]])
        m = (g > 0.00065)
        g *= m

        e = nanshe.imp.segment.extended_region_local_maxima_properties(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        assert (numpy.bincount(e["label"])[1:] == numpy.array([2, 1])).all()

        assert (len(e) == len(p))

        assert (e["local_max"] == p).all()

        assert (e["area"][[0, 2]] == numpy.apply_over_axes(numpy.sum, m, axes=range(1, m.ndim)).squeeze().astype(float)).all()

        # Not exactly equal due to floating point round off error
        assert ((e["centroid"][0] - numpy.array(m[0].nonzero()).mean(axis=1)) < 1e-14).all()

        # Not exactly equal due to floating point round off error
        assert ((e["centroid"][1] - numpy.array(m[0].nonzero()).mean(axis=1)) < 1e-14).all()

        assert (e["centroid"][2] == e["local_max"][2]).all()

        assert (e["intensity"] == g.max(axis=0)[tuple(p.T)]).all()

    @nose.plugins.attrib.attr("3D")
    def test_extended_region_local_maxima_properties_3(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = (g > 0.00065)
        g *= m

        e = nanshe.imp.segment.extended_region_local_maxima_properties(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        assert (numpy.bincount(e["label"])[1:]  == 1).all()

        assert (len(e) == len(p))

        assert (e["local_max"] == p).all()

        assert (e["area"] == numpy.apply_over_axes(numpy.sum, m, axes=range(1, m.ndim)).squeeze().astype(float)).all()

        assert (e["centroid"] == e["local_max"]).all()

        assert (e["intensity"] == g.max(axis=0)[tuple(p.T)]).all()

    @nose.plugins.attrib.attr("3D")
    def test_extended_region_local_maxima_properties_4(self):
        p = numpy.array([[27, 51, 87],
                         [66, 85, 55],
                         [77, 45, 26]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        g = numpy.array([g[0] + g[1], g[2]])
        m = (g > 0.00065)
        g *= m

        e = nanshe.imp.segment.extended_region_local_maxima_properties(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        assert (numpy.bincount(e["label"])[1:] == numpy.array([2, 1])).all()

        assert (len(e) == len(p))

        assert (e["local_max"] == p).all()

        assert (e["area"][[0, 2]] == numpy.apply_over_axes(numpy.sum, m, axes=range(1, m.ndim)).squeeze().astype(float)).all()

        # Not exactly equal due to floating point round off error
        assert ((e["centroid"][0] - numpy.array(m[0].nonzero()).mean(axis=1)) < 1e-14).all()

        # Not exactly equal due to floating point round off error
        assert ((e["centroid"][1] - numpy.array(m[0].nonzero()).mean(axis=1)) < 1e-14).all()

        assert (e["centroid"][2] == e["local_max"][2]).all()

        assert (e["intensity"] == g.max(axis=0)[tuple(p.T)]).all()

    def test_remove_low_intensity_local_maxima_1(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[23, 36],
                              [58, 64]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        e2 = nanshe.imp.segment.remove_low_intensity_local_maxima(e, 1.0)

        assert (len(points) == len(e.props))

        assert (0 == len(e2.props))

    def test_remove_low_intensity_local_maxima_2(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[23, 36],
                              [58, 64]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis=0)[masks[i].nonzero()] < images.max(axis=0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.imp.segment.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[0])

        assert (len(points) == len(e.props))

        assert (len(e.props) == len(e2.props))

    def test_remove_low_intensity_local_maxima_3(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[23, 36],
                              [58, 64]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis=0)[masks[i].nonzero()] < images.max(axis=0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.imp.segment.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[1])

        assert (len(points) == len(e.props))

        assert ((len(e.props) - 1) == len(e2.props))

    def test_remove_low_intensity_local_maxima_4(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[23, 36],
                              [58, 64]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis=0)[masks[i].nonzero()] < images.max(axis=0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.imp.segment.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[1] + \
                                                                  numpy.finfo(float).eps)

        assert (len(points) == len(e.props))

        assert ((len(e.props) - 2) == len(e2.props))

    @nose.plugins.attrib.attr("3D")
    def test_remove_low_intensity_local_maxima_5(self):
        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[23, 36, 21],
                              [58, 64, 62]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        e2 = nanshe.imp.segment.remove_low_intensity_local_maxima(e, 1.0)

        assert (len(points) == len(e.props))

        assert (0 == len(e2.props))

    @nose.plugins.attrib.attr("3D")
    def test_remove_low_intensity_local_maxima_6(self):
        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[23, 36, 21],
                              [58, 64, 62]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis=0)[masks[i].nonzero()] < images.max(axis=0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.imp.segment.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[0])

        assert (len(points) == len(e.props))

        assert (len(e.props) == len(e2.props))

    @nose.plugins.attrib.attr("3D")
    def test_remove_low_intensity_local_maxima_7(self):
        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[23, 36, 21],
                              [58, 64, 62]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis=0)[masks[i].nonzero()] < images.max(axis=0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.imp.segment.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[1])

        assert (len(points) == len(e.props))

        assert ((len(e.props) - 1) == len(e2.props))

    @nose.plugins.attrib.attr("3D")
    def test_remove_low_intensity_local_maxima_8(self):
        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[23, 36, 21],
                              [58, 64, 62]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis=0)[masks[i].nonzero()] < images.max(axis=0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.imp.segment.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[1] + \
                                                                  numpy.finfo(float).eps)

        assert (len(points) == len(e.props))

        assert ((len(e.props) - 2) == len(e2.props))

    def test_remove_too_close_local_maxima_1(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[63, 69],
                              [58, 64]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = masks.max(axis=0).astype(int)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.imp.segment.remove_too_close_local_maxima(e, dist)

        assert (len(points) == len(e.props))

        assert (1 == len(e2.props))

    def test_remove_too_close_local_maxima_2(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[63, 69],
                              [58, 64]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.imp.segment.remove_too_close_local_maxima(e, dist)

        assert (len(points) == len(e.props))

        assert (len(points) == len(e2.props))

    def test_remove_too_close_local_maxima_3(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1.01), dtype=float)
        points = numpy.array([[63, 69],
                              [58, 64]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = masks.max(axis=0).astype(int)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.imp.segment.remove_too_close_local_maxima(e, dist)

        assert (len(points) == len(e.props))

        assert (1 == len(e2.props))

        assert (points[magnitudes == magnitudes.max()] == e2.props["local_max"][0]).all()

    def test_remove_too_close_local_maxima_4(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1.01, 1), dtype=float)
        points = numpy.array([[63, 69],
                              [58, 64]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = masks.max(axis=0).astype(int)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.imp.segment.remove_too_close_local_maxima(e, dist)

        assert (len(points) == len(e.props))

        assert (1 == len(e2.props))

        assert (points[magnitudes == magnitudes.max()] == e2.props["local_max"][0]).all()

    @nose.plugins.attrib.attr("3D")
    def test_remove_too_close_local_maxima_5(self):
        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[63, 69, 26],
                              [58, 64, 21]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = masks.max(axis=0).astype(int)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.imp.segment.remove_too_close_local_maxima(e, dist)

        assert (len(points) == len(e.props))

        assert (1 == len(e2.props))

    @nose.plugins.attrib.attr("3D")
    def test_remove_too_close_local_maxima_6(self):
        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1), dtype=float)
        points = numpy.array([[63, 69, 26],
                              [58, 64, 21]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = nanshe.util.xnumpy.enumerate_masks(masks).max(axis=0)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.imp.segment.remove_too_close_local_maxima(e, dist)

        assert (len(points) == len(e.props))

        assert (len(points) == len(e2.props))

    @nose.plugins.attrib.attr("3D")
    def test_remove_too_close_local_maxima_7(self):
        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1.01), dtype=float)
        points = numpy.array([[63, 69, 26],
                              [58, 64, 21]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = masks.max(axis=0).astype(int)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.imp.segment.remove_too_close_local_maxima(e, dist)

        assert (len(points) == len(e.props))

        assert (1 == len(e2.props))

        assert (points[magnitudes == magnitudes.max()] == e2.props["local_max"][0]).all()

    @nose.plugins.attrib.attr("3D")
    def test_remove_too_close_local_maxima_8(self):
        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1.01, 1), dtype=float)
        points = numpy.array([[63, 69, 26],
                              [58, 64, 21]])

        masks = nanshe.syn.data.generate_hypersphere_masks(
            space, points, radii
        )
        images = nanshe.syn.data.generate_gaussian_images(
            space, points, radii/3.0, magnitudes
        ) * masks
        labels = masks.max(axis=0).astype(int)

        e = nanshe.imp.segment.ExtendedRegionProps(images.max(axis=0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.imp.segment.remove_too_close_local_maxima(e, dist)

        assert (len(points) == len(e.props))

        assert (1 == len(e2.props))

        assert (points[magnitudes == magnitudes.max()] == e2.props["local_max"][0]).all()

    def test_wavelet_denoising_1(self):
        params = {
            "remove_low_intensity_local_maxima" : {
                "percentage_pixels_below_max" : 0
            },
            "wavelet.transform" : {
                "scale" : 5
            },
            "accepted_region_shape_constraints" : {
                "major_axis_length" : {
                    "max" : 25.0,
                    "min" : 0.0
                }
            },
            "accepted_neuron_shape_constraints" : {
                "eccentricity" : {
                    "max" : 0.9,
                    "min" : 0.0
                },
                "area" : {
                    "max" : 600,
                    "min" : 30
                }
            },
            "estimate_noise" : {
                "significance_threshold" : 3.0
            },
            "significant_mask" : {
                "noise_threshold" : 3.0
            },
            "remove_too_close_local_maxima" : {
                "min_local_max_distance" : 100.0
            },
            "use_watershed" : True
        }

        shape = numpy.array((500, 500))

        neuron_centers = numpy.array([[177,  52], [127, 202], [343, 271]])
        original_neuron_image = nanshe.syn.data.generate_gaussian_images(shape, neuron_centers, (50.0/3.0,)*len(neuron_centers), (1.0/3.0,)*len(neuron_centers)).sum(axis=0)
        original_neurons_mask = (original_neuron_image >= 0.00014218114898827068)

        neurons = nanshe.imp.segment.wavelet_denoising(original_neuron_image, **params)

        assert (len(neuron_centers) == len(neurons))
        assert (original_neurons_mask == neurons["mask"].max(axis=0)).all()
        assert ((original_neurons_mask*original_neuron_image) == neurons["image"].max(axis=0)).all()

    def test_wavelet_denoising_2(self):
        params = {
            "remove_low_intensity_local_maxima" : {
                "percentage_pixels_below_max" : 0
            },
            "wavelet.transform" : {
                "scale" : 5
            },
            "accepted_region_shape_constraints" : {
                "major_axis_length" : {
                    "max" : 150.0,
                    "min" : 0.0
                }
            },
            "accepted_neuron_shape_constraints" : {
                "eccentricity" : {
                    "max" : 0.9,
                    "min" : 0.0
                },
                "area" : {
                    "max" : 10000,
                    "min" : 0
                }
            },
            "estimate_noise" : {
                "significance_threshold" : 3.0
            },
            "significant_mask" : {
                "noise_threshold" : 3.0
            },
            "remove_too_close_local_maxima" : {
                "min_local_max_distance" : 100.0
            },
            "use_watershed" : True
        }

        shape = numpy.array((500, 500))

        neuron_centers = numpy.array([[127, 202], [177,  52], [343, 271]])
        neuron_radii = numpy.array((50.0,)*len(neuron_centers))
        neuron_magnitudes = numpy.array((1.0/3.0,)*len(neuron_centers))

        neuron_spreads = neuron_radii / 3.0

        neuron_images = nanshe.syn.data.generate_gaussian_images(shape, neuron_centers, neuron_spreads, neuron_magnitudes)
        neuron_masks = (neuron_images >= (neuron_magnitudes.max() * scipy.stats.norm.pdf(3 * neuron_spreads.max(), scale=neuron_spreads.max())**len(shape)))
        neuron_images *= neuron_masks

        neurons = nanshe.imp.segment.wavelet_denoising(neuron_images.max(axis=0), **params)

        # Resort neuron image order based on most similar.
        result_neurons_distance = scipy.spatial.distance.cdist(neuron_images.reshape(neurons.shape + (-1,)), neurons["image"].reshape(neurons.shape + (-1,)))

        neuron_centers_old = neuron_centers
        neuron_radii_old = neuron_radii
        neuron_magnitudes_old = neuron_magnitudes
        neuron_images_old = neuron_images
        neuron_masks_old = neuron_masks

        neuron_centers = numpy.zeros(neuron_centers_old.shape, dtype=neuron_centers_old.dtype)
        neuron_radii = numpy.zeros(neuron_radii_old.shape, dtype=neuron_radii_old.dtype)
        neuron_magnitudes = numpy.zeros(neuron_magnitudes_old.shape, dtype=neuron_magnitudes_old.dtype)
        neuron_images = numpy.zeros(neuron_images_old.shape, dtype=neuron_images_old.dtype)
        neuron_masks = numpy.zeros(neuron_masks_old.shape, dtype=neuron_masks_old.dtype)
        for i1, i2 in enumerate(result_neurons_distance.argmin(axis=1)):
            neuron_centers[i1] = neuron_centers_old[i2]
            neuron_radii[i1] = neuron_radii_old[i2]
            neuron_magnitudes[i1] = neuron_magnitudes_old[i2]
            neuron_images[i1] = neuron_images_old[i2]
            neuron_masks[i1] = neuron_masks_old[i2]
        neuron_centers_old = None
        neuron_radii_old = None
        neuron_magnitudes_old = None
        neuron_images_old = None
        neuron_masks_old = None

        assert (len(neuron_centers) == len(neurons))
        assert (numpy.abs(neurons["image"].max(axis=0) - neuron_images.max(axis=0)).max() < 1.0e-4)
        assert (numpy.abs(neurons["image"] - neuron_images).max() < 1.0e-4)

    @nose.plugins.attrib.attr("3D")
    def test_wavelet_denoising_3(self):
        params = {
            "remove_low_intensity_local_maxima" : {
                "percentage_pixels_below_max" : 0
            },
            "wavelet.transform" : {
                "scale" : 5
            },
            "accepted_region_shape_constraints" : {
                "major_axis_length" : {
                    "max" : 30.0,
                    "min" : 0.0
                }
            },
            "accepted_neuron_shape_constraints" : {
                "eccentricity" : {
                    "max" : 0.9,
                    "min" : 0.0
                },
                "area" : {
                    "max" : 30000,
                    "min" : 10000
                }
            },
            "estimate_noise" : {
                "significance_threshold" : 3.0
            },
            "significant_mask" : {
                "noise_threshold" : 3.0
            },
            "remove_too_close_local_maxima" : {
                "min_local_max_distance" : 100.0
            },
            "use_watershed" : True
        }

        shape = numpy.array((100, 100, 100))

        neuron_centers = numpy.array([[21, 17, 46], [46, 71, 83], [77, 52, 17]])
        neuron_radii = numpy.array((10.0,)*len(neuron_centers))
        neuron_magnitudes = numpy.array((1.0/3.0,)*len(neuron_centers))

        neuron_spreads = neuron_radii / 3.0

        neuron_images = nanshe.syn.data.generate_gaussian_images(shape, neuron_centers, neuron_spreads, neuron_magnitudes)
        neuron_masks = (neuron_images >= (neuron_magnitudes.max() * scipy.stats.norm.pdf(3 * neuron_spreads.max(), scale=neuron_spreads.max())**len(shape)))
        neuron_images *= neuron_masks

        neurons = nanshe.imp.segment.wavelet_denoising(neuron_images.max(axis=0), **params)

        # Resort neuron image order based on most similar.
        result_neurons_distance = scipy.spatial.distance.cdist(neuron_images.reshape(neurons.shape + (-1,)), neurons["image"].reshape(neurons.shape + (-1,)))

        neuron_centers_old = neuron_centers
        neuron_radii_old = neuron_radii
        neuron_magnitudes_old = neuron_magnitudes
        neuron_images_old = neuron_images
        neuron_masks_old = neuron_masks

        neuron_centers = numpy.zeros(neuron_centers_old.shape, dtype=neuron_centers_old.dtype)
        neuron_radii = numpy.zeros(neuron_radii_old.shape, dtype=neuron_radii_old.dtype)
        neuron_magnitudes = numpy.zeros(neuron_magnitudes_old.shape, dtype=neuron_magnitudes_old.dtype)
        neuron_images = numpy.zeros(neuron_images_old.shape, dtype=neuron_images_old.dtype)
        neuron_masks = numpy.zeros(neuron_masks_old.shape, dtype=neuron_masks_old.dtype)
        for i1, i2 in enumerate(result_neurons_distance.argmin(axis=1)):
            neuron_centers[i1] = neuron_centers_old[i2]
            neuron_radii[i1] = neuron_radii_old[i2]
            neuron_magnitudes[i1] = neuron_magnitudes_old[i2]
            neuron_images[i1] = neuron_images_old[i2]
            neuron_masks[i1] = neuron_masks_old[i2]
        neuron_centers_old = None
        neuron_radii_old = None
        neuron_magnitudes_old = None
        neuron_images_old = None
        neuron_masks_old = None

        assert (len(neuron_centers) == len(neurons))
        assert (numpy.abs(neurons["image"].max(axis=0) - neuron_images.max(axis=0)).max() < 1.0e-6)
        assert (numpy.abs(neurons["image"] - neuron_images).max() < 1.0e-6)

    def test_extract_neurons_1(self):
        image = 5 * numpy.ones((100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25], [74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.util.xnumpy.expand_view(circle_centers, image.shape) - \
        nanshe.util.xnumpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis=1)**.5 < nanshe.util.xnumpy.expand_view(circle_radii, image.shape))
        circle_images = circle_masks * image

        circle_mask_mean = numpy.zeros((len(circle_masks), image.ndim,))
        circle_mask_cov = numpy.zeros((len(circle_masks), image.ndim, image.ndim,))
        for circle_mask_i in xrange(len(circle_masks)):
            each_circle_mask_points = numpy.array(circle_masks[circle_mask_i].nonzero(), dtype=float)

            circle_mask_mean[circle_mask_i] = each_circle_mask_points.mean(axis=1)
            circle_mask_cov[circle_mask_i] = numpy.cov(each_circle_mask_points)

        neurons = nanshe.imp.segment.extract_neurons(image, circle_masks)

        assert (len(circle_masks) == len(neurons))

        assert (circle_masks == neurons["mask"]).all()

        assert (circle_images == neurons["image"]).all()

        assert (numpy.apply_over_axes(numpy.sum, circle_masks, range(1, circle_masks.ndim)) == neurons["area"]).all()

        assert (numpy.apply_over_axes(numpy.max, circle_images, range(1, circle_masks.ndim)) == neurons["max_F"]).all()

        assert (circle_mask_mean == neurons["gaussian_mean"]).all()

        assert (circle_mask_cov == neurons["gaussian_cov"]).all()

        assert (neurons["centroid"] == neurons["gaussian_mean"]).all()

    @nose.plugins.attrib.attr("3D")
    def test_extract_neurons_2(self):
        image = 5 * numpy.ones((100, 100, 100))

        xyz = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25, 25], [74, 74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.util.xnumpy.expand_view(circle_centers, image.shape) - \
        nanshe.util.xnumpy.expand_view(xyz, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis=1)**.5 < nanshe.util.xnumpy.expand_view(circle_radii, image.shape))
        circle_images = circle_masks * image

        circle_mask_mean = numpy.zeros((len(circle_masks), image.ndim,))
        circle_mask_cov = numpy.zeros((len(circle_masks), image.ndim, image.ndim,))
        for circle_mask_i in xrange(len(circle_masks)):
            each_circle_mask_points = numpy.array(circle_masks[circle_mask_i].nonzero(), dtype=float)

            circle_mask_mean[circle_mask_i] = each_circle_mask_points.mean(axis=1)
            circle_mask_cov[circle_mask_i] = numpy.cov(each_circle_mask_points)

        neurons = nanshe.imp.segment.extract_neurons(image, circle_masks)

        assert (len(circle_masks) == len(neurons))

        assert (circle_masks == neurons["mask"]).all()

        assert (circle_images == neurons["image"]).all()

        assert (numpy.apply_over_axes(numpy.sum, circle_masks, range(1, circle_masks.ndim)) == neurons["area"]).all()

        assert (numpy.apply_over_axes(numpy.max, circle_images, range(1, circle_masks.ndim)) == neurons["max_F"]).all()

        assert (circle_mask_mean == neurons["gaussian_mean"]).all()

        assert (circle_mask_cov == neurons["gaussian_cov"]).all()

        assert (neurons["centroid"] == neurons["gaussian_mean"]).all()

    def test_fuse_neurons_1(self):
        fraction_mean_neuron_max_threshold = 0.01

        image = 5 * numpy.ones((100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25], [74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.util.xnumpy.expand_view(circle_centers, image.shape) - \
        nanshe.util.xnumpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis=1)**.5 < nanshe.util.xnumpy.expand_view(circle_radii, image.shape))

        circle_mask_mean = numpy.zeros((len(circle_masks), image.ndim,))
        circle_mask_cov = numpy.zeros((len(circle_masks), image.ndim, image.ndim,))
        for circle_mask_i in xrange(len(circle_masks)):
            each_circle_mask_points = numpy.array(circle_masks[circle_mask_i].nonzero(), dtype=float)

            circle_mask_mean[circle_mask_i] = each_circle_mask_points.mean(axis=1)
            circle_mask_cov[circle_mask_i] = numpy.cov(each_circle_mask_points)

        neurons = nanshe.imp.segment.extract_neurons(image, circle_masks)

        fused_neurons = nanshe.imp.segment.fuse_neurons(neurons[0], neurons[1],
                                                        fraction_mean_neuron_max_threshold)

        assert (neurons["mask"].sum(axis=0) == fused_neurons["mask"]).all()

        assert (neurons["image"].mean(axis=0) == fused_neurons["image"]).all()

        assert (numpy.array(neurons["area"].sum()) == fused_neurons["area"])

        assert (fused_neurons["image"].max() == fused_neurons["max_F"])

        assert (neurons["gaussian_mean"].mean(axis=0) == fused_neurons["gaussian_mean"]).all()

        assert (fused_neurons["centroid"] == fused_neurons["gaussian_mean"]).all()

    @nose.plugins.attrib.attr("3D")
    def test_fuse_neurons_2(self):
        fraction_mean_neuron_max_threshold = 0.01

        image = 5 * numpy.ones((100, 100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25, 25], [74, 74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.util.xnumpy.expand_view(circle_centers, image.shape) - \
        nanshe.util.xnumpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis=1)**.5 < nanshe.util.xnumpy.expand_view(circle_radii, image.shape))

        circle_mask_mean = numpy.zeros((len(circle_masks), image.ndim,))
        circle_mask_cov = numpy.zeros((len(circle_masks), image.ndim, image.ndim,))
        for circle_mask_i in xrange(len(circle_masks)):
            each_circle_mask_points = numpy.array(circle_masks[circle_mask_i].nonzero(), dtype=float)

            circle_mask_mean[circle_mask_i] = each_circle_mask_points.mean(axis=1)
            circle_mask_cov[circle_mask_i] = numpy.cov(each_circle_mask_points)

        neurons = nanshe.imp.segment.extract_neurons(image, circle_masks)

        fused_neurons = nanshe.imp.segment.fuse_neurons(neurons[0], neurons[1],
                                                        fraction_mean_neuron_max_threshold)

        assert (neurons["mask"].sum(axis=0) == fused_neurons["mask"]).all()

        assert (neurons["image"].mean(axis=0) == fused_neurons["image"]).all()

        assert (numpy.array(neurons["area"].sum()) == fused_neurons["area"])

        assert (fused_neurons["image"].max() == fused_neurons["max_F"])

        assert (neurons["gaussian_mean"].mean(axis=0) == fused_neurons["gaussian_mean"]).all()

        assert (fused_neurons["centroid"] == fused_neurons["gaussian_mean"]).all()

    def test_merge_neuron_sets_1(self):
        alignment_min_threshold = 0.6
        overlap_min_threshold = 0.6
        fuse_neurons = {"fraction_mean_neuron_max_threshold" : 0.01}

        image = 5 * numpy.ones((100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25], [74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.util.xnumpy.expand_view(circle_centers, image.shape) - \
        nanshe.util.xnumpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis=1)**.5 < nanshe.util.xnumpy.expand_view(circle_radii, image.shape))

        neurons = nanshe.imp.segment.extract_neurons(image, circle_masks)

        merged_neurons = nanshe.imp.segment.merge_neuron_sets(neurons[:1], neurons[1:], alignment_min_threshold, overlap_min_threshold, fuse_neurons=fuse_neurons)

        assert (len(neurons) == len(circle_centers))

        assert (neurons == merged_neurons).all()

    def test_merge_neuron_sets_2(self):
        alignment_min_threshold = 0.6
        overlap_min_threshold = 0.6
        fuse_neurons = {"fraction_mean_neuron_max_threshold" : 0.01}

        image = 5 * numpy.ones((100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25]])

        circle_radii = numpy.array([25])

        circle_offsets = nanshe.util.xnumpy.expand_view(circle_centers, image.shape) - \
        nanshe.util.xnumpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis=1)**.5 < nanshe.util.xnumpy.expand_view(circle_radii, image.shape))

        neurons = nanshe.imp.segment.extract_neurons(image, circle_masks)

        merged_neurons = nanshe.imp.segment.merge_neuron_sets(neurons, neurons, alignment_min_threshold, overlap_min_threshold, fuse_neurons=fuse_neurons)

        assert (len(neurons) == len(circle_centers))

        assert (neurons == merged_neurons).all()

    @nose.plugins.attrib.attr("3D")
    def test_merge_neuron_sets_3(self):
        alignment_min_threshold = 0.6
        overlap_min_threshold = 0.6
        fuse_neurons = {"fraction_mean_neuron_max_threshold" : 0.01}

        image = 5 * numpy.ones((100, 100, 100))

        xyz = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25, 25], [74, 74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.util.xnumpy.expand_view(circle_centers, image.shape) - \
        nanshe.util.xnumpy.expand_view(xyz, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis=1)**.5 < nanshe.util.xnumpy.expand_view(circle_radii, image.shape))

        neurons = nanshe.imp.segment.extract_neurons(image, circle_masks)

        merged_neurons = nanshe.imp.segment.merge_neuron_sets(neurons[:1], neurons[1:], alignment_min_threshold, overlap_min_threshold, fuse_neurons=fuse_neurons)

        assert (len(neurons) == len(circle_centers))

        assert (neurons == merged_neurons).all()

    @nose.plugins.attrib.attr("3D")
    def test_merge_neuron_sets_4(self):
        alignment_min_threshold = 0.6
        overlap_min_threshold = 0.6
        fuse_neurons = {"fraction_mean_neuron_max_threshold" : 0.01}

        image = 5 * numpy.ones((100, 100, 100))

        xyz = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25, 25]])

        circle_radii = numpy.array([25])

        circle_offsets = nanshe.util.xnumpy.expand_view(circle_centers, image.shape) - \
        nanshe.util.xnumpy.expand_view(xyz, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis=1)**.5 < nanshe.util.xnumpy.expand_view(circle_radii, image.shape))

        neurons = nanshe.imp.segment.extract_neurons(image, circle_masks)

        merged_neurons = nanshe.imp.segment.merge_neuron_sets(neurons, neurons, alignment_min_threshold, overlap_min_threshold, fuse_neurons=fuse_neurons)

        assert (len(neurons) == len(circle_centers))

        assert (neurons == merged_neurons).all()

    def test_postprocess_data_1(self):
        config = {
            "wavelet_denoising" : {
                "remove_low_intensity_local_maxima" : {
                    "percentage_pixels_below_max" : 0.0
                },
                "wavelet.transform" : {
                    "scale" : 4
                },
                "accepted_region_shape_constraints" : {
                    "major_axis_length" : {
                        "max" : 25.0,
                        "min" : 0.0
                    }
                },
                "accepted_neuron_shape_constraints" : {
                    "eccentricity" : {
                        "max" : 0.9,
                        "min" : 0.0
                    },
                    "area" : {
                        "max" : 600,
                        "min" : 30
                    }
                },
                "estimate_noise" : {
                    "significance_threshold" : 3.0
                },
                "significant_mask" : {
                    "noise_threshold" : 3.0
                },
                "remove_too_close_local_maxima" : {
                    "min_local_max_distance" : 10.0
                },
                "use_watershed" : True
            },
            "merge_neuron_sets" : {
                "alignment_min_threshold" : 0.6,
                "fuse_neurons" : {
                    "fraction_mean_neuron_max_threshold" : 0.01
                },
                "overlap_min_threshold" : 0.6
            }
        }

        space = numpy.array([100, 100])
        radii = numpy.array([7, 6, 6, 6, 7, 6])
        magnitudes = numpy.array([15, 16, 15, 17, 16, 16])
        points = numpy.array([[30, 24],
                              [59, 65],
                              [21, 65],
                              [13, 12],
                              [72, 16],
                              [45, 32]])

        masks = nanshe.syn.data.generate_hypersphere_masks(space, points, radii)
        images = nanshe.syn.data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks

        bases_indices = [[1,3,4], [0,2], [5]]

        bases_masks = numpy.zeros((len(bases_indices),) + masks.shape[1:], dtype=masks.dtype)
        bases_images = numpy.zeros((len(bases_indices),) + images.shape[1:], dtype=images.dtype)

        for i, each_basis_indices in enumerate(bases_indices):
            bases_masks[i] = masks[list(each_basis_indices)].max(axis=0)
            bases_images[i] = images[list(each_basis_indices)].max(axis=0)

        neurons = nanshe.imp.segment.postprocess_data(bases_images, **config)

        assert (len(points) == len(neurons))

        neuron_max_matches = nanshe.util.xnumpy.all_permutations_equal(neurons["max_F"], neurons["image"])
        neuron_max_matches = neuron_max_matches.max(axis=0).max(axis=0)

        neuron_points = numpy.array(neuron_max_matches.nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(points))
        for i in xrange(len(neuron_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_points[i] == points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    def test_postprocess_data_2(self):
        config = {
            "wavelet_denoising" : {
                "remove_low_intensity_local_maxima" : {
                    "percentage_pixels_below_max" : 0.0
                },
                "wavelet.transform" : {
                    "scale" : 4
                },
                "accepted_region_shape_constraints" : {
                    "major_axis_length" : {
                        "max" : 25.0,
                        "min" : 0.0
                    }
                },
                "accepted_neuron_shape_constraints" : {
                    "eccentricity" : {
                        "max" : 0.9,
                        "min" : 0.0
                    },
                    "area" : {
                        "max" : 600,
                        "min" : 30
                    }
                },
                "estimate_noise" : {
                    "significance_threshold" : 3.0
                },
                "significant_mask" : {
                    "noise_threshold" : 3.0
                },
                "remove_too_close_local_maxima" : {
                    "min_local_max_distance" : 10.0
                },
                "use_watershed" : True
            },
            "merge_neuron_sets" : {
                "alignment_min_threshold" : 0.6,
                "fuse_neurons" : {
                    "fraction_mean_neuron_max_threshold" : 0.01
                },
                "overlap_min_threshold" : 0.6
            }
        }

        space = numpy.array([100, 100])
        radii = numpy.array([25])
        magnitudes = numpy.array([15])
        points = numpy.array([[25, 25]])

        masks = nanshe.syn.data.generate_hypersphere_masks(space, numpy.vstack([points, points]), numpy.hstack([radii, radii]))
        images = nanshe.syn.data.generate_gaussian_images(space, numpy.vstack([points, points]), numpy.hstack([radii, radii])/3.0, numpy.hstack([magnitudes, magnitudes])) * masks


        print(masks.shape)

        bases_indices = [[0], [1]]

        bases_masks = numpy.zeros((len(bases_indices),) + masks.shape[1:], dtype=masks.dtype)
        bases_images = numpy.zeros((len(bases_indices),) + images.shape[1:], dtype=images.dtype)

        for i, each_basis_indices in enumerate(bases_indices):
            bases_masks[i] = masks[list(each_basis_indices)].max(axis=0)
            bases_images[i] = images[list(each_basis_indices)].max(axis=0)

        neurons = nanshe.imp.segment.postprocess_data(bases_images, **config)

        assert (len(points) == len(neurons))

        neuron_max_matches = nanshe.util.xnumpy.all_permutations_equal(neurons["max_F"], neurons["image"])
        neuron_max_matches = neuron_max_matches.max(axis=0).max(axis=0)

        neuron_points = numpy.array(neuron_max_matches.nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(points))
        for i in xrange(len(neuron_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_points[i] == points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_postprocess_data_3(self):
        config = {
            "wavelet_denoising" : {
                "remove_low_intensity_local_maxima" : {
                    "percentage_pixels_below_max" : 0.0
                },
                "wavelet.transform" : {
                    "scale" : 4
                },
                "accepted_region_shape_constraints" : {
                    "major_axis_length" : {
                        "max" : 30.0,
                        "min" : 0.0
                    }
                },
                "accepted_neuron_shape_constraints" : {
                    "eccentricity" : {
                        "max" : 0.9,
                        "min" : 0.0
                    },
                    "area" : {
                        "max" : 6000.0,
                        "min" : 1000.0
                    }
                },
                "estimate_noise" : {
                    "significance_threshold" : 3.0
                },
                "significant_mask" : {
                    "noise_threshold" : 3.0
                },
                "remove_too_close_local_maxima" : {
                    "min_local_max_distance" : 20.0
                },
                "use_watershed" : True
            },
            "merge_neuron_sets" : {
                "alignment_min_threshold" : 0.6,
                "fuse_neurons" : {
                    "fraction_mean_neuron_max_threshold" : 0.01
                },
                "overlap_min_threshold" : 0.6
            }
        }

        space = numpy.array([100, 100, 100])
        radii = numpy.array([7, 6, 6, 6, 7, 6])
        magnitudes = numpy.array([15, 16, 15, 17, 16, 16])
        points = numpy.array([[30, 24, 68],
                              [59, 65, 47],
                              [21, 65, 21],
                              [13, 12, 21],
                              [72, 16, 67],
                              [45, 32, 27]])

        masks = nanshe.syn.data.generate_hypersphere_masks(space, points, radii)
        images = nanshe.syn.data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks

        bases_indices = [[1,3,4], [0,2], [5]]

        bases_masks = numpy.zeros((len(bases_indices),) + masks.shape[1:], dtype=masks.dtype)
        bases_images = numpy.zeros((len(bases_indices),) + images.shape[1:], dtype=images.dtype)

        for i, each_basis_indices in enumerate(bases_indices):
            bases_masks[i] = masks[list(each_basis_indices)].max(axis=0)
            bases_images[i] = images[list(each_basis_indices)].max(axis=0)

        neurons = nanshe.imp.segment.postprocess_data(bases_images, **config)

        assert (len(points) == len(neurons))

        neuron_max_matches = nanshe.util.xnumpy.all_permutations_equal(neurons["max_F"], neurons["image"])
        neuron_max_matches = neuron_max_matches.max(axis=0).max(axis=0)

        neuron_points = numpy.array(neuron_max_matches.nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(points))
        for i in xrange(len(neuron_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_points[i] == points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_postprocess_data_4(self):
        config = {
            "wavelet_denoising" : {
                "remove_low_intensity_local_maxima" : {
                    "percentage_pixels_below_max" : 0.0
                },
                "wavelet.transform" : {
                    "scale" : 4
                },
                "accepted_region_shape_constraints" : {
                    "major_axis_length" : {
                        "max" : 30.0,
                        "min" : 0.0
                    }
                },
                "accepted_neuron_shape_constraints" : {
                    "eccentricity" : {
                        "max" : 0.9,
                        "min" : 0.0
                    },
                    "area" : {
                        "max" : 70000.0,
                        "min" : 10000.0
                    }
                },
                "estimate_noise" : {
                    "significance_threshold" : 3.0
                },
                "significant_mask" : {
                    "noise_threshold" : 3.0
                },
                "remove_too_close_local_maxima" : {
                    "min_local_max_distance" : 20.0
                },
                "use_watershed" : True
            },
            "merge_neuron_sets" : {
                "alignment_min_threshold" : 0.6,
                "fuse_neurons" : {
                    "fraction_mean_neuron_max_threshold" : 0.01
                },
                "overlap_min_threshold" : 0.6
            }
        }

        space = numpy.array([100, 100, 100])
        radii = numpy.array([25])
        magnitudes = numpy.array([15])
        points = numpy.array([[25, 25, 25]])

        masks = nanshe.syn.data.generate_hypersphere_masks(space, numpy.vstack([points, points]), numpy.hstack([radii, radii]))
        images = nanshe.syn.data.generate_gaussian_images(space, numpy.vstack([points, points]), numpy.hstack([radii, radii])/3.0, numpy.hstack([magnitudes, magnitudes])) * masks

        bases_indices = [[0], [1]]

        bases_masks = numpy.zeros((len(bases_indices),) + masks.shape[1:], dtype=masks.dtype)
        bases_images = numpy.zeros((len(bases_indices),) + images.shape[1:], dtype=images.dtype)

        for i, each_basis_indices in enumerate(bases_indices):
            bases_masks[i] = masks[list(each_basis_indices)].max(axis=0)
            bases_images[i] = images[list(each_basis_indices)].max(axis=0)

        neurons = nanshe.imp.segment.postprocess_data(bases_images, **config)

        assert (len(points) == len(neurons))

        neuron_max_matches = nanshe.util.xnumpy.all_permutations_equal(neurons["max_F"], neurons["image"])
        neuron_max_matches = neuron_max_matches.max(axis=0).max(axis=0)

        neuron_points = numpy.array(neuron_max_matches.nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(points))
        for i in xrange(len(neuron_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_points[i] == points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)
