__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 30, 2014 19:35:11 EDT$"


import numpy
import scipy

import scipy.spatial
import scipy.spatial.distance

import nanshe.expanded_numpy

import nanshe.advanced_image_processing

import synthetic_data


class TestAdvancedImageProcessing(object):
    def test_remove_zeroed_lines_1(self):
        a = numpy.ones((1, 100, 100))
        p = 0.2
        erosion_shape = [ 21, 1 ]
        dilation_shape = [ 1, 3 ]

        r = numpy.array([[0, 0, 0], [a.shape[1]-2, 3, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.expanded_numpy.index_axis_at_pos(nanshe.expanded_numpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.advanced_image_processing.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape)

        assert((a == b).all())

    def test_remove_zeroed_lines_2(self):
        a = numpy.ones((1, 100, 100))
        p = 0.2
        erosion_shape = [ 21, 1 ]
        dilation_shape = [ 1, 3 ]

        r = numpy.array([[0, 0, 0], [1, 3, 4]]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.expanded_numpy.index_axis_at_pos(nanshe.expanded_numpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.advanced_image_processing.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape)

        assert((a == b).all())


    def test_remove_zeroed_lines_3(self):
        a = numpy.ones((1, 100, 100))
        p = 0.2
        erosion_shape = [ 21, 1 ]
        dilation_shape = [ 1, 3 ]

        nr = numpy.random.geometric(p)

        r = numpy.array([numpy.repeat(0, nr), numpy.random.random_integers(1, a.shape[1] - 2, nr)]).T.copy()

        print(r)

        ar = a.copy()
        for each_r in r:
            nanshe.expanded_numpy.index_axis_at_pos(nanshe.expanded_numpy.index_axis_at_pos(ar, 0, each_r[0]), -1, each_r[-1])[:] = 0

        b = nanshe.advanced_image_processing.remove_zeroed_lines(ar, erosion_shape=erosion_shape, dilation_shape=dilation_shape)

        assert((a == b).all())

    def test_extract_f0_1(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        half_window_size = 400
        bias = 100
        step_size = 100

        a = numpy.ones((100, 100, 100))

        b = nanshe.advanced_image_processing.extract_f0(a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            half_window_size=half_window_size,
            bias=bias,
            step_size=step_size)

        assert((b == 0).all())

    def test_extract_f0_2(self):
        spatial_smoothing_gaussian_filter_stdev = 5.0
        which_quantile = 0.5
        temporal_smoothing_gaussian_filter_stdev = 5.0
        half_window_size = 400
        bias = 100
        step_size = 100

        mean = 0.0
        stdev = 1.0

        a = numpy.random.normal(mean, stdev, (100, 100, 100))

        b = nanshe.advanced_image_processing.extract_f0(a,
            spatial_smoothing_gaussian_filter_stdev=spatial_smoothing_gaussian_filter_stdev,
            which_quantile=which_quantile,
            temporal_smoothing_gaussian_filter_stdev=temporal_smoothing_gaussian_filter_stdev,
            half_window_size=half_window_size,
            bias=bias,
            step_size=step_size)

        # Seems to be basically 2 orders of magnitude in reduction. However, it may be a little above exactly two.
        # Hence, multiplication by 99 instead of 100.
        assert( (99.0*b.std()) < a.std() )

        # Turns out that a difference greater than 0.1 will be over 10 standard deviations away.
        assert( ((a - 100.0*b) < 0.1).all() )

    def test_preprocess_data(self):
        ## Does NOT test accuracy.

        config = {
            "normalize_data" : {
                "simple_image_processing.renormalized_images" : {
                    "ord" : 2
                }
            },
            "extract_f0" : {
                "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                "which_quantile" : 0.5,
                "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                "half_window_size" : 20,
                "bias" : 100,
                "step_size" : 10
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
            "wavelet_transform" : {
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

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        image_stack = images.max(axis = 0)

        nanshe.advanced_image_processing.preprocess_data(image_stack, **config)

    def test_generate_dictionary(self):
        p = numpy.array([[27, 51],
                     [66, 85],
                     [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))

        g = synthetic_data.generate_hypersphere_masks(space, p, radii)

        d = nanshe.advanced_image_processing.generate_dictionary(g.astype(float),
                                                                                    **{
                                                                                        "spams.trainDL" : {
                                                                                            "gamma2" : 0,
                                                                                            "gamma1" : 0,
                                                                                             "numThreads" : -1,
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

        assert(g.shape == d.shape)

        assert((g.astype(bool).max(axis = 0) == d.astype(bool).max(axis = 0)).all())

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

        print unmatched_g

        assert(len(unmatched_g) == 0)

    def test_generate_local_maxima_vigra(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype = float)

        g = synthetic_data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        m = nanshe.advanced_image_processing.generate_local_maxima_vigra(g.max(axis = 0))

        assert((numpy.array(m.nonzero()) == p.T).all())

    def test_generate_local_maxima_scikit_image(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype = float)

        g = synthetic_data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        m = nanshe.advanced_image_processing.generate_local_maxima_scikit_image(g.max(axis = 0))

        assert((numpy.array(m.nonzero()) == p.T).all())

    def test_generate_local_maxima(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype = float)

        g = synthetic_data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        m = nanshe.advanced_image_processing.generate_local_maxima(g.max(axis = 0))

        assert((numpy.array(m.nonzero()) == p.T).all())

    def test_extended_region_local_maxima_properties_1(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype = float)

        g = synthetic_data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        m = (g > 0.00065)
        g *= m

        e = nanshe.advanced_image_processing.extended_region_local_maxima_properties(g.max(axis = 0),
                nanshe.expanded_numpy.enumerate_masks(m).max(axis = 0)
        )

        assert((numpy.bincount(e["label"])[1:]  == 1).all())

        assert(len(e) == len(p))

        assert((e["local_max"] == p).all())

        assert((e["area"] == numpy.apply_over_axes(numpy.sum, m, axes = range(1, m.ndim)).squeeze().astype(float)).all())

        assert((e["centroid"] == e["local_max"]).all())

        assert((e["intensity"] == g.max(axis = 0)[tuple(p.T)]).all())

    def test_extended_region_local_maxima_properties_2(self):
        p = numpy.array([[27, 51],
                         [32, 53],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype = float)

        g = synthetic_data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        g = numpy.array([g[0] + g[1], g[2]])
        m = (g > 0.00065)
        g *= m

        e = nanshe.advanced_image_processing.extended_region_local_maxima_properties(g.max(axis = 0),
                nanshe.expanded_numpy.enumerate_masks(m).max(axis = 0)
        )

        assert((numpy.bincount(e["label"])[1:] == numpy.array([2, 1])).all())

        assert(len(e) == len(p))

        assert((e["local_max"] == p).all())

        assert((e["area"][[0, 2]] == numpy.apply_over_axes(numpy.sum, m, axes = range(1, m.ndim)).squeeze().astype(float)).all())

        # Not exactly equal due to floating point round off error
        assert(((e["centroid"][0] - numpy.array(m[0].nonzero()).mean(axis = 1)) < 1e-14).all())

        # Not exactly equal due to floating point round off error
        assert(((e["centroid"][1] - numpy.array(m[0].nonzero()).mean(axis = 1)) < 1e-14).all())

        assert((e["centroid"][2] == e["local_max"][2]).all())

        assert((e["intensity"] == g.max(axis = 0)[tuple(p.T)]).all())

    def test_remove_low_intensity_local_maxima_1(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype = float)
        points = numpy.array([[23, 36],
                           [58, 64]])

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        labels = nanshe.expanded_numpy.enumerate_masks(masks).max(axis = 0)

        e = nanshe.advanced_image_processing.ExtendedRegionProps(images.max(axis = 0), labels)

        e2 = nanshe.advanced_image_processing.remove_low_intensity_local_maxima(e, 1.0)

        assert(len(points) == len(e.props))

        assert(0 == len(e2.props))

    def test_remove_low_intensity_local_maxima_2(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype = float)
        points = numpy.array([[23, 36],
                           [58, 64]])

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        labels = nanshe.expanded_numpy.enumerate_masks(masks).max(axis = 0)

        e = nanshe.advanced_image_processing.ExtendedRegionProps(images.max(axis = 0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis = 0)[masks[i].nonzero()] < images.max(axis = 0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.advanced_image_processing.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[0])

        assert(len(points) == len(e.props))

        assert(len(e.props) == len(e2.props))

    def test_remove_low_intensity_local_maxima_3(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype = float)
        points = numpy.array([[23, 36],
                           [58, 64]])

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        labels = nanshe.expanded_numpy.enumerate_masks(masks).max(axis = 0)

        e = nanshe.advanced_image_processing.ExtendedRegionProps(images.max(axis = 0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis = 0)[masks[i].nonzero()] < images.max(axis = 0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.advanced_image_processing.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[1])

        assert(len(points) == len(e.props))

        assert((len(e.props) - 1) == len(e2.props))

    def test_remove_low_intensity_local_maxima_4(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 10))
        magnitudes = numpy.array((1, 1), dtype = float)
        points = numpy.array([[23, 36],
                           [58, 64]])

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        labels = nanshe.expanded_numpy.enumerate_masks(masks).max(axis = 0)

        e = nanshe.advanced_image_processing.ExtendedRegionProps(images.max(axis = 0), labels)

        percentage_pixels_below_max = numpy.zeros((len(masks),), float)
        for i in xrange(len(masks)):
            pixels_below_max = (images.max(axis = 0)[masks[i].nonzero()] < images.max(axis = 0)[masks[i]].max()).sum()
            pixels = masks[i].sum()

            percentage_pixels_below_max[i] = float(pixels_below_max) / float(pixels)

        percentage_pixels_below_max = numpy.sort(percentage_pixels_below_max)

        e2 = nanshe.advanced_image_processing.remove_low_intensity_local_maxima(e, percentage_pixels_below_max[1] + \
                                                                                   numpy.finfo(float).eps)

        assert(len(points) == len(e.props))

        assert((len(e.props) - 2) == len(e2.props))

    def test_remove_too_close_local_maxima_1(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1), dtype = float)
        points = numpy.array([[63, 69],
                              [58, 64]])

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        labels = masks.max(axis = 0).astype(int)

        e = nanshe.advanced_image_processing.ExtendedRegionProps(images.max(axis = 0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.advanced_image_processing.remove_too_close_local_maxima(e, dist)

        assert(len(points) == len(e.props))

        assert(1 == len(e2.props))

    def test_remove_too_close_local_maxima_2(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1), dtype = float)
        points = numpy.array([[63, 69],
                              [58, 64]])

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        labels = nanshe.expanded_numpy.enumerate_masks(masks).max(axis = 0)

        e = nanshe.advanced_image_processing.ExtendedRegionProps(images.max(axis = 0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.advanced_image_processing.remove_too_close_local_maxima(e, dist)

        assert(len(points) == len(e.props))

        assert(len(points) == len(e2.props))

    def test_remove_too_close_local_maxima_3(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1, 1.01), dtype = float)
        points = numpy.array([[63, 69],
                              [58, 64]])

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        labels = masks.max(axis = 0).astype(int)

        e = nanshe.advanced_image_processing.ExtendedRegionProps(images.max(axis = 0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.advanced_image_processing.remove_too_close_local_maxima(e, dist)

        assert(len(points) == len(e.props))

        assert(1 == len(e2.props))

        assert((points[magnitudes == magnitudes.max()] == e2.props["local_max"][0]).all())

    def test_remove_too_close_local_maxima_4(self):
        space = numpy.array((100, 100))
        radii = numpy.array((5, 5))
        magnitudes = numpy.array((1.01, 1), dtype = float)
        points = numpy.array([[63, 69],
                              [58, 64]])

        masks = synthetic_data.generate_hypersphere_masks(space, points, radii)
        images = synthetic_data.generate_gaussian_images(space, points, radii/3.0, magnitudes) * masks
        labels = masks.max(axis = 0).astype(int)

        e = nanshe.advanced_image_processing.ExtendedRegionProps(images.max(axis = 0), labels)

        dist = scipy.spatial.distance.pdist(points).max()
        i = 0
        while (dist + i * numpy.finfo(type(dist)).eps) == dist:
            i += 1
        dist += i * numpy.finfo(type(dist)).eps

        e2 = nanshe.advanced_image_processing.remove_too_close_local_maxima(e, dist)

        assert(len(points) == len(e.props))

        assert(1 == len(e2.props))

        assert((points[magnitudes == magnitudes.max()] == e2.props["local_max"][0]).all())

    def test_wavelet_denoising(self):
        params = {
            "remove_low_intensity_local_maxima" : {
                "percentage_pixels_below_max" : 0
            },
            "wavelet_transform.wavelet_transform" : {
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
            "denoising.estimate_noise" : {
                "significance_threshhold" : 3.0
            },
            "denoising.significant_mask" : {
                "noise_threshhold" : 3.0
            },
            "remove_too_close_local_maxima" : {
                "min_local_max_distance" : 100.0
            },
            "use_watershed" : True
        }

        shape = numpy.array((500, 500))

        neuron_centers = numpy.array([[177,  52], [127, 202], [343, 271]])
        original_neuron_image = synthetic_data.generate_gaussian_images(shape, neuron_centers, (50.0/3.0,)*len(neuron_centers), (1.0/3.0,)*len(neuron_centers)).sum(axis = 0)
        original_neurons_mask = (original_neuron_image >= 0.00014218114898827068)

        neurons = nanshe.advanced_image_processing.wavelet_denoising(original_neuron_image, **params)

        assert(len(neuron_centers) == len(neurons))
        assert((original_neurons_mask == neurons["mask"].max(axis = 0)).all())
        assert(((original_neurons_mask*original_neuron_image) == neurons["image"].max(axis = 0)).all())

    def test_extract_neurons(self):
        image = 5 * numpy.ones((100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25], [74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.expanded_numpy.expand_view(circle_centers, image.shape) - \
                         nanshe.expanded_numpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis = 1)**.5 < nanshe.expanded_numpy.expand_view(circle_radii, image.shape))
        circle_images = circle_masks * image

        circle_mask_mean = numpy.zeros((len(circle_masks), image.ndim,))
        circle_mask_cov = numpy.zeros((len(circle_masks), image.ndim, image.ndim,))
        for circle_mask_i in xrange(len(circle_masks)):
            each_circle_mask_points = numpy.array(circle_masks[circle_mask_i].nonzero(), dtype=float)

            circle_mask_mean[circle_mask_i] = each_circle_mask_points.mean(axis = 1)
            circle_mask_cov[circle_mask_i] = numpy.cov(each_circle_mask_points)

        neurons = nanshe.advanced_image_processing.extract_neurons(image, circle_masks)

        assert(len(circle_masks) == len(neurons))

        assert((circle_masks == neurons["mask"]).all())

        assert((circle_images == neurons["image"]).all())

        assert((numpy.apply_over_axes(numpy.sum, circle_masks, range(1, circle_masks.ndim)) == neurons["area"]).all())

        assert((numpy.apply_over_axes(numpy.max, circle_images, range(1, circle_masks.ndim)) == neurons["max_F"]).all())

        assert((circle_mask_mean == neurons["gaussian_mean"]).all())

        assert((circle_mask_cov == neurons["gaussian_cov"]).all())

        assert((neurons["centroid"] == neurons["gaussian_mean"]).all())

    def test_fuse_neurons(self):
        fraction_mean_neuron_max_threshold = 0.01

        image = 5 * numpy.ones((100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25], [74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.expanded_numpy.expand_view(circle_centers, image.shape) - \
                         nanshe.expanded_numpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis = 1)**.5 < nanshe.expanded_numpy.expand_view(circle_radii, image.shape))

        circle_mask_mean = numpy.zeros((len(circle_masks), image.ndim,))
        circle_mask_cov = numpy.zeros((len(circle_masks), image.ndim, image.ndim,))
        for circle_mask_i in xrange(len(circle_masks)):
            each_circle_mask_points = numpy.array(circle_masks[circle_mask_i].nonzero(), dtype=float)

            circle_mask_mean[circle_mask_i] = each_circle_mask_points.mean(axis = 1)
            circle_mask_cov[circle_mask_i] = numpy.cov(each_circle_mask_points)

        neurons = nanshe.advanced_image_processing.extract_neurons(image, circle_masks)

        fused_neurons = nanshe.advanced_image_processing.fuse_neurons(neurons[0], neurons[1],
                                                                      fraction_mean_neuron_max_threshold)

        assert((neurons["mask"].sum(axis = 0) == fused_neurons["mask"]).all())

        assert((neurons["image"].mean(axis = 0) == fused_neurons["image"]).all())

        assert(numpy.array(neurons["area"].sum()) == fused_neurons["area"])

        assert(fused_neurons["image"].max() == fused_neurons["max_F"])

        assert((neurons["gaussian_mean"].mean(axis = 0) == fused_neurons["gaussian_mean"]).all())

        assert((fused_neurons["centroid"] == fused_neurons["gaussian_mean"]).all())

    def test_merge_neuron_sets_1(self):
        alignment_min_threshold = 0.6
        overlap_min_threshold = 0.6
        fuse_neurons = {"fraction_mean_neuron_max_threshold" : 0.01}

        image = 5 * numpy.ones((100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25], [74, 74]])

        circle_radii = numpy.array([25, 25])

        circle_offsets = nanshe.expanded_numpy.expand_view(circle_centers, image.shape) - \
                         nanshe.expanded_numpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis = 1)**.5 < nanshe.expanded_numpy.expand_view(circle_radii, image.shape))

        circle_mask_mean = numpy.zeros((len(circle_masks), image.ndim,))
        circle_mask_cov = numpy.zeros((len(circle_masks), image.ndim, image.ndim,))
        for circle_mask_i in xrange(len(circle_masks)):
            each_circle_mask_points = numpy.array(circle_masks[circle_mask_i].nonzero(), dtype=float)

            circle_mask_mean[circle_mask_i] = each_circle_mask_points.mean(axis = 1)
            circle_mask_cov[circle_mask_i] = numpy.cov(each_circle_mask_points)

        neurons = nanshe.advanced_image_processing.extract_neurons(image, circle_masks)

        merged_neurons = nanshe.advanced_image_processing.merge_neuron_sets(neurons[:1], neurons[1:], alignment_min_threshold, overlap_min_threshold, fuse_neurons = fuse_neurons)

        assert(len(neurons) == len(circle_centers))

        assert((neurons == merged_neurons).all())

    def test_merge_neuron_sets_2(self):
        alignment_min_threshold = 0.6
        overlap_min_threshold = 0.6
        fuse_neurons = {"fraction_mean_neuron_max_threshold" : 0.01}

        image = 5 * numpy.ones((100, 100))

        xy = numpy.indices(image.shape)

        circle_centers = numpy.array([[25, 25]])

        circle_radii = numpy.array([25])

        circle_offsets = nanshe.expanded_numpy.expand_view(circle_centers, image.shape) - \
                         nanshe.expanded_numpy.expand_view(xy, reps_before=len(circle_centers))

        circle_offsets_squared = circle_offsets**2

        circle_masks = (circle_offsets_squared.sum(axis = 1)**.5 < nanshe.expanded_numpy.expand_view(circle_radii, image.shape))

        circle_mask_mean = numpy.zeros((len(circle_masks), image.ndim,))
        circle_mask_cov = numpy.zeros((len(circle_masks), image.ndim, image.ndim,))
        for circle_mask_i in xrange(len(circle_masks)):
            each_circle_mask_points = numpy.array(circle_masks[circle_mask_i].nonzero(), dtype=float)

            circle_mask_mean[circle_mask_i] = each_circle_mask_points.mean(axis = 1)
            circle_mask_cov[circle_mask_i] = numpy.cov(each_circle_mask_points)

        neurons = nanshe.advanced_image_processing.extract_neurons(image, circle_masks)

        merged_neurons = nanshe.advanced_image_processing.merge_neuron_sets(neurons, neurons, alignment_min_threshold, overlap_min_threshold, fuse_neurons = fuse_neurons)

        assert(len(neurons) == len(circle_centers))

        assert((neurons == merged_neurons).all())
