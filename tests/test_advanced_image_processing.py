__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 30, 2014 19:35:11 EDT$"


import numpy

import nanshe.expanded_numpy

import nanshe.advanced_image_processing


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

    def test_extract_f0(self):
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
