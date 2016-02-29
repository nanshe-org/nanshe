"""
The module ``data`` provides a few primitives for generating synthetic data.

===============================================================================
Overview
===============================================================================
These include generating mask stacks, gaussian stacks, and random points. These
are primarily used for testing the algorithm, but could be used in benchmarking
or other applications.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 01, 2014 14:55:57 EDT$"


import numpy

import scipy
import scipy.ndimage
import scipy.ndimage.filters

import nanshe.util.iters
import nanshe.util.xnumpy


def generate_hypersphere_masks(space, centers, radii, include_boundary=False):
    """
        Generate a stack of masks (first index indicates which mask); where,
        each contains a hypersphere constructed using a center and radius
        provided.

        Args:
            space(tuple of ints):                The size of the mask.

            centers(list of tuples of numbers):  List of centers with one per
                                                 hypersphere.

            radii(list of numbers):              List of radii with one per
                                                 hypersphere.

            include_boundary(bool):              Whether the mask should
                                                 contain the boundary of the
                                                 hypersphere or not.

        Returns:
            numpy.ndarray:                       A stack of masks (first index
                                                 indicates which mask) with a
                                                 filled hypersphere using a
                                                 center and radius for each
                                                 mask.

        Examples:
            >>> generate_hypersphere_masks((3, 3), (1, 1), 1.25)
            array([[[False,  True, False],
                    [ True,  True,  True],
                    [False,  True, False]]], dtype=bool)

            >>> generate_hypersphere_masks((3, 3, 3), (1, 1, 1), 1.25)
            array([[[[False, False, False],
                     [False,  True, False],
                     [False, False, False]],
            <BLANKLINE>
                    [[False,  True, False],
                     [ True,  True,  True],
                     [False,  True, False]],
            <BLANKLINE>
                    [[False, False, False],
                     [False,  True, False],
                     [False, False, False]]]], dtype=bool)

            >>> generate_hypersphere_masks((9, 9), (4, 4), 5)
            array([[[False, False,  True,  True,  True,  True,  True, False, False],
                    [False,  True,  True,  True,  True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [False,  True,  True,  True,  True,  True,  True,  True, False],
                    [False, False,  True,  True,  True,  True,  True, False, False]]], dtype=bool)

            >>> generate_hypersphere_masks(
            ...     (9, 9), (4, 4), 5, include_boundary=True
            ... )
            array([[[False,  True,  True,  True,  True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [False,  True,  True,  True,  True,  True,  True,  True, False]]], dtype=bool)
    """

    return(generate_hyperdisc_masks(space, centers, radii, include_boundary))


def generate_hyperdisc_masks(space,
                             centers,
                             radii_max,
                             include_boundary_max=False,
                             radii_min=None,
                             include_boundary_min=False):
    """
        Generate a stack of masks (first index indicates which mask); where,
        each contains a hyperdisc constructed using a center and radius
        provided.

        Args:
            space(tuple of ints):                The size of the mask.

            centers(list of tuples of numbers):  List of centers with one per
                                                 hyperdisc.

            radii_max(list of numbers):          List of max radii with one per
                                                 hyperdisc.

            include_boundary_max(bool):          Whether the mask should
                                                 contain the max boundary of
                                                 the hyperdisc or not.

            radii_min(list of numbers):          List of min radii with one per
                                                 hyperdisc.

            include_boundary_min(bool):          Whether the mask should
                                                 contain the min boundary of
                                                 the hyperdisc or not.

        Returns:
            numpy.ndarray:                       A stack of masks (first index
                                                 indicates which mask) with a
                                                 filled hyperdisc using a
                                                 center and radius for each
                                                 mask.

        Examples:
            >>> generate_hyperdisc_masks((3, 3), (1, 1), 1.25)
            array([[[False,  True, False],
                    [ True,  True,  True],
                    [False,  True, False]]], dtype=bool)

            >>> generate_hyperdisc_masks((3, 3), (1, 1), 1.25, radii_min=0.0, include_boundary_min=False)
            array([[[False,  True, False],
                    [ True, False,  True],
                    [False,  True, False]]], dtype=bool)

            >>> generate_hyperdisc_masks((3, 3, 3), (1, 1, 1), 1.25)
            array([[[[False, False, False],
                     [False,  True, False],
                     [False, False, False]],
            <BLANKLINE>
                    [[False,  True, False],
                     [ True,  True,  True],
                     [False,  True, False]],
            <BLANKLINE>
                    [[False, False, False],
                     [False,  True, False],
                     [False, False, False]]]], dtype=bool)

            >>> generate_hyperdisc_masks((9, 9), (4, 4), 5)
            array([[[False, False,  True,  True,  True,  True,  True, False, False],
                    [False,  True,  True,  True,  True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [False,  True,  True,  True,  True,  True,  True,  True, False],
                    [False, False,  True,  True,  True,  True,  True, False, False]]], dtype=bool)

            >>> generate_hyperdisc_masks(
            ...     (9, 9), (4, 4), 5, include_boundary_max=True
            ... )
            array([[[False,  True,  True,  True,  True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True],
                    [False,  True,  True,  True,  True,  True,  True,  True, False]]], dtype=bool)

            >>> generate_hyperdisc_masks(
            ...     (9, 9), (4, 4), 5, True, 4, True
            ... )
            array([[[False,  True,  True,  True,  True,  True,  True,  True, False],
                    [ True,  True, False, False, False, False, False,  True,  True],
                    [ True, False, False, False, False, False, False, False,  True],
                    [ True, False, False, False, False, False, False, False,  True],
                    [ True, False, False, False, False, False, False, False,  True],
                    [ True, False, False, False, False, False, False, False,  True],
                    [ True, False, False, False, False, False, False, False,  True],
                    [ True,  True, False, False, False, False, False,  True,  True],
                    [False,  True,  True,  True,  True,  True,  True,  True, False]]], dtype=bool)
    """

    # Convert to arrays
    space = numpy.array(space)
    centers = numpy.array(centers)
    radii_max = numpy.array(radii_max)
    if radii_min is not None:
        radii_min = numpy.array(radii_min)

    # Add a singleton dimension if there is only one of each.
    if centers.ndim == 1:
        centers = centers[None]

    if radii_max.ndim == 0:
        radii_max = radii_max[None]

    if (radii_min is not None) and (radii_min.ndim == 0):
            radii_min = radii_min[None]

    # Validate the dimensions
    assert (space.ndim == 1)
    assert (centers.ndim == 2)
    assert (radii_max.ndim == 1)
    if radii_min is not None:
        assert (radii_min.ndim == 1)

    # Validate the shapes
    assert (space.shape == centers.shape[1:])
    assert (radii_max.shape == centers.shape[:1])
    if radii_min is not None:
        assert (radii_min.shape == centers.shape[:1])

    radii_max_2 = radii_max**2
    radii_max_2 = radii_max_2[(Ellipsis,) + (None,)*len(space)]
    if radii_min is not None:
        radii_min_2 = radii_min**2
        radii_min_2 = radii_min_2[(Ellipsis,) + (None,)*len(space)]

    space_indices = numpy.indices(space)[None]

    # Create a hyperdisc mask using a center and a radius.
    hyperdisc_masks = None

    point_offsets_dist_2 = (
        (space_indices - centers[(Ellipsis,) + (None,)*len(space)])**2
    ).sum(axis=1)

    if include_boundary_max:
        hyperdisc_masks = (point_offsets_dist_2 <= radii_max_2)
    else:
        hyperdisc_masks = (point_offsets_dist_2 < radii_max_2)
    if radii_min is not None:
        if include_boundary_min:
            hyperdisc_masks &= (radii_min_2 <= point_offsets_dist_2)
        else:
            hyperdisc_masks &= (radii_min_2 < point_offsets_dist_2)

    return(hyperdisc_masks)


def generate_gaussian_images(space, means, std_devs, magnitudes):
    """
        Generate a stack of gaussians (first index indicates which image);
        where, each contains a gaussian using a center and radius provided.

        Note:
            This uses a normalized gaussian filter to create the gaussians. So,
            the sum over any image in the stack will be the magnitude given
            (with some small error).

        Args:
            space(tuple of ints):                The size of the mask.

            means(list of tuples of numbers):    List of means with one per
                                                 gaussian.

            std_devs(list of numbers):           List of standard deviations
                                                 with one per gaussian.

            magnitudes(list of numbers):         List of magnitudes for each
                                                 gaussian.

        Returns:
            numpy.ndarray:                       A stack of masks (first index
                                                 indicates which mask) with a
                                                 gaussian using a center and
                                                 radius for each mask.

        Examples:
            >>> generate_gaussian_images((5, 5), (2, 2), 0.5, 1) #doctest: +NORMALIZE_WHITESPACE
            array([[[ 6.96247819e-08, 2.80886418e-05, 2.07548550e-04, 2.80886418e-05, 6.96247819e-08],
                    [ 2.80886418e-05, 1.13317669e-02, 8.37310610e-02, 1.13317669e-02, 2.80886418e-05],
                    [ 2.07548550e-04, 8.37310610e-02, 6.18693507e-01, 8.37310610e-02, 2.07548550e-04],
                    [ 2.80886418e-05, 1.13317669e-02, 8.37310610e-02, 1.13317669e-02, 2.80886418e-05],
                    [ 6.96247819e-08, 2.80886418e-05, 2.07548550e-04, 2.80886418e-05, 6.96247819e-08]]])

            >>> round(
            ...     generate_gaussian_images((5, 5), (2, 2), 0.5, 1).sum(), 3
            ... )
            1.0

            >>> generate_gaussian_images((5, 5), (2, 2), 0.25, 1) #doctest: +NORMALIZE_WHITESPACE
            array([[[ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [ 0.00000000e+00, 1.12384321e-07, 3.35012940e-04, 1.12384321e-07, 0.00000000e+00],
                    [ 0.00000000e+00, 3.35012940e-04, 9.98659499e-01, 3.35012940e-04, 0.00000000e+00],
                    [ 0.00000000e+00, 1.12384321e-07, 3.35012940e-04, 1.12384321e-07, 0.00000000e+00],
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])

            >>> round(
            ...     generate_gaussian_images((5, 5), (2, 2), 0.25, 1).sum(), 3
            ... )
            1.0

            >>> generate_gaussian_images(
            ...     (5, 5), (2, 2), 0.25, 5
            ... ) #doctest: +NORMALIZE_WHITESPACE
            array([[[ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [ 0.00000000e+00, 5.61921606e-07, 1.67506470e-03, 5.61921606e-07, 0.00000000e+00],
                    [ 0.00000000e+00, 1.67506470e-03, 4.99329749e+00, 1.67506470e-03, 0.00000000e+00],
                    [ 0.00000000e+00, 5.61921606e-07, 1.67506470e-03, 5.61921606e-07, 0.00000000e+00],
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])

            >>> round(
            ...     generate_gaussian_images((5, 5), (2, 2), 0.25, 5).sum(), 3
            ... )
            5.0

            >>> generate_gaussian_images((3, 3, 3), (1, 1, 1), 0.25, 100)
            array([[[[  3.76754623e-09,   1.12308970e-05,   3.76754623e-09],
                     [  1.12308970e-05,   3.34788322e-02,   1.12308970e-05],
                     [  3.76754623e-09,   1.12308970e-05,   3.76754623e-09]],
            <BLANKLINE>
                    [[  1.12308970e-05,   3.34788322e-02,   1.12308970e-05],
                     [  3.34788322e-02,   9.97989922e+01,   3.34788322e-02],
                     [  1.12308970e-05,   3.34788322e-02,   1.12308970e-05]],
            <BLANKLINE>
                    [[  3.76754623e-09,   1.12308970e-05,   3.76754623e-09],
                     [  1.12308970e-05,   3.34788322e-02,   1.12308970e-05],
                     [  3.76754623e-09,   1.12308970e-05,   3.76754623e-09]]]])

            >>> round(
            ...     generate_gaussian_images(
            ...         (3, 3, 3), (1, 1, 1), 0.25, 100
            ...     ).sum(),
            ...     3
            ... )
            100.0
    """

    # Convert to arrays
    space = numpy.array(space)
    means = numpy.array(means)
    std_devs = numpy.array(std_devs)
    magnitudes = numpy.array(magnitudes)

    # Add a singleton dimension if there is only one of each.
    if means.ndim == 1:
        means = means[None]

    if std_devs.ndim == 0:
        std_devs = std_devs[None]

    if magnitudes.ndim == 0:
        magnitudes = magnitudes[None]

    # Validate the dimensions
    assert (space.ndim == 1)
    assert (means.ndim == 2)
    assert (std_devs.ndim == 1)
    assert (magnitudes.ndim == 1)

    # Validate the shapes
    assert (space.shape == means.shape[1:])
    assert (magnitudes.shape == means.shape[:1])

    # Create a gaussian from a mean and a standard deviation.
    images = numpy.zeros(
        magnitudes.shape + tuple(space.tolist()), dtype=float
    )
    for i, (each_mean, each_std_dev, each_magnitude) in enumerate(
            nanshe.util.iters.izip(means, std_devs, magnitudes)
    ):
        images[i][tuple(each_mean)] = each_magnitude
        images[i] = scipy.ndimage.filters.gaussian_filter(
            images[i],
            each_std_dev,
            mode="nearest"
        )

    return(images)


def generate_random_bound_points(space, radii):
    """
        Generate a collection of random points that are the center of a
        hypersphere contained within the space.

        Note:
            Uses the typical convention [0, space_i) for all space_i from
            space. Applies the additional constraint
            [radius_i, space_i - radius_i).

        Args:
            space(tuple of ints):                The size of the space where
                                                 the points will lie.

            radii(list of numbers):              Additional radius for each
                                                 point to remain away from the
                                                 boundary.

        Returns:
            numpy.ndarray:                       An array with the random
                                                 points each using a radius
                                                 from radii to constrain
                                                 itself. First index is which
                                                 point. Second is which
                                                 coordinate.

        Examples:
            >>> numpy.random.seed(0)
            >>> generate_random_bound_points((100, 100), 5)
            array([[49, 52]])

            >>> numpy.random.seed(0)
            >>> generate_random_bound_points((100, 100), (5, 5))
            array([[49, 52],
                   [69, 72]])

            >>> numpy.random.seed(0)
            >>> generate_random_bound_points((100, 100, 100), (5,))
            array([[49, 52, 69]])

            >>> numpy.random.seed(0)
            >>> generate_random_bound_points((100, 100, 100), (5, 5))
            array([[49, 52, 69],
                   [72, 72, 14]])
    """

    # Convert to arrays
    space = numpy.array(space)
    radii = numpy.array(radii)

    # Add a singleton dimension if there is only one of each.
    if radii.ndim == 0:
        radii = radii[None]

    # Validate the dimensions
    assert (space.ndim == 1)
    assert (radii.ndim == 1)

    # Determine the space each centroid can be within
    bound_space = numpy.zeros(radii.shape + space.shape + (2,), dtype=int)
    bound_space[..., 0] = nanshe.util.xnumpy.expand_view(radii, space.shape)
    bound_space[..., 1] = nanshe.util.xnumpy.expand_view(space, reps_before=radii.shape) - \
                          nanshe.util.xnumpy.expand_view(radii, space.shape)

    # Generate a random point for each radius.
    points = numpy.zeros(radii.shape + space.shape, dtype=int)
    for i in nanshe.util.iters.irange(len(radii)):
        for j in nanshe.util.iters.irange(len(space)):
            points[i][j] = numpy.random.randint(
                bound_space[i][j][0], bound_space[i][j][1]
            )

    return(points)
