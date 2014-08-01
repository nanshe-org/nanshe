__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 01, 2014 14:55:57 EDT$"


import itertools

import numpy

import nanshe.expanded_numpy


def generate_hypersphere_masks(space, centers, radii, include_boundary = False):
    """
        Generate a stack of masks (first index indicates which mask); where, each contains a hypersphere constructed
        using a center and radius provided.

        Args:
            space(tuple of ints):                The size of the mask.

            centers(list of tuples of numbers):  List of centers with one per hypersphere.

            radii(list of numbers):              List of radii with one per hypersphere.

            include_boundary(bool):              Whether the mask should contain the boundary of the hypersphere or not.

        Returns:
            numpy.ndarray:                       A stack of masks (first index indicates which mask) with a filled
                                                 hypersphere using a center and radius for each mask.

        Examples:
            >>> generate_hypersphere_masks((3, 3), (1, 1), 1.25)
            array([[[False,  True, False],
                    [ True,  True,  True],
                    [False,  True, False]]], dtype=bool)

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
    """

    # Convert to arrays
    space = numpy.array(space)
    centers = numpy.array(centers)
    radii = numpy.array(radii)

    # Add a singleton dimension if there is only one of each.
    if centers.ndim == 1:
        centers = centers[None]

    if radii.ndim == 0:
        radii = radii[None]

    # Validate the dimensions
    assert(space.ndim == 1)
    assert(centers.ndim == 2)
    assert(radii.ndim == 1)

    # Validate the shapes
    assert(space.shape == centers.shape[1:])
    assert(radii.shape == centers.shape[:1])

    # Create a hypersphere mask using a center and a radius.
    hypersphere_mask = numpy.zeros(radii.shape + tuple(space.tolist()), dtype = bool)
    for i, (each_center, each_radius) in enumerate(itertools.izip(centers, radii)):
        space_index = numpy.indices(space)

        each_point_offset = (space_index - nanshe.expanded_numpy.expand_view(each_center, tuple(space.tolist())))

        each_point_offset_sqd_sum = (each_point_offset**2).sum(axis = 0)

        each_point_offset_dist = each_point_offset_sqd_sum.astype(float)**.5

        if include_boundary:
            hypersphere_mask[i] = (each_point_offset_dist <= each_radius)
        else:
            hypersphere_mask[i] = (each_point_offset_dist < each_radius)

    return(hypersphere_mask)
