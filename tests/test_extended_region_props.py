__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 05, 2014 14:02:24 EDT$"


import itertools

import numpy

import nanshe.expanded_numpy

import nanshe.advanced_image_processing

import synthetic_data


class TestAdvancedImageProcessing(object):
    def test_ExtendedRegionProps_1(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype = float)

        g = synthetic_data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        m = (g > 0.00065)
        g *= m

        e = nanshe.advanced_image_processing.ExtendedRegionProps(g.max(axis = 0),
                nanshe.expanded_numpy.enumerate_masks(m).max(axis = 0)
        )

        assert(len(e.props) == len(p))

        assert((e.count["label"] == numpy.arange(1, len(m) + 1)).all())

        assert((e.count["count"] == 1).all())

        assert((e.label_image == nanshe.expanded_numpy.enumerate_masks(m).max(axis = 0)).all())

        assert((e.intensity_image == g.max(axis = 0)).all())

        assert((e.image_mask == m.max(axis = 0)).all())

        assert((e.props == nanshe.advanced_image_processing.extended_region_local_maxima_properties(g.max(axis = 0),
            nanshe.expanded_numpy.enumerate_masks(m).max(axis = 0), properties = ["label", "centroid"])).all()
        )

        assert(all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))]))

        assert(all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))]))

        assert((e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all())

        assert(all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))]))

