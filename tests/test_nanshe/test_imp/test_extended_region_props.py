__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 05, 2014 14:02:24 EDT$"


import nose
import nose.plugins
import nose.plugins.attrib

import itertools

import numpy

import nanshe.util.xnumpy

import nanshe.imp.segment

import nanshe.syn.data


class TestSegment(object):
    def test_ExtendedRegionProps_1(self):
        p = numpy.array([[27, 51],
                         [66, 85],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        m = (g > 0.00065)
        g *= m

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == 1).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"])
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all()

    def test_ExtendedRegionProps_2(self):
        p = numpy.array([[27, 51],
                         [32, 53],
                         [77, 45]])

        space = numpy.array((100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        g = numpy.array([g[0] + g[1], g[2]])
        m = (g > 0.00065)
        g *= m

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        removal_mask = numpy.zeros(radii.shape, dtype=bool)

        e.remove_prop_mask(removal_mask)

        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == numpy.array([2, 1])).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"])
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.array([1, 1, 2])).all()

    def test_ExtendedRegionProps_3(self):
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

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        removal_mask = numpy.zeros(radii.shape, dtype=bool)

        e.remove_prop_mask(removal_mask)

        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == 1).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"])
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all()

    def test_ExtendedRegionProps_4(self):
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

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        removal_mask = numpy.zeros(radii.shape, dtype=bool)
        removal_mask[0] = True

        m = m[~removal_mask].copy()
        p = p[~removal_mask].copy()
        radii = radii[~removal_mask].copy()

        g = g * m.max(axis=0)

        e.remove_prop_mask(removal_mask)

        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == 1).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"])
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all()

    def test_ExtendedRegionProps_5(self):
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

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        removal_mask = numpy.zeros(radii.shape, dtype=bool)
        removal_mask[0] = True

        m = m[~removal_mask].copy()
        p = p[~removal_mask].copy()
        radii = radii[~removal_mask].copy()

        g = g * m.max(axis=0)

        e.remove_prop_indices(*removal_mask.nonzero()[0])

        print len(e.props)
        print len(p)


        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == 1).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"])
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all()

    @nose.plugins.attrib.attr("3D")
    def test_ExtendedRegionProps_6(self):
        p = numpy.array([[27, 51, 78],
                         [66, 85, 56],
                         [77, 45, 24]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        m = (g > 0.000016)
        g *= m

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == 1).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"]
            )
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all()

    @nose.plugins.attrib.attr("3D")
    def test_ExtendedRegionProps_7(self):
        p = numpy.array([[27, 51, 78],
                         [66, 85, 56],
                         [77, 45, 24]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(space, p, radii/3.0, magnitudes/3)
        g = numpy.array([g[0] + g[1], g[2]])
        m = (g > 0.00065)
        g *= m

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        removal_mask = numpy.zeros(radii.shape, dtype=bool)

        e.remove_prop_mask(removal_mask)

        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == numpy.array([2, 1])).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"]
            )
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.array([1, 1, 2])).all()

    @nose.plugins.attrib.attr("3D")
    def test_ExtendedRegionProps_8(self):
        p = numpy.array([[27, 51, 78],
                         [66, 85, 56],
                         [77, 45, 24]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = (g > 0.000016)
        g *= m

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        removal_mask = numpy.zeros(radii.shape, dtype=bool)

        e.remove_prop_mask(removal_mask)

        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == 1).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"]
            )
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all()

    @nose.plugins.attrib.attr("3D")
    def test_ExtendedRegionProps_9(self):
        p = numpy.array([[27, 51, 78],
                         [66, 85, 56],
                         [77, 45, 24]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = (g > 0.000016)
        g *= m

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        removal_mask = numpy.zeros(radii.shape, dtype=bool)
        removal_mask[0] = True

        m = m[~removal_mask].copy()
        p = p[~removal_mask].copy()
        radii = radii[~removal_mask].copy()

        g = g * m.max(axis=0)

        e.remove_prop_mask(removal_mask)

        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == 1).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"]
            )
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all()

    @nose.plugins.attrib.attr("3D")
    def test_ExtendedRegionProps_10(self):
        p = numpy.array([[27, 51, 78],
                         [66, 85, 56],
                         [77, 45, 24]])

        space = numpy.array((100, 100, 100))
        radii = numpy.array((5, 6, 7))
        magnitudes = numpy.array((1, 1, 1), dtype=float)

        g = nanshe.syn.data.generate_gaussian_images(
            space, p, radii/3.0, magnitudes/3
        )
        m = (g > 0.000016)
        g *= m

        e = nanshe.imp.segment.ExtendedRegionProps(
            g.max(axis=0),
            nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)
        )

        removal_mask = numpy.zeros(radii.shape, dtype=bool)
        removal_mask[0] = True

        m = m[~removal_mask].copy()
        p = p[~removal_mask].copy()
        radii = radii[~removal_mask].copy()

        g = g * m.max(axis=0)

        e.remove_prop_indices(*removal_mask.nonzero()[0])

        print len(e.props)
        print len(p)


        assert (len(e.props) == len(p))

        assert (e.count["label"] == numpy.arange(1, len(m) + 1)).all()

        assert (e.count["count"] == 1).all()

        assert (e.label_image == nanshe.util.xnumpy.enumerate_masks(m).max(axis=0)).all()

        assert (e.intensity_image == g.max(axis=0)).all()

        assert (e.image_mask == m.max(axis=0)).all()

        assert (
            e.props == nanshe.imp.segment.extended_region_local_maxima_properties(
                g.max(axis=0),
                nanshe.util.xnumpy.enumerate_masks(m).max(axis=0), properties=["label", "centroid"]
            )
        ).all()

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_index_array(), tuple(p.T))])

        assert all([(_1 == _2).all() for _1, _2 in itertools.izip(e.get_local_max_label_image().nonzero(), tuple(p.T))])

        assert (e.get_local_max_label_image()[e.get_local_max_label_image().nonzero()] == numpy.arange(1, len(m) + 1)).all()
