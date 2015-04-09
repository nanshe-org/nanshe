__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 08, 2014 08:08:39 EDT$"


import nose
import nose.plugins
import nose.plugins.attrib

import json
import operator
import os
import shutil
import tempfile

import h5py
import numpy

import nanshe.util.xnumpy
import nanshe.io.hdf5.record

import nanshe.syn.data

import nanshe.learner


class TestLearner(object):
    def setup(self):
        self.config_a_block = {
            "debug" : True,
            "generate_neurons" : {
                "postprocess_data" : {
                    "wavelet_denoising" : {
                        "remove_low_intensity_local_maxima" : {
                            "percentage_pixels_below_max" : 0
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
                            "min_local_max_distance" : 100.0
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
                },
                "run_stage" : "all",
                "preprocess_data" : {
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
                        "half_window_size" : 1,
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
                },
                "generate_dictionary" : {
                    "spams.trainDL" : {
                        "gamma2" : 0,
                        "gamma1" : 0,
                        "numThreads" : 1,
                        "K" : 10,
                        "iter" : 100,
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
            }
        }

        self.config_a_block_3D = {
            "debug" : True,
            "generate_neurons" : {
                "postprocess_data" : {
                    "wavelet_denoising" : {
                        "remove_low_intensity_local_maxima" : {
                            "percentage_pixels_below_max" : 0
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
                                "max" : 15000,
                                "min" : 150
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
                    },
                    "merge_neuron_sets" : {
                        "alignment_min_threshold" : 0.6,
                        "fuse_neurons" : {
                            "fraction_mean_neuron_max_threshold" : 0.01
                        },
                        "overlap_min_threshold" : 0.6
                    }
                },
                "run_stage" : "all",
                "preprocess_data" : {
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
                        "half_window_size" : 1,
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
                },
                "generate_dictionary" : {
                    "spams.trainDL" : {
                        "gamma2" : 0,
                        "gamma1" : 0,
                        "numThreads" : 1,
                        "K" : 10,
                        "iter" : 100,
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
            }
        }

        self.config_blocks = {
            "generate_neurons_blocks" : {
                "num_processes" : 4,
                "block_shape" : [10000, -1, -1],
                "num_blocks" : [-1, 5, 5],
                "half_border_shape" : [0, 5, 5],
                "half_window_shape" : [50, 20, 20],

                "debug" : True,

                "generate_neurons" : {
                    "__comment__run_stage" : "Where to run until either preprocessing, dictionary, or postprocessing. If resume, is true then it will delete the previous results at this stage. By default (all can be set explicitly to null string)runs all the way through.",

                    "run_stage" : "all",

                    "__comment__normalize_data" : "These are arguments that will be passed to normalize_data.",

                    "preprocess_data" : {
                        "remove_zeroed_lines" : {
                            "erosion_shape" : [21, 1],
                            "dilation_shape" : [1, 3]
                        },

                        "extract_f0" : {
                            "bias" : 100,

                            "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                            "temporal_smoothing_gaussian_filter_window_size" : 5.0,

                            "half_window_size" : 1,          "__comment__window_size" : "In number of frames",
                            "which_quantile" : 0.5,           "__comment__which_quantile" : "Must be a single value (i.e. 0.5) to extract.",

                            "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                            "spatial_smoothing_gaussian_filter_window_size" : 5.0
                        },

                        "wavelet.transform" : {
                            "scale" : [3, 4, 4]
                        },

                        "normalize_data" : {
                            "renormalized_images": {
                                "ord" : 2
                            }
                        }
                    },


                    "__comment__generate_dictionary" : "These are arguments that will be passed to generate_dictionary.",

                    "generate_dictionary" : {
                        "spams.trainDL" : {
                            "K" : 10,
                            "gamma2": 0,
                            "gamma1": 0,
                            "numThreads": 1,
                            "batchsize": 256,
                            "iter": 100,
                            "lambda1": 0.2,
                            "posD": True,
                            "clean": True,
                            "modeD": 0,
                            "posAlpha": True,
                            "mode": 2,
                            "lambda2": 0
                        }
                    },


                    "postprocess_data" : {

                        "__comment__wavelet_denoising" : "These are arguments that will be passed to wavelet_",

                        "wavelet_denoising" : {

                            "estimate_noise" : {
                                "significance_threshold" : 3.0
                            },

                            "significant_mask" : {
                                "noise_threshold" : 3.0
                            },

                            "wavelet.transform" : {
                                "scale" : 4
                            },

                            "accepted_region_shape_constraints" : {
                                "major_axis_length" : {
                                    "min" : 0.0,
                                    "max" : 25.0
                                }
                            },

                            "remove_low_intensity_local_maxima" : {
                                "percentage_pixels_below_max" : 0

                            },

                            "__comment__min_local_max_distance" : 6.0,
                            "remove_too_close_local_maxima" : {
                                "min_local_max_distance"  : 100.0
                            },

                            "use_watershed" : True,

                            "__comment__accepted_neuron_shape_constraints_area_max" : 250,
                            "accepted_neuron_shape_constraints" : {
                                "area" : {
                                    "min" : 30,
                                    "max" : 600
                                },

                                "eccentricity" : {
                                    "min" : 0.0,
                                    "max" : 0.9
                                }
                            }
                        },


                        "merge_neuron_sets" : {
                            "alignment_min_threshold" : 0.6,
                            "overlap_min_threshold" : 0.6,

                            "fuse_neurons" : {
                                "fraction_mean_neuron_max_threshold" : 0.01
                            }
                        }
                    }
                }
            }
        }

        self.config_blocks_3D = {
            "generate_neurons_blocks" : {
                "num_processes" : 4,
                "block_shape" : [10000, -1, -1, -1],
                "num_blocks" : [-1, 2, 2, 2],
                "half_border_shape" : [0, 5, 5, 5],
                "half_window_shape" : [50, 20, 20, 20],

                "debug" : True,

                "generate_neurons" : {
                    "postprocess_data" : {
                        "wavelet_denoising" : {
                            "remove_low_intensity_local_maxima" : {
                                "percentage_pixels_below_max" : 0
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
                                    "max" : 15000,
                                    "min" : 150
                                }
                            },
                            "estimate_noise" : {
                                "significance_threshold" : 3.0
                            },
                            "significant_mask" : {
                                "noise_threshold" : 2.0
                            },
                            "remove_too_close_local_maxima" : {
                                "min_local_max_distance" : 100.0
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
                    },
                    "run_stage" : "all",
                    "preprocess_data" : {
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
                            "half_window_size" : 1,
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
                    },
                    "generate_dictionary" : {
                        "spams.trainDL" : {
                            "gamma2" : 0,
                            "gamma1" : 0,
                            "numThreads" : 1,
                            "K" : 10,
                            "iter" : 100,
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
                }
            }
        }

        self.config_blocks_drmaa = {
            "generate_neurons_blocks" : {
                "num_processes" : 4,
                "block_shape" : [10000, -1, -1],
                "num_blocks" : [-1, 5, 5],
                "half_border_shape" : [0, 5, 5],
                "half_window_shape" : [50, 20, 20],

                "use_drmaa" : True,
                "num_drmaa_cores" : 1,

                "debug" : True,

                "generate_neurons" : {
                    "__comment__run_stage" : "Where to run until either preprocessing, dictionary, or postprocessing. If resume, is true then it will delete the previous results at this stage. By default (all can be set explicitly to null string)runs all the way through.",

                    "run_stage" : "all",

                    "__comment__normalize_data" : "These are arguments that will be passed to normalize_data.",

                    "preprocess_data" : {
                        "remove_zeroed_lines" : {
                            "erosion_shape" : [21, 1],
                            "dilation_shape" : [1, 3]
                        },

                        "extract_f0" : {
                            "bias" : 100,

                            "temporal_smoothing_gaussian_filter_stdev" : 5.0,
                            "temporal_smoothing_gaussian_filter_window_size" : 5.0,

                            "half_window_size" : 1,          "__comment__window_size" : "In number of frames",
                            "which_quantile" : 0.5,           "__comment__which_quantile" : "Must be a single value (i.e. 0.5) to extract.",

                            "spatial_smoothing_gaussian_filter_stdev" : 5.0,
                            "spatial_smoothing_gaussian_filter_window_size" : 5.0
                        },

                        "wavelet.transform" : {
                            "scale" : [3, 4, 4]
                        },

                        "normalize_data" : {
                            "renormalized_images": {
                                "ord" : 2
                            }
                        }
                    },


                    "__comment__generate_dictionary" : "These are arguments that will be passed to generate_dictionary.",

                    "generate_dictionary" : {
                        "spams.trainDL" : {
                            "K" : 10,
                            "gamma2": 0,
                            "gamma1": 0,
                            "numThreads": 1,
                            "batchsize": 256,
                            "iter": 100,
                            "lambda1": 0.2,
                            "posD": True,
                            "clean": True,
                            "modeD": 0,
                            "posAlpha": True,
                            "mode": 2,
                            "lambda2": 0
                        }
                    },


                    "postprocess_data" : {

                        "__comment__wavelet_denoising" : "These are arguments that will be passed to wavelet_",

                        "wavelet_denoising" : {

                            "estimate_noise" : {
                                "significance_threshold" : 3.0
                            },

                            "significant_mask" : {
                                "noise_threshold" : 3.0
                            },

                            "wavelet.transform" : {
                                "scale" : 4
                            },

                            "accepted_region_shape_constraints" : {
                                "major_axis_length" : {
                                    "min" : 0.0,
                                    "max" : 25.0
                                }
                            },

                            "remove_low_intensity_local_maxima" : {
                                "percentage_pixels_below_max" : 0

                            },

                            "__comment__min_local_max_distance" : 6.0,
                            "remove_too_close_local_maxima" : {
                                "min_local_max_distance"  : 100.0
                            },

                            "use_watershed" : True,

                            "__comment__accepted_neuron_shape_constraints_area_max" : 250,
                            "accepted_neuron_shape_constraints" : {
                                "area" : {
                                    "min" : 30,
                                    "max" : 600
                                },

                                "eccentricity" : {
                                    "min" : 0.0,
                                    "max" : 0.9
                                }
                            }
                        },


                        "merge_neuron_sets" : {
                            "alignment_min_threshold" : 0.6,
                            "overlap_min_threshold" : 0.6,

                            "fuse_neurons" : {
                                "fraction_mean_neuron_max_threshold" : 0.01
                            }
                        }
                    }
                }
            }
        }

        self.config_blocks_3D_drmaa = {
            "generate_neurons_blocks" : {
                "num_processes" : 4,
                "block_shape" : [10000, -1, -1, -1],
                "num_blocks" : [-1, 2, 2, 2],
                "half_border_shape" : [0, 5, 5, 5],
                "half_window_shape" : [50, 20, 20, 20],

                "use_drmaa" : True,
                "num_drmaa_cores" : 1,

                "debug" : True,

                "generate_neurons" : {
                    "postprocess_data" : {
                        "wavelet_denoising" : {
                            "remove_low_intensity_local_maxima" : {
                                "percentage_pixels_below_max" : 0
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
                                    "max" : 15000,
                                    "min" : 150
                                }
                            },
                            "estimate_noise" : {
                                "significance_threshold" : 3.0
                            },
                            "significant_mask" : {
                                "noise_threshold" : 2.0
                            },
                            "remove_too_close_local_maxima" : {
                                "min_local_max_distance" : 100.0
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
                    },
                    "run_stage" : "all",
                    "preprocess_data" : {
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
                            "half_window_size" : 1,
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
                    },
                    "generate_dictionary" : {
                        "spams.trainDL" : {
                            "gamma2" : 0,
                            "gamma1" : 0,
                            "numThreads" : 1,
                            "K" : 10,
                            "iter" : 100,
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
                }
            }
        }

        has_drmaa = True
        try:
            import drmaa
        except ImportError:
            has_drmaa = False
        except RuntimeError:
            has_drmaa = False

        if has_drmaa:
            tempfile.tempdir = os.path.join(os.path.expanduser("~"), "tmp")

        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = os.path.abspath(self.temp_dir)

        self.hdf5_input_filename = os.path.join(self.temp_dir, "input.h5")
        self.hdf5_input_filepath = self.hdf5_input_filename + "/" + "images"
        self.hdf5_input_3D_filename = os.path.join(self.temp_dir, "input_3D.h5")
        self.hdf5_input_3D_filepath = self.hdf5_input_3D_filename + "/" + "images"
        self.hdf5_output_filename = os.path.join(self.temp_dir, "output.h5")
        self.hdf5_output_filepath = self.hdf5_output_filename + "/"
        self.hdf5_output_3D_filename = os.path.join(self.temp_dir, "output_3D.h5")
        self.hdf5_output_3D_filepath = self.hdf5_output_3D_filename + "/"

        self.config_a_block_filename = os.path.join(self.temp_dir, "config_a_block.json")
        self.config_a_block_3D_filename = os.path.join(self.temp_dir, "config_a_block_3D.json")
        self.config_blocks_filename = os.path.join(self.temp_dir, "config_blocks.json")
        self.config_blocks_3D_filename = os.path.join(self.temp_dir, "config_blocks_3D.json")
        self.config_blocks_drmaa_filename = os.path.join(self.temp_dir, "config_blocks_drmaa.json")
        self.config_blocks_3D_drmaa_filename = os.path.join(self.temp_dir, "config_blocks_3D_drmaa.json")

        self.space = numpy.array([110, 110])
        self.radii = numpy.array([6, 6, 6, 6, 7, 6])
        self.magnitudes = numpy.array([15, 16, 15, 17, 16, 16])
        self.points = numpy.array([[30, 24],
                                   [59, 65],
                                   [21, 65],
                                   [80, 78],
                                   [72, 16],
                                   [45, 32]])

        self.bases_indices = [[1, 3, 4], [0, 2], [5]]
        self.linspace_length = 25

        self.masks = nanshe.syn.data.generate_hypersphere_masks(self.space, self.points, self.radii)
        self.images = nanshe.syn.data.generate_gaussian_images(self.space, self.points, self.radii/3.0, self.magnitudes) * self.masks

        self.bases_masks = numpy.zeros((len(self.bases_indices),) + self.masks.shape[1:], dtype=self.masks.dtype)
        self.bases_images = numpy.zeros((len(self.bases_indices),) + self.images.shape[1:], dtype=self.images.dtype)

        for i, each_basis_indices in enumerate(self.bases_indices):
            self.bases_masks[i] = self.masks[list(each_basis_indices)].max(axis=0)
            self.bases_images[i] = self.images[list(each_basis_indices)].max(axis=0)

        self.image_stack = None
        ramp = numpy.concatenate([numpy.linspace(0, 1, self.linspace_length), numpy.linspace(1, 0, self.linspace_length)])

        self.image_stack = numpy.zeros((self.bases_images.shape[0] * len(ramp),) + self.bases_images.shape[1:],
                                       dtype=self.bases_images.dtype)
        for i in xrange(len(self.bases_images)):
            image_stack_slice = slice(i * len(ramp), (i+1) * len(ramp), 1)

            self.image_stack[image_stack_slice] = nanshe.util.xnumpy.all_permutations_operation(
                operator.mul,
                ramp,
                self.bases_images[i]
            )

        self.space3 = numpy.array([60, 60, 60])
        self.radii3 = numpy.array([4, 3, 3, 3, 4, 3])
        self.magnitudes3 = numpy.array([8, 8, 8, 8, 8, 8])
        self.points3 = numpy.array([[15, 16, 17],
                                    [42, 21, 23],
                                    [45, 32, 34],
                                    [41, 41, 42],
                                    [36, 15, 41],
                                    [22, 16, 34]])

        self.masks3 = nanshe.syn.data.generate_hypersphere_masks(self.space3, self.points3, self.radii3)
        self.images3 = nanshe.syn.data.generate_gaussian_images(self.space3, self.points3, self.radii3/3.0, self.magnitudes3) * self.masks3

        self.bases_masks3 = numpy.zeros((len(self.bases_indices),) + self.masks3.shape[1:], dtype=self.masks3.dtype)
        self.bases_images3 = numpy.zeros((len(self.bases_indices),) + self.images3.shape[1:], dtype=self.images3.dtype)

        for i, each_basis_indices in enumerate(self.bases_indices):
            self.bases_masks3[i] = self.masks3[list(each_basis_indices)].max(axis=0)
            self.bases_images3[i] = self.images3[list(each_basis_indices)].max(axis=0)

        self.image_stack3 = None
        ramp = numpy.concatenate([numpy.linspace(0, 1, self.linspace_length), numpy.linspace(1, 0, self.linspace_length)])

        self.image_stack3 = numpy.zeros(
            (self.bases_images3.shape[0] * len(ramp),) + self.bases_images3.shape[1:],
            dtype=self.bases_images3.dtype
        )
        for i in xrange(len(self.bases_images3)):
            image_stack_slice3 = slice(i * len(ramp), (i+1) * len(ramp), 1)

            self.image_stack3[image_stack_slice3] = nanshe.util.xnumpy.all_permutations_operation(
                operator.mul,
                ramp,
                self.bases_images3[i]
            )

        with h5py.File(self.hdf5_input_filename, "w") as fid:
            fid["images"] = self.image_stack

        with h5py.File(self.hdf5_input_3D_filename, "w") as fid:
            fid["images"] = self.image_stack3

        with h5py.File(self.hdf5_output_filename, "w") as fid:
            pass

        with h5py.File(self.hdf5_output_3D_filename, "w") as fid:
            pass

        with open(self.config_a_block_filename, "w") as fid:
            json.dump(self.config_a_block, fid)
            fid.write("\n")

        with open(self.config_a_block_3D_filename, "w") as fid:
            json.dump(self.config_a_block_3D, fid)
            fid.write("\n")

        with open(self.config_blocks_filename, "w") as fid:
            json.dump(self.config_blocks, fid)
            fid.write("\n")

        with open(self.config_blocks_3D_filename, "w") as fid:
            json.dump(self.config_blocks_3D, fid)
            fid.write("\n")

        with open(self.config_blocks_drmaa_filename, "w") as fid:
            json.dump(self.config_blocks_drmaa, fid)
            fid.write("\n")

        with open(self.config_blocks_3D_drmaa_filename, "w") as fid:
            json.dump(self.config_blocks_3D_drmaa, fid)
            fid.write("\n")

    def test_main_1(self):
        executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

        argv = (executable, self.config_a_block_filename, self.hdf5_input_filepath, self.hdf5_output_filepath,)

        assert (0 == nanshe.learner.main(*argv))

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    def test_main_2(self):
        executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

        argv = (executable, self.config_blocks_filename, self.hdf5_input_filepath, self.hdf5_output_filepath,)

        assert (0 == nanshe.learner.main(*argv))

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("DRMAA")
    def test_main_3(self):
        # Attempt to import drmaa.
        # If it fails to import, either the user has no intent in using it or forgot to install it.
        # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
        try:
            import drmaa
        except ImportError:
            # python-drmaa is not installed.
            raise nose.SkipTest("Skipping test test_main_3. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
        except RuntimeError:
            # The drmaa library was not specified, but python-drmaa is installed.
            raise nose.SkipTest("Skipping test test_main_3. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

        executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

        argv = (executable, self.config_blocks_drmaa_filename, self.hdf5_input_filepath, self.hdf5_output_filepath,)

        assert (0 == nanshe.learner.main(*argv))

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_main_4(self):
        executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

        argv = (executable, self.config_a_block_3D_filename, self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath,)

        assert (0 == nanshe.learner.main(*argv))

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_main_5(self):
        executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

        argv = (executable, self.config_blocks_3D_filename, self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath,)

        assert (0 == nanshe.learner.main(*argv))

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D", "DRMAA")
    def test_main_6(self):
        # Attempt to import drmaa.
        # If it fails to import, either the user has no intent in using it or forgot to install it.
        # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
        try:
            import drmaa
        except ImportError:
            # python-drmaa is not installed.
            raise nose.SkipTest("Skipping test test_main_3. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
        except RuntimeError:
            # The drmaa library was not specified, but python-drmaa is installed.
            raise nose.SkipTest("Skipping test test_main_3. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

        executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

        argv = (executable, self.config_blocks_3D_drmaa_filename, self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath,)

        assert (0 == nanshe.learner.main(*argv))

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    def test_generate_neurons_io_handler_1(self):
        nanshe.learner.generate_neurons_io_handler(self.hdf5_input_filepath, self.hdf5_output_filepath, self.config_a_block_filename)

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    def test_generate_neurons_io_handler_2(self):
        nanshe.learner.generate_neurons_io_handler(self.hdf5_input_filepath, self.hdf5_output_filepath, self.config_blocks_filename)

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("DRMAA")
    def test_generate_neurons_io_handler_3(self):
        # Attempt to import drmaa.
        # If it fails to import, either the user has no intent in using it or forgot to install it.
        # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
        try:
            import drmaa
        except ImportError:
            # python-drmaa is not installed.
            raise nose.SkipTest("Skipping test test_generate_neurons_io_handler_3. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
        except RuntimeError:
            # The drmaa library was not specified, but python-drmaa is installed.
            raise nose.SkipTest("Skipping test test_generate_neurons_io_handler_3. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

        nanshe.learner.generate_neurons_io_handler(self.hdf5_input_filepath, self.hdf5_output_filepath, self.config_blocks_drmaa_filename)

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_generate_neurons_io_handler_4(self):
        nanshe.learner.generate_neurons_io_handler(self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath, self.config_a_block_3D_filename)

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_generate_neurons_io_handler_5(self):
        nanshe.learner.generate_neurons_io_handler(self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath, self.config_blocks_3D_filename)

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D", "DRMAA")
    def test_generate_neurons_io_handler_6(self):
        # Attempt to import drmaa.
        # If it fails to import, either the user has no intent in using it or forgot to install it.
        # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
        try:
            import drmaa
        except ImportError:
            # python-drmaa is not installed.
            raise nose.SkipTest("Skipping test test_main_3. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
        except RuntimeError:
            # The drmaa library was not specified, but python-drmaa is installed.
            raise nose.SkipTest("Skipping test test_main_3. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

        nanshe.learner.generate_neurons_io_handler(self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath, self.config_blocks_3D_drmaa_filename)

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    def test_generate_neurons_a_block_1(self):
        nanshe.learner.generate_neurons_a_block(self.hdf5_input_filepath, self.hdf5_output_filepath, **self.config_a_block)

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_generate_neurons_a_block_2(self):
        nanshe.learner.generate_neurons_a_block(self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath, **self.config_a_block_3D)

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    def test_generate_neurons_blocks_1(self):
        nanshe.learner.generate_neurons_blocks(self.hdf5_input_filepath, self.hdf5_output_filepath, **self.config_blocks["generate_neurons_blocks"])

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("DRMAA")
    def test_generate_neurons_blocks_2(self):
        # Attempt to import drmaa.
        # If it fails to import, either the user has no intent in using it or forgot to install it.
        # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
        try:
            import drmaa
        except ImportError:
            # python-drmaa is not installed.
            raise nose.SkipTest("Skipping test test_generate_neurons_blocks_2. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
        except RuntimeError:
            # The drmaa library was not specified, but python-drmaa is installed.
            raise nose.SkipTest("Skipping test test_generate_neurons_blocks_2. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

        nanshe.learner.generate_neurons_blocks(self.hdf5_input_filepath, self.hdf5_output_filepath, **self.config_blocks_drmaa["generate_neurons_blocks"])

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_generate_neurons_blocks_3(self):
        nanshe.learner.generate_neurons_blocks(self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath, **self.config_blocks_3D["generate_neurons_blocks"])

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D", "DRMAA")
    def test_generate_neurons_blocks_4(self):
        # Attempt to import drmaa.
        # If it fails to import, either the user has no intent in using it or forgot to install it.
        # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
        try:
            import drmaa
        except ImportError:
            # python-drmaa is not installed.
            raise nose.SkipTest("Skipping test test_generate_neurons_blocks_2. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
        except RuntimeError:
            # The drmaa library was not specified, but python-drmaa is installed.
            raise nose.SkipTest("Skipping test test_generate_neurons_blocks_2. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

        nanshe.learner.generate_neurons_blocks(self.hdf5_input_3D_filepath, self.hdf5_output_3D_filepath, **self.config_blocks_3D_drmaa["generate_neurons_blocks"])

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    def test_generate_neurons_1(self):
        with h5py.File(self.hdf5_output_filename, "a") as output_file_handle:
            output_group = output_file_handle["/"]

            # Get a debug logger for the HDF5 file (if needed)
            array_debug_recorder = nanshe.io.hdf5.record.generate_HDF5_array_recorder(
                output_group,
                group_name="debug",
                enable=self.config_a_block["debug"],
                overwrite_group=False,
                recorder_constructor=nanshe.io.hdf5.record.HDF5EnumeratedArrayRecorder
            )

            # Saves intermediate result to make resuming easier
            resume_logger = nanshe.io.hdf5.record.generate_HDF5_array_recorder(
                output_group,
                recorder_constructor=nanshe.io.hdf5.record.HDF5ArrayRecorder,
                overwrite=True
            )

            nanshe.learner.generate_neurons.resume_logger = resume_logger
            nanshe.learner.generate_neurons.recorders.array_debug_recorder = array_debug_recorder
            nanshe.learner.generate_neurons(self.image_stack, **self.config_a_block["generate_neurons"])

        assert os.path.exists(self.hdf5_output_filename)

        with h5py.File(self.hdf5_output_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_generate_neurons_2(self):
        with h5py.File(self.hdf5_output_3D_filename, "a") as output_file_handle:
            output_group = output_file_handle["/"]

            # Get a debug logger for the HDF5 file (if needed)
            array_debug_recorder = nanshe.io.hdf5.record.generate_HDF5_array_recorder(
                output_group,
                group_name="debug",
                enable=self.config_a_block["debug"],
                overwrite_group=False,
                recorder_constructor=nanshe.io.hdf5.record.HDF5EnumeratedArrayRecorder
            )

            # Saves intermediate result to make resuming easier
            resume_logger = nanshe.io.hdf5.record.generate_HDF5_array_recorder(
                output_group,
                recorder_constructor=nanshe.io.hdf5.record.HDF5ArrayRecorder,
                overwrite=True
            )

            nanshe.learner.generate_neurons.resume_logger = resume_logger
            nanshe.learner.generate_neurons.recorders.array_debug_recorder = array_debug_recorder
            nanshe.learner.generate_neurons(self.image_stack3, **self.config_a_block_3D["generate_neurons"])

        assert os.path.exists(self.hdf5_output_3D_filename)

        with h5py.File(self.hdf5_output_3D_filename, "r") as fid:
            assert ("neurons" in fid)

            neurons = fid["neurons"].value

        assert (len(self.points3) == len(neurons))

        neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))
        neuron_max_points = numpy.array(neuron_maxes.max(axis=0).nonzero()).T.copy()

        matched = dict()
        unmatched_points = numpy.arange(len(self.points3))
        for i in xrange(len(neuron_max_points)):
            new_unmatched_points = []
            for j in unmatched_points:
                if not (neuron_max_points[i] == self.points3[j]).all():
                    new_unmatched_points.append(j)
                else:
                    matched[i] = j

            unmatched_points = new_unmatched_points

        assert (len(unmatched_points) == 0)

    def teardown(self):
        try:
            os.remove(self.config_a_block_filename)
        except OSError:
            pass
        self.config_a_block_filename = ""

        try:
            os.remove(self.config_a_block_3D_filename)
        except OSError:
            pass
        self.config_a_block_3D_filename = ""

        try:
            os.remove(self.config_blocks_filename)
        except OSError:
            pass
        self.config_blocks_filename = ""

        try:
            os.remove(self.config_blocks_drmaa_filename)
        except OSError:
            pass
        self.config_blocks_drmaa_filename = ""

        try:
            os.remove(self.config_blocks_3D_filename)
        except OSError:
            pass
        self.config_blocks_3D_filename = ""

        try:
            os.remove(self.config_blocks_3D_drmaa_filename)
        except OSError:
            pass
        self.config_blocks_3D_drmaa_filename = ""

        try:
            os.remove(self.hdf5_input_filename)
        except OSError:
            pass
        self.hdf5_input_filename = ""

        try:
            os.remove(self.hdf5_input_3D_filename)
        except OSError:
            pass
        self.hdf5_input_3D_filename = ""

        try:
            os.remove(self.hdf5_output_filename)
        except OSError:
            pass
        self.hdf5_output_filename = ""

        try:
            os.remove(self.hdf5_output_3D_filename)
        except OSError:
            pass
        self.hdf5_output_3D_filename = ""

        shutil.rmtree(self.temp_dir)
        self.temp_dir = ""

        tempfile.tempdir = None
