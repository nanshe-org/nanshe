__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 30, 2015 23:17:09 EDT$"


from glob import glob
import sys

from setuptools import setup, find_packages
import versioneer


versioneer.VCS = "git"
versioneer.versionfile_source = "nanshe/_version.py"
versioneer.versionfile_build = None
versioneer.tag_prefix = "v"
versioneer.parentdir_prefix = "nanshe-"

build_requires = []
install_requires = []
tests_require = []
if len(sys.argv) == 1:
    pass
elif ("--help" in sys.argv) or ("-h" in sys.argv):
    pass
elif sys.argv[1] == "bdist_conda":
    build_requires = [
        "openblas",
        "fftw",
        "setuptools",
        "psutil",
        "numpy",
        "scipy",
        "h5py",
        "bottleneck",
        "pyfftw",
        "scikit-image",
        "vigra",
        "spams",
        "rank_filter",
        "pyqt",
        "volumina"
    ]

    install_requires = [
        "openblas",
        "fftw",
        "setuptools",
        "psutil",
        "numpy",
        "scipy",
        "h5py",
        "bottleneck",
        "pyfftw",
        "scikit-image",
        "vigra",
        "spams",
        "rank_filter",
        "pyqt",
        "volumina"
    ]

    tests_require = [
        "nose"
    ]
elif sys.argv[1] == "build_sphinx":
    import sphinx.apidoc

    sphinx.apidoc.main([
        sphinx.apidoc.__file__,
        "-f", "-T", "-e", "-M",
        "-o", "docs",
        ".", "setup.py", "tests", "versioneer.py"
    ])

setup(
    name="nanshe",
    version=versioneer.get_version(),
    description="An image processing toolkit.",
    url="https://github.com/jakirkham/nanshe",
    license="GPLv3",
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    scripts=glob("bin/*"),
    packages=find_packages(exclude=["tests*"]),
    cmdclass=versioneer.get_cmdclass(),
    build_requires=build_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    zip_safe=True
)
