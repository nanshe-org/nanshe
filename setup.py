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
if sys.argv[1] == "bdist_conda":
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
