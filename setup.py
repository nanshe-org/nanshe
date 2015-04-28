__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 30, 2015 23:17:09 EDT$"


from glob import glob
import os
import shutil
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
sphinx_build_pdf = False
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
        "openblas",
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

    build_prefix_arg_index = None
    for each_build_arg in ["-b", "--builder"]:
        try:
            build_arg_index = sys.argv.index(each_build_arg)
        except ValueError:
            continue

        if sys.argv[build_arg_index + 1] == "pdf":
            sphinx_build_pdf = True
            sys.argv[build_arg_index + 1] = "latex"

elif sys.argv[1] == "clean":
    saved_rst_files = ["docs/index.rst", "docs/readme.rst"]

    tmp_rst_files = glob("docs/*.rst")

    print "removing 'docs/*.rst'"
    for each_saved_rst_file in saved_rst_files:
        print "skipping '" + each_saved_rst_file + "'"
        tmp_rst_files.remove(each_saved_rst_file)

    for each_tmp_rst_file in tmp_rst_files:
        os.remove(each_tmp_rst_file)

    if os.path.exists("build/sphinx/doctrees"):
        print "removing 'build/sphinx/doctrees'"
        shutil.rmtree("build/sphinx/doctrees")
    else:
        print "'build/sphinx/doctrees' does not exist -- can't clean it"

    if (len(sys.argv) > 2) and (sys.argv[2] in ["-a", "--all"]):
        if os.path.exists("build/sphinx"):
            print "removing 'build/sphinx'"
            shutil.rmtree("build/sphinx")
        else:
            print "'build/sphinx' does not exist -- can't clean it"

setup(
    name="nanshe",
    version=versioneer.get_version(),
    description="An image processing toolkit.",
    url="https://github.com/jakirkham/nanshe",
    license="GPLv3",
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    scripts=glob("bin/*"),
    py_modules=["versioneer"],
    packages=find_packages(exclude=["tests*"]),
    cmdclass=versioneer.get_cmdclass(),
    build_requires=build_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="nose.collector",
    zip_safe=True
)

if sphinx_build_pdf:
    make_cmd = os.environ.get("MAKE", "make")
    cwd = os.getcwd()
    os.chdir("build/sphinx/latex")
    os.execlpe(make_cmd, "all", os.environ)
    os.chdir(cwd)
