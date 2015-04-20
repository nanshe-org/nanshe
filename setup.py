__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 30, 2015 23:17:09 EDT$"


from glob import glob

from setuptools import setup, find_packages
import versioneer


versioneer.VCS = "git"
versioneer.versionfile_source = "nanshe/_version.py"
versioneer.versionfile_build = None
versioneer.tag_prefix = ""
versioneer.parentdir_prefix = "nanshe-"


setup(
    name="nanshe",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(exclude=["tests*"]),
    url="https://github.com/jakirkham/nanshe",
    license="GPLv3",
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    description="An image processing toolkit.",
    scripts=glob("bin/*")
)
