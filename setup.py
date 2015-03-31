__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 30, 2015 23:17:09 EDT$"


from glob import glob

from setuptools import setup, find_packages

setup(
    name="nanshe",
    version="0.1a",
    packages=find_packages(exclude=["tests*"]),
    url="https://github.com/jakirkham/nanshe",
    license="GPLv3",
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    description="An image processing toolkit.",
    setup_requires=["nose>=1.2", "sphinx>=1.3"],
    scripts=glob("bin/*")
)
