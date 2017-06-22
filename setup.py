from description import __version__, __author__
from setuptools import setup, find_packages

readme = """Spacing error correction algorithms. """

setup(
   name="soyspacing",
   version=__version__,
   author=__author__,
   author_email='soy.lovit@gmail.com',
   url='https://github.com/lovit/soyspacing',
   description="Spacing Error Correction Tools",
   long_description=readme,
   install_requires=["numpy>=1.12.0"],
   keywords = ['Korean spacing error correction'],
   packages=find_packages()
)