from description import __version__, __author__
from setuptools import setup

readme = """Spacing error correction algorithms. It contains two algorithms"""

setup(
   name="soyspacing",
   version=__version__,
   author=__author__,
   author_email='soy.lovit@gmail.com',
   url='https://github.com/lovit/soyspacing',
   description="Spacing Error Correction Tools",
   long_description=readme,
   install_requires=["numpy"],
   keywords = ['spacing error correction']
)