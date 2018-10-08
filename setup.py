import soyspacing
import setuptools
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="soyspacing",
    version=soyspacing.__version__,
    author=soyspacing.__author__,
    author_email='soy.lovit@gmail.com',
    description="Spacing Error Correction Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lovit/soyspacing',
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.12.0"],
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
    keywords = [
        'Space-error-correction',
        'Korean-spacing',
        'Korean',
        'Natural-Language-Processing'
    ],
)