
from setuptools import setup, find_packages

long_description = open("README.md").read()

requirements = [
      "h5py>=2.10.0",
      "numpy>=1.19.4",
      "pandas>=1.0.1",
      "antspyx",
      "antspyt1w>=0.2.3",
      "pathlib",
      "dipy",
      "nibabel",
      "scipy",
      "siq",
      "scikit-learn"]

setup(name='antspymm',
      version='1.3.5',
      description='multi-channel/time-series medical image processing with antspyx',
      long_description=long_description,
      long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
      url='https://github.com/stnava/ANTsPyMM',
      author='Avants, Gosselin, Tustison, Reardon',
      author_email='stnava@gmail.com',
      license='Apache 2.0',
      install_requires=requirements,
      packages=['antspymm'],
      zip_safe=False)
