
from setuptools import setup

long_description = open("README.md").read()

setup(name='antspymm',
      version='0.3.9',
      description='multi-channel/time-series medical image processing with antspyx',
      long_description=long_description,
      long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
      url='https://github.com/stnava/ANTsPyMM',
      author='Avants, Gosselin, Tustison, Reardon',
      author_email='stnava@gmail.com',
      license='Apache 2.0',
      packages=['antspymm'],
      zip_safe=False)
