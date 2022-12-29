
from setuptools import setup

long_description = open("README.md").read()

with open("./requirements.txt") as f:
      requirements = f.read().splitlines()

setup(name='antspymm',
      version='0.7.7',
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
