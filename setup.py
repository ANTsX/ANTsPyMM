
from setuptools import setup

long_description = open("README.md").read()

setup(name='antspyt1w',
      version='0.2.3',
      description='T1w human neuroimage processing with antspyx',
      long_description=long_description,
      long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
      url='https://github.com/stnava/ANTsPyT1w',
      author='Avants, Gosselin, Tustison',
      author_email='stnava@gmail.com',
      license='Apache 2.0',
      packages=['antspyt1w'],
      zip_safe=False)
