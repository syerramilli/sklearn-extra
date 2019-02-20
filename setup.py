# Initialize setup
import os
import sys
from setuptools import setup,find_packages
here = os.path.abspath(os.path.dirname(__file__))

setup(name='sklearn-extra',
      version='0.1',
      description='Custom tools extending scikit-learn',
      url='http://github.com/syerramilli/sklearn-extra',
      author='Suraj Yerramilli',
      author_email='surajyerramilli@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy','scipy','scikit-learn'],
      zip_safe=False)