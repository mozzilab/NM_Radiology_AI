#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
      name="nmrezman",
      version="1.10",
      description="NM Results Management AI Tools",
      url="http://github.com/mozzilab/NM_Radiology_AI",
      author="NM HIT Team",
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      license="MIT",
      zip_safe=False,
)