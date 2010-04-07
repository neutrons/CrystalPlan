#!/usr/bin/env python
"""Script for installing the CrystalPlan utility."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- Imports ---
from distutils.core import setup
#from setuptools import setup
import sys
import CrystalPlan_version

#Two packages: the GUI and the model code
packages = ['CrystalPlan', 'CrystalPlan.model', 'CrystalPlan.gui', 'CrystalPlan.model.pygene']
package_dir = {'CrystalPlan': '',  'CrystalPlan.model':'model', 'CrystalPlan.gui':'gui', 'CrystalPlan.model.pygene':'model/pygene'}
#data_files = [ ('instruments', './instruments/*.csv'), ('instruments', './instruments/*.xls') ]
data_files = []
package_data = {'CrystalPlan':['instruments/*.xls', 'instruments/*.csv']}
scripts = ['CrystalPlan.py']

#Package requiremetns
install_requires = ['enthought']

def pythonVersionCheck():
    # Minimum version of Python
    PYTHON_MAJOR = 2
    PYTHON_MINOR = 5

    if sys.version_info < (PYTHON_MAJOR, PYTHON_MINOR):
        print >> sys.stderr, 'You need at least Python %d.%d for %s %s' \
              % (PYTHON_MAJOR, PYTHON_MINOR, CrystalPlan_version.package_name, CrystalPlan_version.version)
        sys.exit(-3)

if __name__ == "__main__":
    pythonVersionCheck()

    setup(name=CrystalPlan_version.package_name,
          version=CrystalPlan_version.version,
          description=CrystalPlan_version.description,
          author=CrystalPlan_version.author, author_email=CrystalPlan_version.author_email,
          url=CrystalPlan_version.url,
          scripts=scripts,
          packages=packages,
          package_dir=package_dir,
          data_files=data_files,
          package_data=package_data)