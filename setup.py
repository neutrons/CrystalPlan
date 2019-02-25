#!/usr/bin/env python
"""Script for installing the CrystalPlan utility."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- Imports ---
from setuptools import setup, find_packages
import sys
import CrystalPlan_version

#Two packages: the GUI and the model code
packages = find_packages()
packages = ['CrystalPlan', 'CrystalPlan.model',  'CrystalPlan.pyevolve', 'CrystalPlan.gui', 'CrystalPlan.model.pygene']
package_dir = {'CrystalPlan': '.',  'CrystalPlan.pyevolve':'pyevolve', 'CrystalPlan.model':'model', 'CrystalPlan.gui':'gui', 'CrystalPlan.model.pygene':'model/pygene'}
#data_files = [ ('instruments', './instruments/*.csv'), ('instruments', './instruments/*.xls') ]
data_files = []
package_data = {'CrystalPlan':['instruments/*.xls', 'instruments/*.csv', 'instruments/*.detcal',
                               'docs/*.*', 'docs/animations/*.*', 'docs/eq/*.*', 'docs/screenshots/*.*' ],
    'CrystalPlan.model':['data/*.*'],
    'CrystalPlan.gui':['icons/*.png']
}
scripts = ['crystalplan.py']

#Package requirements
install_requires = ['Traits', 'Mayavi', 'numpy', 'scipy']

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
          package_data=package_data,
          #include_package_data=True,
          install_requires=install_requires,
          #test_suite='model.test_all.get_all_tests'
          )
