#!/usr/bin/env python
"""Script for installing the OldCrystalPlan utility."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- Imports ---
from setuptools import setup, find_packages
import sys
import OldCrystalPlan_version

#Two packages: the GUI and the model code
packages = find_packages()
packages = ['OldCrystalPlan', 'OldCrystalPlan.model',  'OldCrystalPlan.pyevolve', 'OldCrystalPlan.gui', 'OldCrystalPlan.model.pygene']
package_dir = {'OldCrystalPlan': '.',  'OldCrystalPlan.pyevolve':'pyevolve', 'OldCrystalPlan.model':'model', 'OldCrystalPlan.gui':'gui', 'OldCrystalPlan.model.pygene':'model/pygene'}
#data_files = [ ('instruments', './instruments/*.csv'), ('instruments', './instruments/*.xls') ]
data_files = []
package_data = {'OldCrystalPlan':['instruments/*.xls', 'instruments/*.csv', 'instruments/*.detcal',
                               'docs/*.*', 'docs/animations/*.*', 'docs/eq/*.*', 'docs/screenshots/*.*' ],
    'OldCrystalPlan.model':['data/*.*'],
    'OldCrystalPlan.gui':['icons/*.png']
}
scripts = ['oldcrystalplan.py']

#Package requirements
install_requires = ['EnthoughtBase', 'Traits', 'Mayavi', 'numpy', 'scipy']

def pythonVersionCheck():
    # Minimum version of Python
    PYTHON_MAJOR = 2
    PYTHON_MINOR = 5

    if sys.version_info < (PYTHON_MAJOR, PYTHON_MINOR):
        print >> sys.stderr, 'You need at least Python %d.%d for %s %s' \
              % (PYTHON_MAJOR, PYTHON_MINOR, OldCrystalPlan_version.package_name, OldCrystalPlan_version.version)
        sys.exit(-3)

if __name__ == "__main__":
    pythonVersionCheck()

    setup(name=OldCrystalPlan_version.package_name,
          version=OldCrystalPlan_version.version,
          description=OldCrystalPlan_version.description,
          author=OldCrystalPlan_version.author, author_email=OldCrystalPlan_version.author_email,
          url=OldCrystalPlan_version.url,
          scripts=scripts,
          packages=packages,
          package_dir=package_dir,
          data_files=data_files,
          package_data=package_data,
          #include_package_data=True,
          install_requires=install_requires,
          #test_suite='model.test_all.get_all_tests'
          )
