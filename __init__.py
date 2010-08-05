"""CrystalPlan package file."""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: CrystalPlan.py 1127 2010-04-01 19:28:43Z 8oz $

#print "CrystalPlan module being imported from", __file__
#print "CrystalPlan module, __name__ is", __name__

from CrystalPlan_version import version as __version__

#Manipulate the PYTHONPATH to put model directly in view of it
#   This way, "import model" works.
import sys
import os
(head, tail) = os.path.split(__file__)
sys.path.insert(0, head)

#Import the sub-modules: needed when running the installed script version.
import model
import gui
import pyevolve

#And the stuff in here
import CrystalPlan_version
