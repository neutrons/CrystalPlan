#!/usr/bin/env python
# whether or not to use vglrun
import os
VGLRUN=""
if os.environ.get('NXSESSIONID') is not None:
  os.system('command -v vglrun >/dev/null 2>&1 || { echo >&2 "CrystalPlan requires VirtualGL but it is not installed.  Aborting."; exit 1; }')
  VGLRUN="vglrun"
elif os.environ.get('TLSESSIONDATA') is not None:
  os.system('command -v vglrun >/dev/null 2>&1 || { echo >&2 "CrystalPlan requires VirtualGL but it is not installed.  Aborting."; exit 1; }')
  VGLRUN="vglrun"

"""The CrystalPlan application.

CrystalPlan is an experiment planning tool for crystallography.
You can choose an instrument and supply your sample's lattice
parameters to simulate which reflections will be measured,
by which detectors and at what wavelengths.

Author: Janik Zikovsky, zikovskyjl@ornl.gov
Version: $Id$
"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#Simply import and launch the GUI
os.system(VGLRUN +" /usr/bin/python <<END\n"\
"import CrystalPlan.gui.main\n"\
"CrystalPlan.gui.main.handle_arguments_and_launch(InstalledVersion=True)\n"\
"END")
