#!/bin/sh
# whether or not to use vglrun
if [ -n "${NXSESSIONID}" ]; then
  command -v vglrun >/dev/null 2>&1 || { echo >&2 "MantidPlot requires VirtualGL but it's not installed.  Aborting."; exit 1; }
  VGLRUN="vglrun"
elif [ -n "${TLSESSIONDATA}" ]; then
  command -v vglrun >/dev/null 2>&1 || { echo >&2 "MantidPlot requires VirtualGL but it's not installed.  Aborting."; exit 1; }
  VGLRUN="vglrun"
fi
$VGLRUN /usr/bin/python <<END
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
import CrystalPlan.gui.main
CrystalPlan.gui.main.handle_arguments_and_launch(InstalledVersion=True)
END
