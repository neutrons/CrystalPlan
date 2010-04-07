#!/usr/bin/env python
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

#--- GUI Imports ---
import CrystalPlan.gui
import CrystalPlan.gui.launch_gui

#--- Model Imports ---
import CrystalPlan.model

CrystalPlan.gui.launch_gui.launch_gui()