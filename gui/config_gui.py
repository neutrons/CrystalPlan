"""Module holding configuration data and options for the GUI."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- Traits Imports ---
from enthought.traits.api import HasTraits,Int,Float,Str,Property,Bool, List


# ===========================================================================================
class GuiConfig(HasTraits):
    show_d_spacing = Bool(False, desc="whether to show d-spacing instead of q- in figures and charts.")

    max_3d_points = Int(400000, desc="the maximum # of points to show in the reciprocal space 3D view of single reflections.")

    #inelastic_mode = Bool(False, desc="whether the software is running for an inelastic instrument.")


#Shared between GUI elements, this is the configuration data.
cfg = GuiConfig()




