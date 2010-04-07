"""Module holding configuration data for the calculation and modeling."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- Traits Imports ---
from enthought.traits.api import HasTraits,Int,Float,Str,Property,Bool, List, String


# ===========================================================================================
class ModelConfig(HasTraits):

    use_multiprocessing = Bool(False, desc='to use multiprocessing when calculating coverage.')
    reflection_divergence_degrees = Float(0.3, desc='the default half-width of the divergence of scattered beams, in degrees.')
    default_detector_filename = String("instruments/TOPAZ_detectors_2010.csv", desc="the default file and path to the detectors .CSV geometry specification file.")
    force_pure_python = Bool(False, desc='to force calculations to use pure Python, rather than inline C code.')

cfg = ModelConfig()