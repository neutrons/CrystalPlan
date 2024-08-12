"""
Experiment module: Holds the Experiment class.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import warnings
import numpy as np
import time
import threading
import weave
import scipy.ndimage
from cPickle import loads, dumps
import os

#--- Model Imports ---
import instrument
import goniometer
import crystals
import config
import numpy_utils
from crystals import Crystal
from reflections import Reflection, ReflectionRealMeasurement
import crystal_calc
from numpy_utils import *
import utils


#======================================================================================
#========== EXPERIMENT STOPPING CRITERIA ==============================================
#======================================================================================

class StoppingCriterionInfo:
    """Simple class describing info on the stopping criterion for a run."""

    def __init__(self, name, friendly_name=""):
        """Constructor."""
        self.name = name
        self.friendly_name = friendly_name


#List of all the stopping criteria available
criteria = [ \
    StoppingCriterionInfo("pcharge", "proton charge (pC)"),
    StoppingCriterionInfo("runtime", "runtime (seconds)"),
    StoppingCriterionInfo("counts", "counts"),
    StoppingCriterionInfo("monitorcounts", "monitor counts"),
    StoppingCriterionInfo("roicounts", "roi counts") ]

#----------------------------------------------------
def get_stopping_criteria_names():
    """Return a list of string names of stopping criteria."""
    return [x.friendly_name for x in criteria]

#----------------------------------------------------
def get_stopping_criterion_from_friendly_name(friendly):
    """Find a match to a criterion using the friendly name, and
    return the unfriendly name."""
    for x in criteria:
        if friendly == x.friendly_name:
            return x.name
    return ""

#----------------------------------------------------
def get_stopping_criterion_friendly_name(name):
    """Find the friendly name of a stopping criterion, using the unfriendly name."""
    for x in criteria:
        if name == x.name:
            return x.friendly_name
    return ""


#===================================================================================================
#========== PARAMETERS FOR DISPLAYING QSPACE COVERAGE ==============================================
#===================================================================================================

#These constants are keys to the parameters dictionaries
PARAM_POSITIONS = "PARAM_POSITIONS"
PARAM_TRY_POSITION = "PARAM_TRY_POSITION"
PARAM_DETECTORS = "PARAM_DETECTORS"
PARAM_SLICE = "PARAM_SLICE"
PARAM_ENERGY_SLICE = "PARAM_ENERGY_SLICE"
PARAM_INVERT = "PARAM_INVERT"
PARAM_SYMMETRY = "PARAM_SYMMETRY"
PARAM_DISPLAY = "PARAM_DISPLAY"
PARAM_REFLECTIONS = "PARAM_REFLECTIONS"
PARAM_REFLECTION_MASKING = "PARAM_REFLECTION_MASKING"
PARAM_REFLECTION_HIGHLIGHTING = "PARAM_REFLECTION_HIGHLIGHTING"
PARAM_REFLECTION_DISPLAY = "PARAM_REFLECTION_DISPLAY"

#List of all valid parameters (see above)
VALID_PARAMS = [PARAM_POSITIONS, PARAM_TRY_POSITION, PARAM_DETECTORS, PARAM_ENERGY_SLICE, PARAM_SLICE, PARAM_INVERT, PARAM_SYMMETRY, PARAM_DISPLAY, PARAM_REFLECTIONS, PARAM_REFLECTION_MASKING, PARAM_REFLECTION_HIGHLIGHTING, PARAM_REFLECTION_DISPLAY]

#-------------------------------------------------------------------------------
class ParamsDict(dict):
    """Subclass of dictionary used to store the display paramters.
    Restricts inputs to valid keys."""
    
    #Create a lock to avoid threading problems.
    _lock = threading.Lock()
    
    def __init__(self, **kwargs):
        """Constructor."""
        dict.__init__(self, **kwargs)

    def __setitem__(self, key, value):
        """Override the setitem operator to prevent invalid entries.
        Raises an error otherwise."""
        if key is None: raise KeyError("Key of None specificed.")
        if not (key in VALID_PARAMS):
            raise KeyError("Unexpected key supplied!");
        #Otherwise, the key is okay. Check the value
        if not (value is None) and (not isinstance(value, Params)):
            raise ValueError("You can only pass instances of model.experiment.Params (and subclasses of it).")
        #All good!
        try:
            self._lock.acquire()
            dict.__setitem__(self, key, value)
        finally:
            self._lock.release()
            
    def __getitem__(self, key):
        """Override the getitem operator to return None when no entry
        exists, instead of throwing an error."""
        #But we can only ask keys that are valid.
        if key is None: raise KeyError("Key of None specificed.")
        if not (key in VALID_PARAMS):
            raise KeyError("Unexpected key supplied!");      
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            #If the key doesn't exist, return None
            #Other errors should be raised.
            return None

        

#========================================================================================================
#========================================================================================================
        
#-------------------------------------------------------------------------------
class Params:
    """Empty base class for describing parameters for q-space coverage display and 
    calculation."""

    def __eq__(self, other):
        """Default equality operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        return True

    
    
#-------------------------------------------------------------------------------
class ParamPositions(Params):
    """List of which positions to add up to calculate coverage."""
    
    #self.positions is a dictionary.
    #   The key is the ID of the PositionCoverage object saved in the instrument object.
    #   The value is True or False, depending on whether that position is used
    
    def __init__(self, pos_dict):
        if pos_dict is None:
            raise StandardError("ParamPositions constructor called with a None argument.")
        self.positions = pos_dict.copy()
        
    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        return (self.positions == other.positions)
        

#-------------------------------------------------------------------------------
class ParamTryPosition(Params):
    """Try a single extra position by adding it to the already selected list."""

    def __init__(self, try_position, use_trial):
        #Note: we don't copy the position coverage object, we just refer to it.

        #The position coverage object holding the qspace coverage and the angles.
        self.try_position = try_position
        #Do we use the trial position?
        self.use_trial = use_trial

    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        if not self.use_trial and not other.use_trial: return True
        return (self.try_position == other.try_position) and (self.use_trial == other.use_trial) 


#-------------------------------------------------------------------------------
class ParamDetectors(Params):
    """List of which detectors to add up to calculate coverage."""
    
    #Bool array of which of the detectors that will be used in the calculation.
    detectors = None
        
    def __init__(self, detectors):
        self.detectors = detectors
        
    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        return (self.detectors == other.detectors)

    def is_checked(self, num):
        """Is the detector at index 'num' selected in this parameter?"""
        if not self.detectors is None:
            return self.detectors[num]
        else:
            return False

#-------------------------------------------------------------------------------
class ParamSymmetry(Params):
    """Consider crystal symmetry in q-space volume calculation?"""
    
    def __init__(self, use_symmetry):
        #Whether or not you use symmetry
        self.use_symmetry = use_symmetry

    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        return (self.use_symmetry == other.use_symmetry)



#-------------------------------------------------------------------------------
class ParamInvert(Params):
    """Do you invert the coverage data to show gaps more easily?"""
    
    def __init__(self, invert):
        #Whether or not you invert
        self.invert = invert
        
    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        return (self.invert == other.invert)
   

#-------------------------------------------------------------------------------
class ParamSlice(Params):
    """How is the data "sliced" spherically to show it."""
    #Whether or not you slice it.
    use_slice = True

    #Min and max to slice
    slice_min = 0
    slice_max = 8

    def __init__(self, use_slice, slice_min=0, slice_max=10):
        self.use_slice = use_slice
        self.slice_min = slice_min
        self.slice_max = slice_max

    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        if (self.use_slice != other.use_slice): return False
        return (self.slice_min == other.slice_min) and (self.slice_max == other.slice_max)


#-------------------------------------------------------------------------------
class ParamDisplay(Params):
    """Assorted parameters for how to display the calculated coverage.
    These are parameters that do not affect the calculation, just the display."""

    def __init__(self):
        #Do we show redundancy?
        self.show_redundancy = False

    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        return (self.show_redundancy == other.show_redundancy)


#-------------------------------------------------------------------------------
class ParamReflections(Params):
    """Parameters for calculating single peak reflections."""
    #No attributes yet.

    def __init__(self):
        pass

    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        #Never the same, just recalculate!
        return False


#-------------------------------------------------------------------------------
class ParamReflectionMasking(ParamSlice):
    """The options on how to slice or mask the reflection peaks."""
    def __init__(self, use_slice=False, slice_min=0, slice_max=10):
        #Call the parent constructor. For the slice
        ParamSlice.__init__(self, use_slice, slice_min, slice_max)
        #Integer that is entry in the list of choices.
        self.masking_type = 1
        #I/sigI threshold for measured peaks.
        self.threshold = 2.0
        #Are we just showing the primary ones?
        self.primary_reflections_only = False
        #Are we showing equivalent reflections as counting as measured?
        self.show_equivalent_reflections = False

    def __eq__(self, other):
        """Equality (==) operator."""
        if not ParamSlice.__eq__(self, other):
            return False
        else:
            return (self.primary_reflections_only == other.primary_reflections_only) and \
                    (self.masking_type == other.masking_type) and\
                    (self.threshold == other.threshold) and\
                    (self.show_equivalent_reflections == other.show_equivalent_reflections)


#-------------------------------------------------------------------------------
class ParamReflectionHighlighting(Params):
    """The options on what peaks to highlight in the 3D view."""
    #TODO:
    pass


#-------------------------------------------------------------------------------
class ParamReflectionDisplay(Params):
    """The options on how to display single-reflection peaks."""
    #Constants for how to display
    DISPLAY_AS_PIXELS = 0
    DISPLAY_AS_SPHERES = 1
    COLOR_BY_PREDICTED = 0
    COLOR_BY_MEASURED = 1

    def __init__(self):
        #How to display? 0=pixels, 1=spheres
        self.display_as = self.DISPLAY_AS_SPHERES
        #How to do color map?
        self.color_map = self.COLOR_BY_PREDICTED
        #Relative size of spheres.
        # For pixels, this is the size in pixels.
        self.size = 1.0
        #Automatic size?
        self.automatic_size = True

    def __eq__(self, other):
        """Equality (==) operator."""
        if (self is None): return (other is None)
        if (other is None): return (self is None)
        if not (self.display_as == other.display_as): return False
        if not (self.color_map == other.color_map): return False
        if self.automatic_size and other.automatic_size:
            return True
        if (not self.automatic_size) and (not other.automatic_size):
            if self.display_as == self.DISPLAY_AS_PIXELS:
                #Since pixels are only integers, they are only different display params if the rounded one is different.
                return (round(self.size) == round(other.size))
            else:
                return (self.size == other.size)
        return False



#========================================================================================================
#========================================================================================================
#========================================================================================================
class CoverageStats:
    """Simple class holds coverage stat for a spherical slice of q-space."""

    #Limits in q radius
    qmin = 0
    qmax = 0

    #Coverage percentage coverage[i] (where number of times measured of the point > i)
    #   i.e. coverage[0] is the basic measurement
    #   i.e. coverage[1] is any redundant measurements (2 or more times)
    coverage = None

    def __init__(self, qmin, qmax):
        self.qmin = qmin
        self.qmax = qmax
        self.coverage = np.zeros(4)

    #========================================================================================================
    def __eq__(self, other):
        return utils.equal_objects(self, other)


#========================================================================================================
#========================================================================================================
#========================================================================================================
class ReflectionStats:
    """Simple class holds stats about how many peaks are measured, how many times, etc."""
    #Total # of peaks
    total = 0
    #Measured at least once
    measured = 0
    #Measured at least twice
    redundant = 0
    def __str__(self):
        return "Total %d, measured %d, redundant %d." % (self.total, self.measured, self.redundant)


#========================================================================================================
#========================================================================================================
#========================================================================================================
class Experiment:
    """The experiment class holds the settings for a particular experiment.
    Additionally, some methods in Experiment evaluate the quality of the experiment.
    """

    #Range in h,k,l as a list that can be indexed
    def get_range_hkl(self):
        return [self.range_h, self.range_k, self.range_l]
    range_hkl = property(get_range_hkl)


    #-------------------------------------------------------------------------------
    def initialize(self):
        """Empty initializer"""
        #Lock on the qspace_displayed member, for threading.
        self._lock_qspace_displayed = threading.Lock()

        #Dictionary containing all the parameters for calculating/displaying the coverage
        self.params = ParamsDict()

        #The sample crystal that is being examined. type=Crystal
        self.crystal = Crystal("new crystal")

        #Ranges of h,k,l values to look at. These are INCLUSIVE (last value is also in the list).
        self.range_h = (-6, 6)
        self.range_k = (-6, 6)
        self.range_l = (-6, 6)

        #Will the hkl range be chosen automatically?
        self.range_automatic = True

        #Will the hkl reflections limit q to a sphere of radius 2pi/dmin
        self.range_limit_to_sphere = True

        #List of Reflection objects for each reflection
        self.reflections = None

        #3xN array of the hkl values of each reflection. This is used to speed up some calculations.
        self.reflections_hkl = None

        #Dictionary of all reflections where the index is a tuple of (h,k,l) - used to speed up look-ups
        self.reflections_dict = {}

        #3xN array of the q vector of each reflection. This may be used to speed up displays.
        self.reflections_q_vector = None

        #Nx1 array of the # of times each reflection was measured. This may be used to speed up displays.
        self.reflections_times_measured = None

        #Nx1 array of the # of times each reflection was measured, ALSO counting equivalent reflections (due to symmetries)
        self.reflections_times_measured_with_equivalents = None

        #N-sized 1D bool array - mask of which reflections to display in the 3D view.
        self.reflections_mask = None
        self.primary_reflections_mask = None

        #Helper stuff
        self.reflection_masked_index_to_real_index = None
        self.reflections_q_norm = None

        #q-space coverage map, with the current settings. Indices are x,y,z
        self.qspace = None

        #Slice from qmin to qmax of the qspace found
        self.qspace_displayed = None

        #For volume symmetry map
        self.volume_symmetry = None

        #Coverage percentage for several slices
        self.coverage_stats = list()

        #Overall coverage and redundancy percentages.
        self.overall_coverage = 0.0
        self.overall_redundancy = 0.0

        #Statistics for reflections, with and without symmetry
        self.reflection_stats = ReflectionStats()
        self.reflection_stats_with_symmetry = ReflectionStats()
        #Adjusted statistics, e.g. with edges.
        self.reflection_stats_adjusted = ReflectionStats()
        self.reflection_stats_adjusted_with_symmetry = ReflectionStats()

        #For output
        self.verbose = True

        #For real measurements; list of loaded files.
        self.real_measurement_filenames = []

        #Nx1 array of the # of times each reflection was measured. This may be used to speed up displays.
        self.reflections_times_real_measured = None

        #Nx1 array of the # of times each reflection was measured, ALSO counting equivalent reflections (due to symmetries)
        self.reflections_times_real_measured_with_equivalents = None


    #-------------------------------------------------------------------------------
    def __init__(self, instrument_to_use):
        """Constructor. Subscribe to messages.
            inst: The instrument this experiment refers to."""
            
        self.initialize()

        #Reference to the instrument
        self.inst = instrument_to_use

    #========================================================================================================
    def __eq__(self, other):
        return utils.equal_objects(self, other)


    #========================================================================================================
    #======================================= PICKLING =====================================
    #========================================================================================================
    def __getstate__(self):
        """Return a dictionary containing all the stuff to pickle in an experiment."""
        #Exclude all these attributes.
        exclude_list = ['reflections', 'reflections_dict', 'reflections_q_vector',
        'reflections_times_measured', 'reflections_times_measured_with_equivalents',
        'reflections_times_real_measured', 'reflections_times_real_measured_with_equivalents',
        'reflections_hkl', 'reflections_mask', 'primary_reflections_mask',
        'qspace', 'qspace_displayed', 'volume_symmetry',
        'coverage_stats', 'reflection_stats',
        'reflection_stats_with_symmetry', 'reflection_stats_adjusted', 'reflection_stats_adjusted_with_symmetry',
        'reflection_masked_index_to_real_index', 'reflections_q_norm'
        ]
        d =  utils.getstate_except(self, exclude_list)
        return d

    #========================================================================================================
    def __setstate__(self, d):
        """Set the state of experiment, using d as the settings dictionary."""
        self.initialize()
        for (key, value) in d.items():
            setattr(self, key, value)
        #Ok, now fix everything
        # self.inst should be good though, its own setstate does it.
        self.initialize_reflections()
        self.recalculate_reflections(None, calculation_callback=None)
        self.initialize_volume_symmetry_map()
        self.calculate_coverage(None, None)


    #-------------------------------------------------------------------------------
    def set_parameters(self, params_dict):
        """Set the parameters dictionary. Does not re-do calculations.
            params_dict: the new dictionary of Params objects to use. Uses a copy of the dictionary."""
        if not isinstance(params_dict, ParamsDict):
            raise ValueError("experiment.set_parameters(): params_dict is not a ParamsDict object.")
        self.params = params_dict.copy()


    #========================================================================================================
    #======================================= REFLECTION RANGE =====================================
    #========================================================================================================

    #-------------------------------------------------------------------------------
    def automatic_hkl_range(self, check_only=False):
        """Automatically determine the correct HKL range of peaks that will fit
        all peaks.

        Parameters
            check_only: bool, set to True to just check how many peaks there are, and return them."""
        #This is the q we want
        qlim = self.inst.qlim
        #This the q of each reciprocal lattice vector
        rec = self.crystal.reciprocal_lattice

        #Look in each corner to find the hkl values there
        corner_hkl = np.zeros( (3, 8) )
        ql = [-qlim, qlim]
        i = 0
        for x in ql:
            for y in ql:
                for z in ql:
                    hkl = crystal_calc.get_hkl_from_q( column([x,y,z]), rec)
                    #Save it here
                    corner_hkl[:, i] = hkl.flatten()
                    i += 1
        #Round them up
        ranges = []
        corner_hkl = np.round(corner_hkl)
        for (i, field) in enumerate(['range_h', 'range_k', 'range_l']):
            index_min = np.min(corner_hkl[i,:])
            index_max = np.max(corner_hkl[i,:])
            ranges.append( index_max - index_min + 1)
            #Set the range to these
            if not check_only:
                setattr(self, field, (index_min, index_max) )
        #Return the # of reflections
        return ranges[0]*ranges[1]*ranges[2]


    #-------------------------------------------------------------------------------
    def initialize_reflections(self):
        """Using the desired range of HKL values, initialize all the Reflection objects
        in the experiment.
        """
        if self.range_automatic:
            self.automatic_hkl_range()

        #Clear existing stuff
        refls_dict = {}

        #Lists of h,k and l
        h_list = range(int(self.range_h[0]), int(self.range_h[1]+1))
        k_list = range(int(self.range_k[0]), int(self.range_k[1]+1))
        l_list = range(int(self.range_l[0]), int(self.range_l[1]+1))

        #Overall number of reflections
        num_h, num_k, num_l = (len(h_list), len(k_list), len(l_list))
        n = num_h * num_k * num_l

        #--- Make the hkl 3xN array ---
        reflections_hkl = np.zeros( (3, n) )
        #First axis (h) varies slowest. Each element repeats num_k * num_l times
        reflections_hkl[0, :] = np.tile(h_list, (num_k * num_l, 1)).ravel('F')
        #Second axis,( each element repeats num_l times) repeats num_h times
        will_repeat = np.tile(k_list, (num_l, 1)).ravel('F')
        reflections_hkl[1, :] = np.tile( will_repeat, num_h)
        #Last axis (l) varies fastest. Repeat sequence over and over
        reflections_hkl[2, :] = np.tile(l_list, num_h*num_k)
        
        
        # Get a list of all the reflection conditions contained.
        rc_list = self.crystal.get_reflection_conditions()
        # Start with all true
        visible = h_list==h_list
        for rc in rc_list:
            if not rc is None:
                visible = visible & rc.reflection_visible_matrix(reflections_hkl)
            
        # Take off anything not visible
        reflections_hkl = reflections_hkl[:,visible]
        
        #Calculate all the q vectors at once
        all_q_vectors = np.dot(self.crystal.reciprocal_lattice, reflections_hkl)
        self.reflections_q_norm = np.sqrt(np.sum(all_q_vectors**2, axis=0))

        if self.range_limit_to_sphere:
            #Limit to a sphere of radius 2pi/dmin
            inside_sphere = (self.reflections_q_norm < self.inst.qlim)
            reflections_hkl = reflections_hkl[:, inside_sphere]
            all_q_vectors = all_q_vectors[:, inside_sphere]
            self.reflections_q_norm = self.reflections_q_norm[inside_sphere]

        #Create each object, and add to the list
        refls = list()
        for i in xrange(reflections_hkl.shape[1]):
            hkl = tuple(reflections_hkl[:, i])
            new_refl = Reflection( hkl, all_q_vectors[:, i] )
            refls.append( new_refl )
            refls_dict[hkl] = new_refl

        #Save them in the object
        self.reflections_hkl = reflections_hkl
        self.reflections_dict = refls_dict
        self.reflections = refls
        self.reflections_q_vector = all_q_vectors

        #Now we find the primary reflections using crystal symmetry.
        self.find_primary_reflections()

        #Clear the list of reflection times measured
        self.get_reflections_times_measured(clear_list=True)

        #And if you had any peaks files loaded, reload them.
        self.reload_peaks_files()

        #At this point, the reflections mask needs to be updated since the reflections changed.
        self.calculate_reflections_mask()




    #-------------------------------------------------------------------------------
    def get_reflection(self, h, k, l):
        """Return a Reflection instance in this experiment.

        Parameters
            h,k,l: integer indices of the reflection. If floats are supplied,
                    they are rounded to the nearest integer.

        Returns:
            ref: a Reflection instance, or None if the indices were out of bounds.
        """
        #The dictionary has the hkl tuple as its index
        return self.reflections_dict.get( (int(round(h)), int(round(k)), int(round(l))), None)


    #-------------------------------------------------------------------------------
    def get_reflection_closest_to_q(self, q_vector, use_masking=True):
        """Return the reflection found closest to the given q_vector.

        Parameters
        ----------
            q_vector: a 3x1 column array giving the qx, qy, qz of the point we are looking at.
            use_masking: look at the masked points. If False, all reflections are considered.
        """
        if self.reflections_q_vector is None:
            return None
        if use_masking and not self.reflections_mask is None:
            #Length of the masking vector and the q-vector must match
            if len(self.reflections_mask) != self.reflections_q_vector.shape[1]:
                import warnings
                warnings.warn("Warning! Non-matching size of reflections_q_vector and reflections_mask")
                return None
            elif len(self.reflections_mask)==0:
                #No mask
                return None
            q = self.reflections_q_vector[:, self.reflections_mask]
        else:
            q = self.reflections_q_vector
        #All points were masked away?
        if len(q) == 0:
            return None
        #Difference, squared
        q_diff = (q-q_vector)**2
        #Summed over x,y,z
        q_diff = np.sum(q_diff, axis=0)
        #Find the index of the lowest value
        if len(q_diff) <= 0:
            return None
        index = np.argmin(q_diff)
        #Return the reflection
        return self.get_reflection_from_masked_id(index)


    #-------------------------------------------------------------------------------
    def get_reflection_from_masked_id(self, id):
        """Return a Reflection from a point ID in a 3D view.

        Parameters
            id: point id in the 3D view; AKA the id in the reflections list
                AFTER it has been masked by self.reflections_mask

        Returns:
            ref: a Reflection instance, or None if the indices were out of bounds.
        """

        #We need to convert the masked ID to the pre-mask ID
        if not hasattr(self, 'reflection_masked_index_to_real_index'):
            #No reflections have been masked yet.
            return None
        
        if self.reflection_masked_index_to_real_index is None:
            return None
        if (id < 0) or (id >= len(self.reflection_masked_index_to_real_index) ):
            return None
        real_id = self.reflection_masked_index_to_real_index[id]
        return self.reflections[real_id]

    #-------------------------------------------------------------------------------
    def get_equivalent_reflections(self, refl):
        """Using the symmetry of the crystal, return a list of reflections
        that are equivalent to the provided one.

        Parameters:
            refl: one Reflection object in the list.

        Return
            a list of Reflection objects, including the provided one."""

        if refl is None:
            return []
        
        #If it has been previously calculated, just return it
        if not len(refl.equivalent)==0:
            return refl.equivalent

        #Get the point group object
        # @type pg PointGroup
        pg = self.crystal.get_point_group()
        if pg is None:
            return [refl]

        #By using a set we eliminate doubled entries
        output = set()

        #This returns tuples with equivalent hkl
        equivalent_hkl = pg.get_equivalent_hkl(refl.hkl)
        #We now have to find the reflection object
        for other_hkl in equivalent_hkl:
            other_refl = self.get_reflection(other_hkl[0], other_hkl[1], other_hkl[2])
            if not other_refl is None:
                output.add( other_refl )

        #Put the main one at the head of the list; return as a list
        output.remove( refl )

        return [refl] + list(output)

    #-------------------------------------------------------------------------------
    def find_primary_reflections(self):
        """Make a list of all the "primary" reflections - the half- or quarter- or whatever- sphere
        of peaks that describes the crystal fully.
        This also calculates all equivalent reflections for all peaks.
        """

        t1 = time.time()

        verbose = False

        #Array where True = this is the primary reflection
        primary = np.zeros( (len(self.reflections),), dtype=bool)

        #Find the point group of the crystal
        # @type pg: PointGroup
        pg = self.crystal.get_point_group()
        if pg is None:
            self.primary_reflections = np.ones( (len(self.reflections),), dtype=bool)
            return

        #For speed, save the ranges
        range_hkl = self.range_hkl
        ranges = [(x[1]-x[0]) for x in range_hkl]
        recip_a = self.crystal.recip_a
        recip_b = self.crystal.recip_a

        for first_refl in self.reflections:
            first_refl.considered = False

        #-----------------------------------
        def sorting_value(refl):
            """Give a value for sorting the hkl peaks.
                The idea of sorting is that we look at the angle of rotation of the hkl around
            a particular axis. For example, around the "l" axis, we find the angle from +h
            by doing atan2(k,h). We want this angle to be the smallest possible.
                Each point group has a list of axes around which to sort, as a string 'lkh',
            where the first one is the most important axis. 'lkh' sorts by the l angle, then k
            angle for identical l, etc.
            """
            if refl is None: return -1e10
            #If we already considered it, reject it
            if refl.considered: return -1e10
            (h,k,l) = list(refl.hkl)
            #pg.preferred_axis gives you the rotation axes to consider, in decreasing order of importance
            angles = []
            for axis in pg.preferred_axis:
                if axis == '3':
                    x = h * recip_a[0] + k * recip_b[0]
                    y = h * recip_a[1] + k * recip_b[1]
                    #x = (h+k)*cos(pi/6)
                    #y = (h-k)*sin(pi/6)
                    angle = abs(np.arctan2(y, x))
                elif axis == 'l':
                    angle = np.arctan2(k, h)
                elif axis == 'k':
                    angle = np.arctan2(l, h)
                else:
                    angle = np.arctan2(l, k)
                if angle < 0: angle += 2*np.pi
                angles.append(angle)
            #We will sort using this list of angles
            return angles
        #------ (end sorting_value) ----


        for first_refl in self.reflections:
            if not first_refl.considered:
                #List of all equivalent reflections
                equivalent_list = self.get_equivalent_reflections(first_refl)

                if verbose: print "hkl is ", first_refl.hkl

                #Sort using the criterion
                sort_me = [(sorting_value(refl), refl) for refl in equivalent_list]
                sort_me.sort()

                #Mark the primary ones
                primary_refl = sort_me[0][1]
                for i in xrange(len(sort_me)):
                    # @type refl Reflection
                    refl = sort_me[i][1]
                    refl.is_primary = (i==0)
                    refl.considered = True
                    #Save a link to the primary reflection for this reflection
                    refl.primary = primary_refl
                    #And the list of equivalent ones
                    refl.equivalent = equivalent_list
                    
                #Make sure the primary one is primary
                primary_refl.is_primary = True

                if verbose: print [refl.hkl for (val, refl) in sort_me]

        #Set the primary mask array
        for (peak_num, refl) in enumerate(self.reflections):
            if refl.is_primary:
                primary[peak_num] = True
        print "Primary peaks: ", sum(primary), "out of", len(self.reflections), "peaks. Found in %.3f sec." % (time.time()-t1)
        if verbose: print [refl.hkl for refl in self.reflections if refl.primary]

        self.primary_reflections_mask = primary

    #-------------------------------------------------------------------------------
    def calculate_peak_shape(self, hkl, delta_hkl, poscov, det_num, num_points=100):
        """Calculate/simulate the shape (in real space and wavelength) of a single peak.
        'num_points' points will be generated randomly, using normal distribution
        around the central hkl values.

        Parameters:
            hkl: indices of peak of interest.
            delta_hkl: standard deviation around each index (i.e. half-width).
            poscov: PositionCoverage object giving the sample orientation angles.
            det_num: number of the detector we are looking at.
            num_points: number of points to simulate

        Returns:
            horizontal_delta, vertical_delta : half-width on the detector face
            wavelength_delta : half-width in wavelength space.
        """
        #TODO: Invalid input checks here.
        from scipy.stats import norm

        #Create 3 normal distribution functions
        norm_funcs = [norm(loc=hkl[i], scale=delta_hkl[i]) for i in xrange(3)]

        #Create a 3xN array with the points
        hkl_array = np.vstack( (norm_funcs[0].rvs(num_points), norm_funcs[1].rvs(num_points), norm_funcs[2].rvs(num_points)) )

        #Data needed in calculation
        rot_matrix = self.inst.goniometer.make_sample_rot_matrix(poscov.angles)
        ub_matrix = self.crystal.ub_matrix
        beam = crystal_calc.get_scattered_beam( hkl_array, rot_matrix, ub_matrix)
        det = self.inst.detectors[det_num]
        #Calculate the coordinates, but ignore the limits of wavelength
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam, -np.inf, +np.inf)
        #Return the points positions
        return (h,v,wl)
    
    #========================================================================================================
    #======================================= REFLECTION CALCULATIONS =====================================
    #========================================================================================================

    #-------------------------------------------------------------------------------
    def recalculate_reflections(self, pos_param, calculation_callback=None):
        """Re-calculate reflections for every sample position and
        every detector. initialize_reflections() should have been called previously.

        Parameters:
            pos_param: a ParamPositions object holding which positions to keep in the calculation.
                or: a list of PositionCoverage objects directly.
            calculation_callback: a function that will be called on each step of the calculation.
                Will be used by GUI to update.
                Function must expect one parameter, a PositionCoverage object.
        """

        refls = self.reflections #for minor speed-up
        if refls is None:
            #No points defined yet
            return

        #Clear the measurement list of each reflection
        for ref in refls:
            ref.measurements = list()

        #Get the list of the detectors to use
        det_bool = self.get_detectors_bool_array()

        #Get all the positions to use
        if isinstance(pos_param, list):
            #The list is already directly provided
            positions_used = pos_param
        else:
            #Extract it from the ParamPositions
            positions_used = self.get_positions_used(pos_param)

        #Ok, go through the positions.
        for poscov in positions_used:
            if not poscov is None:
                poscov_id = id(poscov)

                #Report progress
                if self.verbose: print "Calculating hkl reflections for angles at ", poscov.angles
                if not calculation_callback is None:
                    if callable(calculation_callback):
                        calculation_callback(poscov)

                #This is the sample orientation rotation matrix
                rot_matrix = self.inst.goniometer.make_sample_rot_matrix(poscov.angles)

                #This UB matrix comes from the crystal data
                ub_matrix = self.crystal.ub_matrix
                #Calculate the scattered beam direction and inverse wavelength.
                beam = crystal_calc.get_scattered_beam( self.reflections_hkl, rot_matrix, ub_matrix)

                #Find the wavelength range
                (wl_min, wl_max) = self.inst.get_wavelength_range(poscov.angles)

                for (detector_num, det) in enumerate(self.inst.detectors):
                    #Do the parameters say to use this detector?
                    if det_bool[detector_num]:
                        #Compute the scattered beam position in the coordinates of the detector.
                        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam, wl_min, wl_max)
                        #These are only the ones that did hit the detector
                        indices = np.nonzero(hits_it)[0]
                        for i in indices:
                            #print self.reflections_hkl[:, i], "hit the detector", detector_num
                            refls[i].add_measurement(poscov_id, detector_num, h[i], v[i], wl[i], distance[i])

        #We make the array of how many times measured, for all the positions
        self.get_reflections_times_measured(None)

        #We make the array of how many times REALLY measured, for all the positions
        self.get_reflections_times_real_measured(None)

        #Now we find the primary reflections using crystal symmetry.
        self.find_primary_reflections()

        #Continue on with masking
        self.calculate_reflections_mask()

        #And get some statistics
        self.calculate_reflection_coverage_stats()

    #-------------------------------------------------------------------------------
    def calculate_reflections_mask(self):
        """Generate a mask of which reflections will be displayed in 3D, based
        on the parameters saved."""
        
        if self.reflections is None:
            return

        #Some checks.
        if self.reflections_times_measured is None:
            self.get_reflections_times_measured(None)
            
        assert len(self.reflections_times_measured ) == len(self.reflections), "The list of reflections_times_measured should be the same length as the # of reflections."
        assert len(self.reflections_times_measured_with_equivalents ) == len(self.reflections), "The list of reflections_times_measured_with_equivalents should be the same length as the # of reflections."

        #Get the slicing parameters.
        # @type mask_param ParamReflectionMasking
        mask_param = self.params[PARAM_REFLECTION_MASKING]
        if mask_param is None or (len(self.reflections) == 0):
            #-- Just show everything --
            mask = np.ones( (len(self.reflections), ), dtype=bool)
        else:
            #-- Yes, some parameters exist --
            
            #Get the limits to slice to
            qmin = mask_param.slice_min
            qmax = mask_param.slice_max
            if mask_param.use_slice:
                #Make the slice
                mask = (self.reflections_q_norm <= qmax) & (self.reflections_q_norm >= qmin)
            else:
                #Not using slice, just show everything
                mask = np.ones( (len(self.reflections), ), dtype=bool )

            #Okay, now look at the other parameters
            if mask_param.masking_type > 0:
                #Are we gonna use all equivalent reflections?
                if mask_param.show_equivalent_reflections:
                    rtm_predict = self.reflections_times_measured_with_equivalents.ravel()
                    rtm_real = self.reflections_times_real_measured_with_equivalents.ravel()
                else:
                    rtm_predict = self.reflections_times_measured.ravel()
                    rtm_real = self.reflections_times_real_measured.ravel()

                #Yes, there is some masking.
                if mask_param.masking_type == 1:
                    #Predicted reflections only
                    mask &= (rtm_predict > 0)
                elif mask_param.masking_type == 2:
                    #NON-Predicted reflections only
                    mask &= (rtm_predict == 0)
                elif mask_param.masking_type == 3:
                    #REAL Measured reflections only
                    mask &= (rtm_real > 0)
                elif mask_param.masking_type == 4:
                    #NON-REAL-measured reflections only
                    mask &= (rtm_real == 0)
                elif mask_param.masking_type == 5:
                    #Predicted but not measured
                    mask &= ((rtm_predict > 0) & (rtm_real == 0))
                elif mask_param.masking_type == 6:
                    #Measured but not predicted
                    mask &= ((rtm_predict == 0) & (rtm_real > 0))

            if mask_param.primary_reflections_only and hasattr(self, 'primary_reflections_mask') :
                # Also mask out the peaks that are not primary.
                mask &= self.primary_reflections_mask

        #Save it
        self.reflections_mask = mask

        #Okay, now we make an indexing array, for convenience later
        self.reflection_masked_index_to_real_index = np.arange(len(self.reflections))[mask]

    #-------------------------------------------------------------------------------
    def get_reflections_times_measured(self, pos_param=None, clear_list=False):
        """Make the array that holds the # of times each one is measured
        This will speed up drawing.

        Parameters:
        -----------
            pos_param: a ParamPositions object holding which positions to keep in the calculation.
                None means use all of them (default).
            clear_list: set to True to simply generate an empty array here.
        """
        #Make a list of position ids
        if pos_param is None:
            position_ids = None
        else:
            positions_used = self.get_positions_used(pos_param)
            position_ids = [id(pos) for pos in positions_used]

        #Initialize the arrays
        rtm = np.zeros( (len(self.reflections), 1) )
        rtme = np.zeros( (len(self.reflections), 1) )

        if not clear_list:
            for (index, ref) in enumerate(self.reflections):
                # @type ref Reflection
                rtm[index, 0] = ref.times_measured(position_ids, add_equivalent_ones=False)
                rtme[index, 0] = ref.times_measured(position_ids, add_equivalent_ones=True)

        self.reflections_times_measured = rtm
        self.reflections_times_measured_with_equivalents = rtme
        assert len(self.reflections_times_measured ) == len(self.reflections), "The list of reflections_times_measured should be the same length as the # of reflections."
        assert len(self.reflections_times_measured_with_equivalents ) == len(self.reflections), "The list of reflections_times_measured_with_equivalents should be the same length as the # of reflections."
            

    #-------------------------------------------------------------------------------
    def get_reflections_measured(self, measured=True):
        """Return a list of the measured (or not-measured) reflections.

        Parameters:
            measured, bool: True to get the measured reflections.
                            False to get the non-measured reflections.

        Returns:
            list of Reflection objects."""

        output = list()
        for ref in self.reflections:
            times_measured = ref.times_measured()
            if (measured and (times_measured > 0)) or ((not measured) and (times_measured == 0)):
                output.append(ref)
        return output


    #-------------------------------------------------------------------------------
    def get_reflections_times_real_measured(self, threshold):
        """Make the array that holds the # of times each one was REALLY measured (from ISAW
        peaks files loaded previously)
        This will speed up drawing.

        Parameters:
        -----------
            pos_param: a ParamPositions object holding which positions to keep in the calculation.
                None means use all of them (default).
            threshold: I/sigI threshold to consider a peak "measured". Set None to use the last
                saved parameter.
        """

        #Find the threshold if not specified.
        if threshold is None:
            refmask = self.params[PARAM_REFLECTION_MASKING] #@type refmask ParamReflectionMasking
            if refmask is None: refmask = ParamReflectionMasking()
            threshold = refmask.threshold

        #Initialize the arrays
        rtm = np.zeros( (len(self.reflections), 1) )
        rtme = np.zeros( (len(self.reflections), 1) )

        for (index, ref) in enumerate(self.reflections):
            # @type ref Reflection
            rtm[index, 0] = ref.times_real_measured(threshold, add_equivalent_ones=False)
            rtme[index, 0] = ref.times_real_measured(threshold, add_equivalent_ones=True)

        self.reflections_times_real_measured = rtm
        self.reflections_times_real_measured_with_equivalents = rtme

    #========================================================================================================
    #======================================= FOUR-CIRCLE FUNCTIONS =====================================
    #========================================================================================================
    def get_angles_to_measure_hkl(self, h, k, l, wl_tolerance):
        """ Find the goniometer angles that will allow you to measure
        the given h,k,l. Only for Four-Circle instruments.

        Parameters:
            h,k,l: HKL of the reflection of interest
            wl_tolerance: error tolerance on the detection
                wavelength, in angstroms.

        Returns:
            angles: list of the goniometer angles that measure that HKL,
                    or None if no valid angles could be found.
            wavelength: actual measurement wavelength

        """

        #This UB matrix comes from the crystal data
        ub_matrix = self.crystal.ub_matrix

        #type inst InstrumentFourCircle
        inst = self.inst

        #@type g HB3AGoniometer
        g = inst.goniometer

        # These are the limits of detector angles
        det_angle_limits = g.gonio_angles[3].random_range;

        # hkl as a vector
        hkl = np.array([h, k, l])

        # Wavelength
        wl = self.inst.wl_input

        #There is a limited range of accessible detector angles = a limited
        # range of beam_wanted values. We need to find the one, if any
        # that measures as close as possible to the input wavelength



        def _error_in_wavelength(detector_angle, return_rot):
            """ Compute the wavelength error of finding the desired HKL
            with the detector at the given detector_angle.
            A low error means the wavelength of measurement is close to the
            input wavelength

            Parameters:
                detector_angle: in radians, your single-pixel detector rotation angle.
                return_rot: True if you want to return rot (the rotation matrix)
                            and wavelength, in addition to error.
                            Returns: rot, wavelength, error
            """
            # First, calculate the normal beam direction for this detector angle
            #  Detector is in the horizontal plane
            beam_wanted = np.array([sin(-detector_angle[0]), 0.0, cos(detector_angle[0])])
            # Use the utility function to get the rotation matrix and wavelength
            (rot, wavelength) = crystal_calc.get_sample_rotation_matrix_to_get_beam(beam_wanted, hkl, ub_matrix, starting_rot_matrix=None)
            # Error in wavelength
            error = np.abs(wavelength - wl)

            # increase error if you are off the range of the goniometer.
            diff_min = detector_angle - det_angle_limits[0]
            if diff_min < 0: error += np.abs(diff_min * 10)
            diff_max = detector_angle - det_angle_limits[1]
            if diff_max > 0: error += diff_max * 10

            #print "For angle", detector_angle, " i find it at WL ", wavelength, " giving an error of ", error

            if return_rot:
                return (rot, wavelength, error)
            else:
                return error

        #args = (initial_R, ending_vec, starting_vec)
        args = (False,)

        # Pick the middle of the detector range as a starting angle
        best_detector_angle = (det_angle_limits[0] + det_angle_limits[1])/2

        x0 = np.array([ best_detector_angle ])
        res = scipy.optimize.fmin(_error_in_wavelength, x0, args, xtol=1e-3, ftol=1e-3, disp=0, maxiter=150)
        best_detector_angle = res.reshape( (1) )[0] #avoid error with 0-dimension array
        #print "best_detector_angle", best_detector_angle

        # Call it again to get the rotation matrix
        (rot, best_wavelength, error) = _error_in_wavelength( [best_detector_angle], True)
        if (error > wl_tolerance):
            return (None, None)
        (phi,chi,omega) = numpy_utils.angles_from_rotation_matrix(rot)

        # The starting vector = an the q-vector of the unrotated hkl
        starting_vec = np.dot(ub_matrix, hkl)
        # But we want it to be rotated by this much, so it ends up pointing there vvv
        ending_vec = np.dot(rot, starting_vec)
        #print starting_vec, "->", ending_vec

        # Now we let the LimitedGoniometer class find possible phi chi omega
        #print "Goniometer angles were ", (phi,chi,omega)
        ret = g.calculate_angles_to_rotate_vector(starting_vec, ending_vec, starting_angles=None, search_method=0)
        if ret is None:
            return None, None
        else:
            (phi,chi,omega) = ret

        # These are the 4 goniometer angles
        angles = [phi,chi,omega, best_detector_angle]
        #print "Goniometer angles are  ", angles, "with wavelength", best_wavelength
        return (angles, best_wavelength)


    #-------------------------------------------------------------------------------
    def fourcircle_measure_all_reflections(self, progress_dialog=None):
        """ Will go through all the initialized reflections and find the goniometer angles,
        if any, that can measure them. The corresponding orientation for each
        hkl will be added to the experiment plan.

        Parameters:
            progress_dialog: handle to the wx.ProgressDialog object to report
                progress.
        """

        # Clear the list of positions
        self.inst.positions = []

        u_matrix = self.crystal.get_u_matrix()
        count = 0
        for ref in self.reflections:
            count += 1
            # Report progress
            if progress_dialog is None:
                print ".",
                if count % 50 == 0 : print
                pass
                #print "Looking for HKL %d %d %d" % (ref.h, ref.k, ref.l)
            else:
                (keep_going, skip) = progress_dialog.Update(count, "Looking for HKL %d %d %d" % (ref.h, ref.k, ref.l))
                # User can abort
                if not keep_going:
                    break

            #@type ref Reflection
            angles, wavelength = self.get_angles_to_measure_hkl(ref.h, ref.k, ref.l, 0.01)
            if not (angles is None):
                #Create a PositionCoverage object that holds both the position and the coverage
                pos = instrument.PositionCoverage(angles, np.array([]), sample_U_matrix=u_matrix)
                pos.comment = "HKL %d,%d,%d" %(ref.h, ref.k, ref.l)
                #Add it to the list.
                self.inst.positions.append(pos)

                #Add it to the measurement in ref
                poscov_id = len(self.inst.positions)-1
                # Only a single measurement in the list.
                ref.measurements = [ (id(pos), 0, 0, 0, wavelength, 1.0) ]

        #We make the array of how many times measured, for all the positions
        self.get_reflections_times_measured(None)

        #We make the array of how many times REALLY measured, for all the positions
        self.get_reflections_times_real_measured(None)

        #Continue on with masking
        self.calculate_reflections_mask()

        #And get some statistics
        self.calculate_reflection_coverage_stats()
        
        if progress_dialog is None: print "Done!"


    #========================================================================================================
    #======================================= EXPERIMENT PARAMETERS =====================================
    #========================================================================================================


    #-------------------------------------------------------------------------------
    def get_detectors_used(self):
        """Return a list of all the detectors to be used in the calculation.
        """
        #@type det_param ParamDetectors
        det_param = self.params[PARAM_DETECTORS]
        #None: use every detector
        if det_param is None:
            return self.inst.detectors
        #Go through the bool array, return only the True ones
        det_list = []
        for (i, use_this_one) in enumerate(det_param.detectors):
            if use_this_one and (i < len(self.inst.detectors)):
                det_list.append(self.inst.detectors[i])
        return det_list

    #-------------------------------------------------------------------------------
    def get_detectors_bool_array(self):
        """Return a bool array with True for all the detectors to use.
        """
        #@type det_param ParamDetectors
        det_param = self.params[PARAM_DETECTORS]
        #None: use every detector
        if det_param is None:
            return [True]*len(self.inst.detectors)
        #Incorrect parameter is saved? Use all.
        if len(det_param.detectors) != len (self.inst.detectors):
            #TODO: Output a warning maybe?
            return [True]*len(self.inst.detectors)
        else:
            return det_param.detectors


    #-------------------------------------------------------------------------------
    def get_positions_used(self, pos_param, also_try_position=True):
        """Return a list of each PositionCoverage object to consider in the
        coverage calculation.

        Parameters:
            pos_param: The ParamPositions object giving the ones to use, or None
                if we should use the global one.
            also_try_position: Also include the trial position.
        """
        #Retrieve the parameters
        if pos_param is None: pos_param = self.params[PARAM_POSITIONS]
        if pos_param is None:
            positions = dict() #Use an empty dictionary
        else:
            positions = pos_param.positions

        #The "try" position; only if the parameter is saved, and it says to use it.
        try_param = self.params[PARAM_TRY_POSITION]
        try_position = None
        if not try_param is None:
            if try_param.use_trial:
                try_position = try_param.try_position


        #Ok, we make the list of PositionCoverage objects to use.
        positions_used = list()
        #Go through the dictionary and put in all the True entries
        for i in range(len(self.inst.positions)):
            #Default to false if it is not found
            if positions.get(id(self.inst.positions[i]), False):
                positions_used.append ( self.inst.positions[i]  )
        #And add the "try" object, if it exists
        if also_try_position and not try_position is None:
            positions_used.append( try_position )

        return positions_used


    #========================================================================================================
    def _friendly_time(self, seconds):
        """Make a friendly string describing time in seconds/minutes/hours/days.
        2 days and 03h 45m 22s
        """
        try:
            if seconds < 0:
                return "%d seconds" % seconds

            m, s = divmod(seconds, 60.)
            h, m = divmod(m, 60.)
            d, h = divmod(h, 24.)
            w, d = divmod(d, 7.)
            y, w = divmod(w, 52.) #Approximately!
            ages, y = divmod(y, 14e9) #age of the universe
            if ages>0:
                return "%d age%s of the universe, %d year%s, %d week%s, %d day%s and %0dh %02dm %02ds" % (ages,  ['', 's'][ages>0], y,  ['', 's'][y>0], w,  ['', 's'][w>0], d,  ['', 's'][d>0],  h, m, s)
            elif y>0:
                return "%d year%s, %d week%s, %d day%s and %0dh %02dm %02ds" % (y,  ['', 's'][y>0], w,  ['', 's'][w>0], d,  ['', 's'][d>0],  h, m, s)
            elif w>0:
                return "%d week%s, %d day%s and %0dh %02dm %02ds" % (w,  ['', 's'][w>0], d,  ['', 's'][d>0],  h, m, s)
            elif d>0:
                return "%d day%s and %0dh %02dm %02ds" % (d,  ['', 's'][d>0],  h, m, s)
            elif h>0:
                return "%dh %02dm %02ds" % (h, m, s)
            else:
                return "%02dm %02ds" % (m, s)
        except:
            return "unknown time"


    #========================================================================================================
    def estimated_time_string(self):
        """Return a string describing an estimate of the time to perform the experiment, given
        the selected positions."""
        positions_used = self.get_positions_used(None)
        s = ""
        #Total time and charge
        time = 0.
        pcharge = 0.
        other = 0.
        for poscov in positions_used:
            # @type poscov PositionCoverage
            if not poscov is None:
                if poscov.criterion == "runtime":
                    time += poscov.criterion_value
                elif poscov.criterion == "pcharge":
                    pcharge += poscov.criterion_value
                else:
                    other += poscov.criterion_value

        if time > 0:
            s = self._friendly_time(time)
        if pcharge > 0:
            if len(s)>0: s += "; and "
            # SNS parameters says 1.5e14 protons per pulse, at full power of 1.4 MW
            pC_per_second_at_100kw = (1.5e14 * 60 / 1.4) * 1.602177e-19 * 1e12
            s += "%.3e pC proton charge, approx %s (at 1 MW accelerator power)." % (pcharge,  self._friendly_time(pcharge/pC_per_second_at_100kw) )
        if other > 0:
            if len(s)>0: s += "; and an "
            s += "unknown time to accumulate the given # of counts."
        #String was built up, make sure there is something though
        if s == "":
            s = "0 seconds."
        return s

    #-------------------------------------------------------------------------------
    def save_sample_orientations_to_CSV_file(self, filename):
        """Save the list of currently checked sample orientations to a CSV file."""
        #List of positions
        pos_list = self.get_positions_used(None, also_try_position=False)
        g = self.inst.goniometer
        #Start CSV file
        f = open(filename, "w")
        g.csv_make_header(f, self.crystal.name, self.crystal.description)
        #Go to some angle
        for (i, pos) in enumerate(pos_list):
            if not pos is None:
                g.csv_add_position(f, pos.angles, pos.criterion, pos.criterion_value, pos.comment)
        #Save the file
        f.close()

    #========================================================================================================
    #======================================= VOLUME COVERAGE CALCULATIONS =====================================
    #========================================================================================================
            
    #-------------------------------------------------------------------------------
    def calculate_coverage(self, pos_param=None, det_param=None):
        """Recalculate the full coverage map. Unless parameters are passed to the function,
        it will use the saved parameters in self.params[].
            pos_param: a ParamPositions object specifing which positions to use. None to use the global one.
            det_param: a ParamDetectors object specifing which detectors to use. None to use the global one.
        """
        if self.inst is None:   
            warnings.warn("experiment.calculate_coverage(): called with experiment.inst == None.")
            return

        #Detectors
        if det_param is None: det_param = self.params[PARAM_DETECTORS]
        if det_param is None:
            detectors = None
        else:
            detectors = det_param.detectors

        #Make the list of positions
        positions_used = self.get_positions_used(pos_param)

        #Have the instrument calculate the total coverage.
        if isinstance(self.inst, instrument.InstrumentInelastic):
            #---- Inelastic calculation ---
            slice = self.params[PARAM_ENERGY_SLICE] #@type slice ParamSlice
            if slice is None:
                #Use default slice amounts
                print "Warning for exp.calculate_coverage(): Energy slice parameter not set. Using defaults."
                self.qspace = self.inst.total_coverage(detectors, positions_used)
            else:
                #Use the set parameter
                print "Doing energy slice of", slice.slice_min, " to ", slice.slice_max
                self.qspace = self.inst.total_coverage(detectors, positions_used, slice_min=slice.slice_min, slice_max=slice.slice_max)
        elif isinstance(self.inst, instrument.InstrumentFourCircle):

            # Don't calculate 3D coverage for four-circles
            self.qspace = np.array([])
            return

        else:
            #---- Elastic calculation ----
            self.qspace = self.inst.total_coverage(detectors, positions_used)

        #Continue processing sequentially
        self.symmetry_coverage()



    #-------------------------------------------------------------------------------
    def initialize_volume_symmetry_map(self):
        """Create a map for each q-space voxel that maps it to equivalent symmetry elements"""
        #@type pg PointGroup
        pg = self.crystal.get_point_group()
        if pg is None:
            print "ERROR!"
            return

        t1 = time.time()

        order = len(pg.table)
        #@type inst Instrument
        inst = self.inst

        #Initialize the symmetry map. Last dimension = the ORDER equivalent indices
        n = len(inst.qx_list)
        numpix = n**3
        symm = np.zeros( (numpix, order) , dtype=int)

        if self.verbose: print "Starting volume symmetry calculation. Order is %d. Matrix is %d voxels (%d to a side)." % (order, n**3, n)

        #--- From get_hkl_from_q functions: (moved here for speed) --
        #Get the inverse the B matrix to do the reverse conversion
        B = self.crystal.get_B_matrix()
        invB = np.linalg.inv(B)

        #Limit +- in q space
        qlim = inst.qlim
        
        if config.cfg.force_pure_python:
            #----------- Pure Python Version --------------

            #Go through each pixel
            q_arr = np.zeros( (3, numpix) )
            for (ix, qx) in enumerate(inst.qx_list):
                for (iy, qy) in enumerate(inst.qx_list):
                    for (iz, qz) in enumerate(inst.qx_list):
                        i = iz + iy*n + ix*n*n
                        #Find the (float) HKL of this voxel at qx,qy,qz.
                        q_arr[:, i] = (qx,qy,qz)

            #Matrix multiply invB.hkl to get all the HKLs as a column array
            hkl = np.dot(invB, q_arr)

            #Now get ORDER equivalent HKLs, as a long list.
            #(as equivalent q)
            q_equiv = np.zeros( (3, numpix, order) )
            for ord in xrange(order):
                #Ok, we go TABLE . hkl to get the equivalent hkl
                #Them, B . hkl gives you the Q vector
                q_equiv[:,:, ord] =  np.dot(B,   np.dot(pg.table[ord], hkl) )

                #Now we need to find the index into the array.
                #Start by finding the x,y,z, indices
                ix = numpy_utils.index_array_evenly_spaced(-qlim, n, inst.q_resolution, q_equiv[0, :, ord])
                iy = numpy_utils.index_array_evenly_spaced(-qlim, n, inst.q_resolution, q_equiv[1, :, ord])
                iz = numpy_utils.index_array_evenly_spaced(-qlim, n, inst.q_resolution, q_equiv[2, :, ord])

                #Now put the index into the symmetry matrix
                index = iz + iy*n + ix*n*n
                index[np.isnan(index)] = -1 #Put -1 where a NAN was found
                symm[:, ord] = index


        else:
            #--------------- Inline C version (about 17x faster than Python) ---------------
            code = """
            //-- Calculate  the hkl array ---
            int ix, iy, iz;
            int eix, eiy, eiz, eindex;
            int index, ord;
            double qx, qy, qz;
            double eqx, eqy, eqz;
            double h, k, l;
            double eh, ek, el;
            for (ix=0; ix<n; ix++)
            {
                qx = ix*qres - qlim;
                for (iy=0; iy<n; iy++)
                {
                    qy = iy*qres - qlim;
                    for (iz=0; iz<n; iz++)
                    {
                        qz = iz*qres - qlim;
                        index = iz + iy*n + ix*n*n;
                        //Ok, now we matrix multiply invB.hkl to get all the HKLs as a column array
                        h = qx * INVB2(0,0) + qy * INVB2(0,1) + qz * INVB2(0,2);
                        k = qx * INVB2(1,0) + qy * INVB2(1,1) + qz * INVB2(1,2);
                        l = qx * INVB2(2,0) + qy * INVB2(2,1) + qz * INVB2(2,2);

                        //Now go through each equivalency table.
                        for (ord=0; ord<order; ord++)
                        {
                            //Do TABLE.hkl to find a new equivalent hkl
                            eh = h * TABLE3(ord, 0,0) + k * TABLE3(ord, 0,1) + l * TABLE3(ord, 0,2);
                            ek = h * TABLE3(ord, 1,0) + k * TABLE3(ord, 1,1) + l * TABLE3(ord, 1,2);
                            el = h * TABLE3(ord, 2,0) + k * TABLE3(ord, 2,1) + l * TABLE3(ord, 2,2);
                            //Now, matrix mult B . equiv_hkl to get the other q vector
                            eqx = eh * B2(0,0) + ek * B2(0,1) + el * B2(0,2);
                            eqy = eh * B2(1,0) + ek * B2(1,1) + el * B2(1,2);
                            eqz = eh * B2(2,0) + ek * B2(2,1) + el * B2(2,2);

                            //Ok, now you have to find the index into QSPACE
                            eix = round( (eqx+qlim)/qres ); if ((eix >= n) || (eix < 0)) eix = -1; 
                            eiy = round( (eqy+qlim)/qres ); if ((eiy >= n) || (eiy < 0)) eiy = -1;
                            eiz = round( (eqz+qlim)/qres ); if ((eiz >= n) || (eiz < 0)) eiz = -1;

                            if ((eix < 0) || (eiy < 0) || (eiz < 0))
                            {
                                //One of the indices was out of bounds.
                                //Put this marker to mean NO EQUIVALENT
                                SYMM2(index, ord) = -1;
                            }
                            else
                            {
                                //No problem!, Now I put it in there
                                eindex = eiz + eiy*n + eix*n*n;
                                //This pixel (index) has this equivalent pixel index (eindex) for this order transform ord.
                                SYMM2(index, ord) = eindex;
                            }

                        }
                        
                    }
                }
            }"""
            qres = inst.q_resolution
            n = len(self.inst.qx_list)
            table = np.array(pg.table) #Turn the list of 3x3 arrays into a Nx3x3 array
            varlist = ['B', 'invB', 'symm', 'qres', 'qlim', 'n', 'order', 'table']
            weave.inline(code, varlist, compiler='gcc', support_code="")

        #Done with either version
        self.volume_symmetry = symm

        if self.verbose: print "Volume symmetry map done in %.3f sec." % (time.time()-t1)


        
    #-------------------------------------------------------------------------------
    def use_symmetry(self):
        """Utility functions returns True if the parameters say to use crystal symmetry"""
        symmetry = self.params[PARAM_SYMMETRY]
        if symmetry is None:
            #Default to false if no parameter.
            return False
        else:
            return symmetry.use_symmetry

    #-------------------------------------------------------------------------------
    def symmetry_coverage(self):
        """If the parameters require, use the crystal symmetry"""
        if self.inst is None:
            warnings.warn("experiment.symmetry_coverage(): called with experiment.inst == None.")
            return
        
        if self.use_symmetry():
            #We do the symmetry application here
            self.apply_volume_symmetry()

        #Adjust qspace using the full sphere (cut all outside the qlim range)
        self.qspace = self.qspace * (self.inst.qspace_radius < self.inst.qlim)

        #Continue processing sequentially. Invert?
        self.invert_coverage()

        #Now is the time to calculate some stats
        self.calculate_coverage_stats()


    #-------------------------------------------------------------------------------
    def apply_volume_symmetry(self, use_inline_c=True):
        """Applies the volume symmetry, in place, to self.qspace"""
        t1 = time.time()

        #Get the # of pixels and the order from the symmetry map
        symm = self.volume_symmetry
        (numpix, order) = symm.shape

        if use_inline_c and not config.cfg.force_pure_python:
            #------ C version (about 400x faster than python) -------
            #Put some variables in the workspace
            old_q = self.qspace.flatten() * 1.0
            qspace_flat = old_q * 0.0

            support = ""
            code = """
            int pix, ord, index;
            for (pix=0; pix<numpix; pix++)
            {
                //Go through each pixel
                for (ord=0; ord<order; ord++)
                {
                    //Now go through each equivalent q.
                    index = SYMM2(pix, ord);
                    if (index >= 0)
                    {
                        //Valid index.
                        QSPACE_FLAT1(pix) += OLD_Q1(index);
                        //printf("%d\\n", index);
                    }
                }
            }"""
            varlist = ['old_q', 'qspace_flat', 'numpix', 'order', 'symm']
            weave.inline(code, varlist, compiler='gcc', support_code=support)
            #Reshape it back as a 3D array.
            n = len(self.inst.qx_list)
            self.qspace = qspace_flat.reshape( (n,n,n) )
        else:
            #---- Pure python version ----

            #Clear the starting space
            old_q = self.qspace
            new_q = self.qspace * 0
            for pix in xrange(numpix):
                for ord in xrange(order):
                    eq_index = symm[pix, ord]
                    if eq_index >= 0:
                        #Add up to this pixel, the equivalent one.
                        #The list includes this given voxel too.
                        new_q.flat[pix] += old_q.flat[eq_index]
            self.qspace = new_q

        #Done.
        if self.verbose: print "Volume symmetry computed in %.3f sec." % (time.time()-t1)



    
    #-------------------------------------------------------------------------------
    def invert_coverage(self):
        """Invert the coverage map, showing the gaps in coverage, if requested 
        by the parameters.
        Called after calculate_coverage()."""
        if self.inst is None: 
            warnings.warn("experiment.invert_coverage(): called with experiment.inst == None.")
            return
        
        invert = self.params[PARAM_INVERT]
        if invert is None:
            #Don't invert if no setting is saved.
            do_invert = False
        else:
            do_invert = invert.invert
            
        self._lock_qspace_displayed.acquire()
        if do_invert:
            #Ok, we invert, and account for the sphere that fits in the box
            self.qspace_displayed = 1.0*(self.qspace == 0) * (self.inst.qspace_radius < self.inst.qlim)
        else:
            #Or we don't
            self.qspace_displayed = self.qspace.copy()
        self._lock_qspace_displayed.release()
        
        #Continue processing
        self.slice_coverage()
                    
                    
    #-------------------------------------------------------------------------------
    def slice_coverage(self):
        """Do a slice of the q-space coverage from qmin to qmax.
        Called after invert_coverage().
        """
        if self.inst is None: 
            warnings.warn("experiment.slice_coverage(): called with experiment.inst == None.")
            return

        t1 = time.time()
 
        #Find the slice amounts saved
        slice = self.params[PARAM_SLICE]
        if slice is None:
            slice = ParamSlice(False) #Create a "no-slice" slice
            
        #Get the limits to slice to
        qmin = slice.slice_min
        qmax = slice.slice_max

        #The raw qspace was already clipped to the sphere, so if there is no slice, you don't do anything.
        if slice.use_slice:
            #Does a quick slice by using the radius array
            self._lock_qspace_displayed.acquire()
            if qmin <= 0:
                #Speed up a bit
                self.qspace_displayed = self.qspace_displayed * (self.inst.qspace_radius < qmax)
            else:
                self.qspace_displayed = self.qspace_displayed * ((self.inst.qspace_radius < qmax) & (self.inst.qspace_radius > qmin))
            self._lock_qspace_displayed.release()

        #Finish processing
        self.finalize_displayed()
        
    #-------------------------------------------------------------------------------
    def finalize_displayed(self):
        """Function to ensure that the displayed q-space shows up properly in mayavi."""

        self._lock_qspace_displayed.acquire()
        # We null all the 6 faces of the cube. This makes it possible 
        # for the contour3d function to plot domains adjacent to the edge.
        self.qspace_displayed[0,:,:]=0
        self.qspace_displayed[:,0,:]=0
        self.qspace_displayed[:,:,0]=0
        self.qspace_displayed[-1,:,:]=0
        self.qspace_displayed[:,-1,:]=0
        self.qspace_displayed[:,:,-1]=0

        #Keep a point in each corner. This helps the display stay valid even if the data is empty.
        self.qspace_displayed[ 0, 0, 0] = 1
        self.qspace_displayed[-1, 0, 0] = 1
        self.qspace_displayed[ 0,-1, 0] = 1
        self.qspace_displayed[ 0, 0,-1] = 1
        self.qspace_displayed[-1,-1, 0] = 1
        self.qspace_displayed[ 0,-1,-1] = 1
        self.qspace_displayed[-1, 0,-1] = 1
        self.qspace_displayed[-1,-1,-1] = 1

        self._lock_qspace_displayed.release()
        
    #-------------------------------------------------------------------------------
    def get_qspace_displayed(self):
        """Returns a reference to the qspace_displayed 3d matrix.
        Creates it if needed.
        Should be thread safe."""
        if self.qspace_displayed is None:
            #Try to create an empty array for 
            if self.qspace is None:
                if self.inst is None:
                    raise StandardError("Experiment.get_qspace_displayed() called before experiment.inst was initialised.")
                else:
                    #Make sure the q-space limits and stuff are initialized
                    self.inst.make_qspace()
                    #Make an empty one of right data size.
                    self.qspace = self.inst.make_blank_qspace(np.int16)
            #Create a dummy qspace
            self._lock_qspace_displayed.acquire()
            self.qspace_displayed = self.qspace.copy()
            self._lock_qspace_displayed.release()
                
        #At this point we should have a valid qspace_displayed
        return self.qspace_displayed



    #========================================================================================================
    #======================================= COVERAGE STATS =====================================
    #========================================================================================================

        
    #-------------------------------------------------------------------------------
    def calculate_coverage_stats(self):
        """Calculate coverage statistics for the current qspace data."""
        if self.inst is None: return

        t1 = time.time()
        qlim = self.inst.qlim
        
        #Number of slices to make
        num = 30
        q_step = qlim/num
        
        #Initialize the list
        self.coverage_stats = list()

        if config.cfg.force_pure_python:
            #Overall calcs
            (self.overall_coverage, self.overall_redundancy) = self.overall_coverage_stats(self.qspace)

            #--- Python version ---
            for i in range(num):
                #Initialize the CoverageStats object
                qmin = i*q_step
                qmax = qmin + q_step
                covstat = CoverageStats(qmin, qmax)

                bool_array = ((self.inst.qspace_radius < qmax) & (self.inst.qspace_radius > qmin))
                #Calculate the slice
                points_in_slice = np.sum(bool_array)
                slice = (self.qspace * bool_array)
                for k in range(3):
                    #Calculate the coverage where you measure the point more than i times.
                    covstat.coverage[k] = 100.0 * np.sum( (slice > k) ) / points_in_slice

                #Append to list
                self.coverage_stats.append(covstat)
        else:
            #--- Inline C version ----
            #   about 40x faster than the python version.

            #Parameters to be passed.
            covered_points0 = np.zeros(num)
            covered_points1 = np.zeros(num)
            covered_points2 = np.zeros(num)
            covered_points3 = np.zeros(num)
            total_points = np.zeros(num)
            qspace_radius = self.inst.qspace_radius.flatten()
            qspace = self.qspace.flatten()
            qspace_size = qspace.size
            
            support = ""
            code = """
            int i, j;
            int slice;
            int val;
            int overall_points = 0;
            int overall_covered_points = 0;
            int overall_redundant_points = 0;
            
            for (i=0; i<qspace_size; i++)
            {
                //Coverage value at this points
                val = QSPACE1(i);

                //Do the overall stats
                if (QSPACE_RADIUS1(i) < qlim)
                {
                    //But only within the sphere
                    overall_points++;
                    if (val > 0)
                    {
                        overall_covered_points++;
                        if (val > 1)
                        {
                            overall_redundant_points++;
                        }
                    }
                }

                //Which slice are we looking at?
                slice = QSPACE_RADIUS1(i) / q_step;
                if ((slice < num) && (slice >= 0))
                {
                    total_points[slice]++;
                    if (val>0)
                    {
                        covered_points0[slice]++;
                        if (val>1)
                        {
                            covered_points1[slice]++;
                            if (val>2)
                            {
                                covered_points2[slice]++;
                                if (val>3)
                                {
                                    covered_points3[slice]++;
                                }
                            }
                        }
                    }
                }
            }
            //Return as a tuple
            py::tuple results(3);
            results[0] = overall_points;
            results[1] = overall_covered_points;
            results[2] = overall_redundant_points;
            return_val = results;"""
            ret_val = weave.inline(code,['qspace', 'qspace_radius', 'q_step', 'qlim', 'total_points', 'qspace_size', 'num', 'covered_points0', 'covered_points1', 'covered_points2', 'covered_points3'],
                                compiler='gcc', support_code = support)
            #The function returns a tuple
            (overall_points, overall_covered_points, overall_redundant_points) = ret_val

            #Overall stats
            self.overall_coverage = 100.0 * overall_covered_points / overall_points;
            self.overall_redundancy = 100.0 * overall_redundant_points / overall_points;

            for i in range(num):
                #Initialize the CoverageStats object
                qmin = i*q_step
                qmax = qmin + q_step
                covstat = CoverageStats(qmin, qmax)
                covstat.coverage[0] = 100.0 * covered_points0[i]/total_points[i]
                covstat.coverage[1] = 100.0 * covered_points1[i]/total_points[i]
                covstat.coverage[2] = 100.0 * covered_points2[i]/total_points[i]
                covstat.coverage[3] = 100.0 * covered_points3[i]/total_points[i]
                #Append to list
                self.coverage_stats.append(covstat)

        #if self.verbose: print "calculate_coverage_stats took %s sec." % (time.time()-t1)


    #-------------------------------------------------------------------------------
    def overall_coverage_stats(self, qspace):
        """Given a 3D qspace coverage array, calculate the overall coverage % of a sphere.
        
        Parameters:
            qspace: 3D array of coverage; points should be limited to a sphere of the same size.
            
        Returns a tuple with the (coverage, redundancy) as percentage."""
        if self.inst is None: return
        t1 = time.time()
        #Add up the points
        points_in_sphere = np.sum((self.inst.qspace_radius < self.inst.qlim))
        covered_in_sphere = np.sum( qspace > 0)
        redundant_in_sphere = np.sum( qspace > 1)
        coverage = 100.0 * covered_in_sphere/points_in_sphere
        if covered_in_sphere > 0:
            redundant = 100.0 * redundant_in_sphere/covered_in_sphere
        else:
            redundant = 0
        if self.verbose: print "overall_coverage_stats took %s sec." % (time.time()-t1)
        return (coverage, redundant)


    #-------------------------------------------------------------------------------
    def calculate_reflection_coverage_stats(self, edge_avoidance=False, edge_x=0, edge_y=0):
        """Counts the number of reflections that have been measured, with or without considering
        crystal symmetry.
        """
        if self.inst is None: return
        #Start without symmetry
        self.reflection_stats.total = len(self.reflections)
        self.reflection_stats.measured = np.sum(self.reflections_times_measured > 0)
        self.reflection_stats.redundant = np.sum(self.reflections_times_measured > 1)
        #Now do it with symmetry
        assert len(self.primary_reflections_mask)==len(self.reflections), "The primary reflections mask should be the same length as the reflections; otherwise, incorrect stats will be computed."
        assert len(self.reflections_times_measured_with_equivalents)==len(self.reflections), "reflections_times_measured_with_equivalents is the right length."
        mask = self.primary_reflections_mask
        if len(mask) > 0:
            self.reflection_stats_with_symmetry.total = np.sum(mask)
            self.reflection_stats_with_symmetry.measured = np.sum(self.reflections_times_measured_with_equivalents[mask,:] > 0)
            self.reflection_stats_with_symmetry.redundant = np.sum(self.reflections_times_measured_with_equivalents[mask,:] > 1)

        #Stats using edge avoidance
        if edge_avoidance:
            self.calculate_reflection_coverage_stats_adjusted(edge_x, edge_y)

        return None

    #-------------------------------------------------------------------------------
    def calculate_reflection_coverage_stats_adjusted(self, edge_x=0, edge_y=0):
        """Adjusts the coverage stats to take into account edge avoidance.
        """
        if self.inst is None: return
        detectors = self.inst.detectors

        #Total to add up
        rsa = 0
        rsaws = 0

        self.reflection_stats_adjusted.total = len(self.reflections)
        self.reflection_stats_adjusted_with_symmetry.total = np.sum(self.primary_reflections_mask)
        #assert self.reflection_stats_adjusted_with_symmetry.total <= len(self.reflections), "Can't have more primary reflections than reflections."
        #@type refl Reflection
        for refl in self.reflections:
            mav = []
            mysum = 0

            if len(refl.measurements) > 0:
                for meas in refl.measurements:
                    (poscov_id, detector_num, horizontal, vertical, wavelength, distance) = meas
                    mav.append( detectors[detector_num].edge_avoidance(horizontal, vertical, edge_x, edge_y) )
                #Sum it
                mysum = np.sum(np.array(mav) > 0.99)
                
            refl.measurement_adjusted_sum = mysum
            rsa += (mysum > 0)
            refl.measurement_adjusted_value = mav

        #Now handle symmetry
        mask = self.primary_reflections_mask
        for refl in self.reflections:
            if refl.is_primary:
                anything = False
                for equiv in refl.equivalent:
                    if (equiv.measurement_adjusted_sum > 0):
                        anything = True
                        break
                if anything:
                    rsaws += 1


        #Save 'em
        self.reflection_stats_adjusted.measured = rsa
        self.reflection_stats_adjusted_with_symmetry.measured = rsaws
        assert self.reflection_stats_adjusted_with_symmetry.measured <= self.reflection_stats_adjusted_with_symmetry.total, "Can't have more measured reflections than primary reflections."

        return None

            
    #-------------------------------------------------------------------------------
    def get_coverage_stats_data(self):
        """Return the coverage stats as numpy arrays.
            Return: (coverage_q, coverage_data)
                coverage_q: numpy array of |q| of the point
                coverage_data: list of numpy arrays containging the coverage (once, twice, etc.)
        """
        coverage_data = list()

        if (len(self.coverage_stats) == 0):
            #Nothing in the coverage data. We generate some fake coverage for debugging purposes.
            coverage_q = np.linspace(0, 8, 10)
            if False:
                #Debugging and testing mode
                y = 100*np.random.rand(10)
            else:
                y = coverage_q * 0.
            coverage_data.append( y )
            coverage_data.append( y / 2 )
            coverage_data.append( y / 4 )
            return (coverage_q, coverage_data)

        #Do the x-axis
        coverage_q = np.zeros(len(self.coverage_stats))
        for i in range(len(self.coverage_stats)):
            coverage_q[i] = (self.coverage_stats[i].qmin + self.coverage_stats[i].qmax) / 2
        #Now all the y axes
        for redundancy_num in range(4):
            y = np.zeros(len(self.coverage_stats))
            for i in range(len(self.coverage_stats)):
                coverage_q[i] = (self.coverage_stats[i].qmin + self.coverage_stats[i].qmax) / 2
                y[i] = self.coverage_stats[i].coverage[redundancy_num]
            coverage_data.append( y )
            
        #Return as tuple
        return (coverage_q, coverage_data)


    #========================================================================================================
    #======================================= PEAKS FILE LOADING =====================================
    #========================================================================================================

    #---------------------------------------------------------------------------------------------
    def load_peaks_file(self, filename, append=False, sequential_detector_numbers=False):
        """Loads a .peaks or .integrate file (made by ISAW) into the program, adding on
        to each reflection.

        Parameters:
            filename: path to .peaks or .integrate
            append: add on to the measurements; otherwise, they get cleared
            sequential_detector_numbers :: detector numbers are from 1 to X in order. 
                This is the old format (before April 2011).
        """
        if not os.path.exists(filename):
            raise IOError("The file %s does not exist!" % filename)

        print "Loading ISAW peaks/integrate file %s." % filename

        errors = []
        angles = [0,0,0]
        poscov = None

        #Clear them?
        if not append:
            self.clear_reflection_real_measurements()

        f = open(filename,'r')
        #@type line string
        for line in f:
            try:
                #Detector number marker
                if line.startswith("1"):
                    arr = line.split()
                    
                    #Find the detector number
                    detnum = int(arr[2])
                    if (sequential_detector_numbers):
                        # make it 0-based
                        detnum = detnum - 1
                        if detnum>=0 and detnum < len(self.inst.detectors):
                            det = self.inst.detectors[detnum] 
                        else:
                            print "Warning: Detector number", detnum, "was not found!"
                    else:
                        det = None
                        detname = "%d" % detnum
                        # Use the detector number as the name
                        for i in xrange(len(self.inst.detectors)):
                            if self.inst.detectors[i].name == detname:
                                # Point to the detector
                                det = self.inst.detectors[i]
                                # You still need the detector "number" aka its index.
                                detnum = i
                                break
                        if det is None:
                            print "Warning: Detector number", detnum, "was not found!"
                    

                    #Find the angles, convert to radians
                    (chi, phi, omega) =  [np.deg2rad(float(arr[i])) for i in xrange(3,6)]
                    angles = np.array( [phi, chi, omega] )


                #Marker that this is a peak line
                if line.startswith("3"):
                    arr = line.split()
                    (h,k,l) = [int(arr[i]) for i in xrange(2,5)]

                    #Was the peak indexed (hkl is not 0,0,0)?
                    if not ((h,k,l) == (0,0,0)):
                        #@type ref Reflection
                        ref = self.get_reflection(h, k, l)

                        if ref is None:
                            errors.append("Peak was not in the list: Det %d, hkl: %d, %d, %d\n" % (detnum, h, k, l))
                        else:
                            #Ok, let's record this info in the same Reflection.
                            rm = ReflectionRealMeasurement()
                            rm.angles = angles * 1.0 #Make sure to copy the numpy array
                            rm.detector_num = detnum
                            col = float(arr[5])
                            row = float(arr[6])
                            #Convert to horizontal 
                            if not det is None:
                                #@type det FlatDetector
                                rm.horizontal = det.width * (det.xpixels/2-col)/det.xpixels
                                rm.vertical = -det.height * (det.ypixels/2-row)/det.ypixels

                            rm.wavelength = float(arr[11])
                            rm.integrated = float(arr[14])
                            rm.sigI = float(arr[15])
                            rm.measurement_num = int(arr[1]) #The peak #

                            #Add it
                            ref.real_measurements.append(rm)
            except IOError:
                raise
            except:
                #Ignore errors and keep loading the file
                pass
                        
        return errors


    #---------------------------------------------------------------------------------------------
    def load_HFIR_peaks_file(self, filename, append=False):
        """Loads a .int (made by HFIR software) into the program, adding on
        to each reflection.

        Parameters:
            filename: path to file
            append: add on to the measurements; otherwise, they get cleared
        """
        if not os.path.exists(filename):
            raise IOError("The file %s does not exist!" % filename)

        print "Loading HFIR .int file %s." % filename

        errors = []
        angles = [0,0,0]
        poscov = None

        #Clear them?
        if not append:
            self.clear_reflection_real_measurements()

        f = open(filename,'r')
        #First two lines are ignored
        title = f.readline()
        line2 = f.readline()

        #3rd line is the wavelength
        line = f.readline()
        arr = line.split()
        wavelength = float(arr[0])
        print "Wavelength was %f Angstroms." % wavelength

        count = 0

        #@type line string
        for line in f:
            try:
                arr = line.split()
                (h,k,l) = [int(arr[i]) for i in xrange(0,3)]

                #Was the peak indexed (hkl is not 0,0,0)?
                if not ((h,k,l) == (0,0,0)):
                    #@type ref Reflection
                    ref = self.get_reflection(h, k, l)

                    if ref is None:
                        errors.append("Peak was not in the list: Det %d, hkl: %d, %d, %d\n" % (detnum, h, k, l))
                    else:
                        #Ok, let's record this info in the same Reflection.
                        rm = ReflectionRealMeasurement()
                        rm.angles = np.array([0,0,0])
                        rm.detector_num = 0
                        #@type det FlatDetector
                        rm.horizontal = 0
                        rm.vertical = 0

                        rm.wavelength = wavelength
                        rm.integrated = float(arr[3])
                        rm.sigI = float(arr[4])
                        rm.measurement_num = count
                        count += 1

                        #Add it
                        ref.real_measurements.append(rm)
                        
            except IOError:
                raise
            except:
                #Ignore errors and keep loading the file
                pass

        return errors




    #---------------------------------------------------------------------------------------------
    def clear_reflection_real_measurements(self):
        """Remove all entries of "real" measurements from the reflections."""
        #@type ref Reflection
        for ref in self.reflections:
            ref.real_measurements = []

    #---------------------------------------------------------------------------------------------
    def reload_peaks_files(self):
        """Reload the files that were recorded before. Used when re-initializing
        crystal reflections."""
        self.clear_reflection_real_measurements()
        #Reload each one
        for filename in self.real_measurement_filenames:
            if not os.path.exists(filename):
                print "Error loading peaks file: %s. File does not exist!" % filename
            else:
                self.load_peaks_file(filename, append=True)

        #Now, redo the stats
        self.get_reflections_times_real_measured(None)

    #---------------------------------------------------------------------------------------------
    def compare_to_peaks_file(self, filename):
        """Compare the contents of an ISAW .peaks file with the measurements in this experiment."""
        if not os.path.exists(filename):
            raise IOError("The file %s does not exist!" % filename)

        out = ""

        f = open(filename,'r')
        detnum = -1
        numbad = 0
        numgood = 0
        #@type line string
        for line in f:
            #Detector number marker
            if line.startswith("1"):
                arr = line.split()
                #Find the detector number, make it 0-based
                detnum = int(arr[2]) - 1

            #Marker that this is a peak line
            if line.startswith("3"):
                arr = line.split()
                (h,k,l) = [int(arr[i]) for i in xrange(2,5)]

                #Was the peak indexed (hkl is not 0,0,0)?
                if not ((h,k,l) == (0,0,0)):
                    #@type ref Reflection
                    ref = self.get_reflection(h, k, l)

                    if ref is None:
                        out += "Peak was not in the list: Det %d, hkl: %d, %d, %d\n" % (detnum, h, k, l)
                        numbad += 1
                    else:
                        if ref.times_measured() <= 0:
                            out += "Measured peak was not calculated to be measured: Det %d, hkl: %d, %d, %d\n" % (detnum, h, k, l)
                            numbad += 1
                        else:
                            #Check that its on the right detector
                            calc_detnum = ref.measurements[0][1]
                            if calc_detnum == detnum:
                                #Correct
#                                out += "Good!: Det %d, hkl: %d, %d, %d\n" % (detnum, h, k, l)
                                numgood += 1
                                pass
                            else:
                                out += "Wrong detector. Expected %d (%s): Peaks file says detector # %d, hkl: %d, %d, %d\n" % \
                                        (calc_detnum, self.inst.detectors[calc_detnum].name, detnum, h, k, l)
                                numbad += 1


        return (numbad, numgood, out)





#================================================================================
#============================ LOADING AND SAVING ======================================
#================================================================================

#-------------------------------------------------------------------------------
def save_to_file(the_experiment, filename):
    """Save (pickle) the experiment to a file. Only the necessary data is
    saved, the rest will be recalculated upon loading."""
    #Pickle dumps
    datas = dumps(the_experiment)
    f = open(filename, 'w')
    f.write(datas)
    f.close()

#-------------------------------------------------------------------------------
def load_from_file(filename):
    """Loads an experiment from a previously-saved file."""
    f = open(filename, 'r') #@type f file
    datas = f.read()
    f.close()
    #working with old files
    datas=datas.replace('enthought.','')
    the_experiment = loads(datas)
    return the_experiment


#========================================================================================================
exp = None


#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest
import time
import copy

#==================================================================
class TestExperiment(unittest.TestCase):

    def setUp(self):
        instrument.inst = instrument.Instrument("../instruments/TOPAZ_geom_all_2011.csv")
        instrument.inst.set_goniometer(goniometer.Goniometer() )
        self.exp = Experiment(instrument.inst)
        e = self.exp
        e.inst.d_min = 0.5
        e.crystal.lattice_lengths = (1.0, 1.0, 1.0)
        e.crystal.lattice_angles_deg = (90.0, 90.0, 90.0)
        #UB and reciprocal
        e.crystal.make_ub_matrix()
        e.crystal.calculate_reciprocal()

    def test_constructor(self):
        """Experiment.__init__()"""
        e = self.exp

    def setup_reflections(self):
        e = self.exp
        e.range_h = (-1,2) #4 items
        e.range_k = (-2,3) #6 items
        e.range_l = (-3,4) #8 items
        e.range_automatic = False
        e.range_limit_to_sphere = False
        e.initialize_reflections()

    def test_initialize_reflections(self):
        """Experiment.initialize_reflections()"""
        self.setup_reflections()
        e = self.exp
        n = 4*6*8
        assert len(e.reflections) == (n), "reflections correct length"
        assert e.reflections_hkl.shape == (3, n), "hkl correct shape"
        assert e.reflections[0].hkl == (-1, -2, -3), "First reflection okay."
        assert e.reflections[1].hkl == (-1, -2, -2), "2nd reflection okay."
        assert e.reflections[-1].hkl == (2, 3, 4), "Last reflection okay."
        #Default reciprocal lattice is trivial, so the q-vector matches hkl * 2*pi.
        answer = np.array([-1, -2, -3]) * 2 * np.pi
        assert np.allclose( e.reflections[0].q_vector.flatten(),  answer), "First reflection q_vector is %s." % answer

    def test_initialize_reflections_automatic(self):
        e = self.exp
        e.range_limit_to_sphere = True
        e.range_h = (-10,10)
        e.range_k = (-10,10)
        e.range_l = (-10,10)
        e.initialize_reflections()
        assert len(e.reflections) == 27, "Correct # of reflections. We got %d" % len(e.reflections)
        assert np.all(e.reflections_q_norm <= e.inst.qlim), "No reflection is past qlim"
        e.range_automatic = True
        e.range_limit_to_sphere = False
        e.initialize_reflections()
        assert len(e.reflections) == 125, "Correct # of reflections (we got %d), automatic mode, not limited to sphere" % len(e.reflections)
        e.range_h = (-10,10)
        e.range_k = (-10,10)
        e.range_l = (-10,10)
        e.range_automatic = False
        e.range_limit_to_sphere = True
        e.initialize_reflections()
        assert len(e.reflections) == 27, "Correct # of reflections (we got %d), limited to sphere" % len(e.reflections)


    def test_get_reflection(self):
        self.setup_reflections()
        e = self.exp
        assert e.get_reflection(-1, -2, -3).hkl == (-1, -2, -3), "get_reflection works"
        assert e.get_reflection(2, 3, 4).hkl == (2, 3, 4), "get_reflection works"
        assert e.get_reflection(3, 0, 0) is None, "get_reflection out of bounds on h"
        assert e.get_reflection(-2, 0, 0) is None, "get_reflection out of bounds on h"
        assert e.get_reflection(0, -3, 0) is None, "get_reflection out of bounds on k"
        assert e.get_reflection(0, 4, 0) is None, "get_reflection out of bounds on k"
        assert e.get_reflection(-1, -2, -4) is None, "get_reflection out of bounds on l"
        assert e.get_reflection(-1, -2, 5) is None, "get_reflection out of bounds on l"
        assert e.get_reflection(1.3, 2.4, 3.1).hkl == (1, 2, 3), "get_reflection floats are clipped"
        assert e.get_reflection(0.3, 0.7, 0.9).hkl == (0, 1, 1), "get_reflection floats are rounded"

    def test_get_reflection_non_cubic(self):
        e = self.exp
        e.range_limit_to_sphere = True
        self.setup_reflections()
        #Now test some get hkls
        assert e.get_reflection(-1, 0, 0).hkl == (-1, 0, 0), "get_reflection works"
        assert e.get_reflection(0, 1, 0).hkl == (0, 1, 0), "get_reflection works"


    def setup_calculated_reflections(self):
        """Setup some calculations for reflections"""
        # @type e Experiment
        e = self.exp
        e.range_h = (-10,10) 
        e.range_k = (-10,10)
        e.range_l = (-10,10)
        e.range_automatic = False
        e.range_limit_to_sphere = False
        e.initialize_reflections()
        e.inst.wl_min = 0.5
        e.inst.wl_max = 4.0
        #Position coverage object with 0 sample rotation
        poscov = instrument.PositionCoverage( [0, 0, 0.], None, np.identity(3))
        e.inst.positions.append(poscov)
        pos_param = ParamPositions( {id(poscov):True} )
        self.pos_param = pos_param
        #Test get_positions_used
        pos = e.get_positions_used(pos_param)
        assert len(pos) == 1, "get_positions_used returns one poscov object."
        e.recalculate_reflections(pos_param)

    def continue_get_reflections_measured(self, measured, num_expected, message):
        """Mini-test of get_reflections_measured"""
        l = self.exp.get_reflections_measured(measured)
#        print len(l)
        assert isinstance(l, list), "get_reflections_measured returns a list"+message
        assert len(l)==num_expected, "get_reflections_measured returns the right # of refls (we wanted %d, got %d)%s" % (num_expected, len(l), message)
        if num_expected > 0:
            assert isinstance(l[0], Reflection), "get_reflections_measured list contains a Reflection object"+message

    def test_recalculate_reflections(self):
        self.setup_calculated_reflections()
        e = self.exp
        #Manual total check
        total = 0
        for ref in e.reflections:
            total += ref.times_measured()
        assert  total == 8, "# of reflections measured at wl_min=0.5"
        #And now we use the get_reflections_measured mini-test
        self.continue_get_reflections_measured(True, 8, " at wl_min=0.5")
        self.continue_get_reflections_measured(False, 9253, " NOT measured, at wl_min=0.5")
        #--- Another wl_min, too large
        e.inst.wl_min = 2.0
        e.recalculate_reflections(self.pos_param)
        self.continue_get_reflections_measured(True, 0, " at wl_min=2.0")
        #--- Smaller wl_min ---
        e.inst.wl_min = 0.2
        e.recalculate_reflections(self.pos_param)
        self.continue_get_reflections_measured(True, 216, " at wl_min=0.2")
        #--- Limit wl_max ---
        e.inst.wl_min = 0.2
        e.inst.wl_max = 0.5
        e.recalculate_reflections(self.pos_param)
        num_reflections = 208
        self.continue_get_reflections_measured(True, num_reflections, " at wl_max=0.5")
        #------ Add a 2nd position, same angles though ----
        poscov2 = instrument.PositionCoverage( [0, 0, 0.], None, sample_U_matrix=np.eye(3))
        e.inst.positions.append(poscov2)
        pos_param2 = copy.copy(self.pos_param)
        pos_param2.positions [id(poscov2)] = True
        e.recalculate_reflections(pos_param2)
        #Same # of reflections measured
        self.continue_get_reflections_measured(True, num_reflections, " at wl_max=0.5")
        #But counting the total measurements = twice as much
        total = 0
        for ref in e.reflections: total += ref.times_measured()
        assert  total == num_reflections*2, "total times measured with two positions"
        #--- Now test the detector settings ----
        self.exp.params[PARAM_DETECTORS] = ParamDetectors([False]*48)
        e.recalculate_reflections(self.pos_param)
        self.continue_get_reflections_measured(True, 0, " when no detectors are selected.")
        self.exp.params[PARAM_DETECTORS] = ParamDetectors([True] + [False]*47)
        e.recalculate_reflections(self.pos_param)
        self.continue_get_reflections_measured(True, 1, " when only 1 detector is selected.")
        self.exp.params[PARAM_DETECTORS] = ParamDetectors([True]*24 + [False]*24)
        e.recalculate_reflections(self.pos_param)
        self.continue_get_reflections_measured(True, num_reflections/2, " when only half the detectors are selected.")
        self.exp.params[PARAM_DETECTORS] = ParamDetectors([True]*48)
        e.recalculate_reflections(self.pos_param)
        self.continue_get_reflections_measured(True, num_reflections, " when all the detectors are selected.")
        # --- Half the detectors, counted twice ----
        self.exp.params[PARAM_DETECTORS] = ParamDetectors([True]*24 + [False]*24)
        e.recalculate_reflections(pos_param2)
        total = 0
        for ref in e.reflections: total += ref.times_measured()
        assert  total == num_reflections, "total times measured with two positions, with half the detectors"


    def test_reflection_masking(self):
        self.setup_calculated_reflections()
        # @type e Experiment
        e = self.exp
        num = len(e.reflections)
        e.params[PARAM_REFLECTION_MASKING] = None
        e.calculate_reflections_mask()
        assert np.sum(e.reflections_mask)==num, "Masking = None means all reflections are there"
        e.params[PARAM_REFLECTION_MASKING] = ParamReflectionMasking(False, 0, 0.1)
        e.calculate_reflections_mask()
        assert np.sum(e.reflections_mask)==num, "Masking: slice not on means all reflections are there"
        e.params[PARAM_REFLECTION_MASKING] = ParamReflectionMasking(True, 0, 200)
        e.calculate_reflections_mask()
        assert np.sum(e.reflections_mask)==num, "Masking: slice with large limits includes all"

        expected = 2103
        e.params[PARAM_REFLECTION_MASKING] = ParamReflectionMasking(True, 0, 50)
        e.calculate_reflections_mask()
        assert np.sum(e.reflections_mask)==expected, "Masking: slice 0 to 50, %d peaks." % (expected)
        e.params[PARAM_REFLECTION_MASKING] = ParamReflectionMasking(True, 0, 50)
        e.calculate_reflections_mask()
        assert e.reflections_q_vector[:, e.reflections_mask].shape == (3, expected), "Masking applied to array works."
        e.params[PARAM_REFLECTION_MASKING] = ParamReflectionMasking(False, 0, 50)
        e.calculate_reflections_mask()
        assert e.reflections_q_vector[:, e.reflections_mask].shape == (3, num), "Masking applied to array works, even with no masking."
        assert e.reflections_times_measured[e.reflections_mask,:].shape == (num, 1), "Masking applied to array works, even with no masking."

        #Now test some get hkls
        assert e.get_reflection(-1, -2, -3).hkl == (-1, -2, -3), "get_reflection works"
        assert e.get_reflection(2, 3, 4).hkl == (2, 3, 4), "get_reflection works"
        assert e.get_reflection(10, 10, 10).hkl == (10, 10, 10), "get_reflection works"


#    def test_calculate_peak_shape(self):
#        # @type e Experiment
#        e = self.exp
#        e.crystal.set_lattice_lengths( (5,5,5) )
#        e.crystal.set_lattice_angles( (0.4, 0.4, 0.5) )
#        e.crystal.make_ub_matrix()
#
#        self.setup_calculated_reflections()
#        count = 0
#        for ref in e.reflections:
#            if ref.times_measured() > 0:
#                (dh, dv, dl) = e.calculate_peak_shape(ref.hkl, (0.02,0.02,0.02),
#                                    e.inst.positions[0], ref.measurements[0][1], num_points=200)
#                if True:
#                    from pylab import figure, plot, axis, show
#                    figure(count)
#                    plot(dh, dv, 'ko')
#                    axis('equal')
#                    count +=1
#                    if count>12: break
#
#        show()
#        for ref in e.reflections:
#            print ref.hkl, "measured", ref.times_measured(), ref.measurements


    def profile_initialize_reflections(self):
        """Not a test, a profiler example."""
        e = self.exp
        for n in [5, 10, 20, 30]:
            e.range_h = (-n,n)
            e.range_k = (-n,n)
            e.range_l = (-n,n)
            t1 = time.time()
            e.initialize_reflections()
            print time.time()-t1, " seconds to initialize_reflections for ", (2*n+1)**3

    def profile_recalculate_reflections(self):
        """Not a test, a profiler example."""
        e = self.exp
        pos_param = ParamPositions( {} )
        for angles in [ [0.1, 0.2, 0.3] ]*20:
            poscov = instrument.PositionCoverage( angles , None)
            e.inst.positions.append(poscov)
            pos_param.positions[id(poscov)] = True
        for n in [5, 10, 20, 30]:
            e.range_h = (-n,n)
            e.range_k = (-n,n)
            e.range_l = (-n,n)
            e.initialize_reflections()
            t1 = time.time()
            e.recalculate_reflections(pos_param)
            print time.time()-t1, " seconds to recalculate_reflections for ", (2*n+1)**3
            t1 = time.time()
            e.get_reflections_times_measured(pos_param)
            print time.time()-t1, " seconds to get_reflections_times_measured for ", (2*n+1)**3

    def _check_get_equivalent_reflections(self, hkl, hkl_list):
        e = self.exp
        message = "point group %s and hkl %s" % (e.crystal.point_group_name, hkl)
        refl_list = e.get_equivalent_reflections(e.get_reflection(hkl[0], hkl[1], hkl[2]))
        found = [tuple([int(x) for x in refl.hkl]) for refl in refl_list]
        print message, "; found ", found
        assert len(found)==len(hkl_list)+1, "correct # of results for %s. We wanted %d results but got %d" % (message, len(hkl_list), len(found))
        for hkl_wanted in hkl_list:
            assert hkl_wanted in found, "hkl %s was found in the list of results for %s" % (hkl_wanted, message)
        
    def test_get_equivalent_reflections(self):
        """Test each of the 11 Laue classes to make sure the reflections work."""
        self.setup_calculated_reflections()
        e = self.exp
        e.crystal.point_group_name = crystals.get_point_group_names(long_name=True)[0]
        self._check_get_equivalent_reflections( (1,2,3), [ (-1,-2,-3) ])
        self._check_get_equivalent_reflections( (0,0,0), [ ])

    def test_reflection_stats(self):
        """Test each of the 11 Laue classes to make sure the reflections work."""
        self.setup_calculated_reflections()
        #@type e Experiment
        e = self.exp
        e.calculate_reflection_coverage_stats_adjusted(edge_x=5, edge_y=5)
        print e.reflection_stats
        print e.reflection_stats_adjusted


    def test_find_primary_reflections(self):
        # @type e Experiment
#        self.setup_calculated_reflections()
        e = self.exp
        e.crystal.point_group_name = crystals.get_point_group_names(long_name=True)[0]
        e.range_h = (-2, 2)
        e.range_k = (-2, 2)
        e.range_l = (0, 0)
        e.range_automatic = False
        e.initialize_reflections()
        e.recalculate_reflections(None)
        e.find_primary_reflections()


    def test_volume_symmetry(self):
        e = self.exp #@type e Experiment
        e.crystal.lattice_lengths = (10, 10, 10)
        e.crystal.lattice_angles_deg = (90.0, 90.0, 90.0)
        e.crystal.lattice_angles_deg = (80., 80., 80.)
        e.crystal.make_ub_matrix()
        e.crystal.calculate_reciprocal()
        e.inst.d_min = 1.0
        e.inst.q_resolution = 0.5
        e.inst.make_qspace()
        e.inst.simulate_position( [0, 0, 0] )
        #Set the point group using this distorted way
        e.crystal.point_group_name = crystals.get_point_group_from_name("mmm").long_name

        #Compare speed and results
        print "--- C-code "
        e.initialize_volume_symmetry_map()
        c_map = e.volume_symmetry
        
        print "--- Python-code "
        config.cfg.force_pure_python = True
        e.initialize_volume_symmetry_map()
        python_map = e.volume_symmetry
        config.cfg.force_pure_python = False

        assert np.allclose(c_map, python_map), "Volume symmetry map is identical when made with C and Python"
        
        #Do use symmetry
        e.params[PARAM_SYMMETRY] = ParamSymmetry(True)
        print "--- C-code "
        e.calculate_coverage(None, None)
        qspace_c = e.qspace
        #Now compare the speed with pure python
        print "--- Python-code "
        config.cfg.force_pure_python = True
        e.calculate_coverage(None, None)
        qspace_python = e.qspace
        assert np.allclose(qspace_c, qspace_python), "Volume coverage, with symmetry, is identical when made with C and Python"
#
#    def dont_test_pickle(self):
#        e = self.exp
#        datas = dumps(e)
#        print "Length of dumped is ", len(datas)
#        e2 = loads(datas)
#        assert e.crystal.name == e2.crystal.name
#
    def test_save_load(self):
        e = self.exp #@type e Experiment
        #Prepare the experiment
        self.setup_reflections()
        e.inst.d_min = 1.0
        e.inst.q_resolution = 0.5
        e.inst.make_qspace()
        pd = {}
        poscov = e.inst.simulate_position( [0, 1, 2] )
        pd[id(poscov)] = True
        poscov = e.inst.simulate_position( [3, 4, 5] )
        pd[id(poscov)] = True
        e.recalculate_reflections(None, None)
        pos_param = ParamPositions( pd )
        e.params[PARAM_POSITIONS] = pos_param
        e.calculate_coverage(None, None)
        #Now save to a file.
        test_filename = os.path.expanduser("~") + "/test_save.exp"
        save_to_file(e, test_filename)
        print "Pickled size is", os.path.getsize("test_save.exp")
        #Load it out
        e2 = load_from_file(test_filename)
        os.remove(test_filename)
#        assert e==e2, "Experiment matches before/after loading."

    def test_load_peaks(self):
        e = self.exp #@type e Experiment
        e.crystal.lattice_lengths = (10, 10, 10)
        e.crystal.lattice_angles_deg = (90.0, 90.0, 90.0)
        e.crystal.make_ub_matrix()
        e.crystal.calculate_reciprocal()
        e.initialize_reflections()
        errors = e.load_peaks_file("../model/data/TOPAZ_1241.integrate", append=False)
        #@type ref Reflection
        ref = e.get_reflection(5, 1, -1)
        assert len(ref.real_measurements)==2, "Found 2 real measurements for 5,1,-1"
        #@type rrm ReflectionRealMeasurement
        rrm = ref.real_measurements[0]
        #Check one of them.
        assert np.allclose(rrm.angles, np.deg2rad([-59.37, 45.00, 150.49])), "Angles match. %s" % rrm.angles
#        assert np.allclose( [rrm.col], [217.40] ), "Col"
#        assert np.allclose( [rrm.row], [66.71] ), "Row"
        assert np.allclose( [rrm.wavelength], [3.368968]), "Wavelength"
        assert np.allclose( [rrm.integrated], [6069.83]), "Integrated"
        assert np.allclose( [rrm.sigI], [82.90]), "sigI"
        assert rrm.detector_num == 1, "Detector 1"
        #print "# of errors:", len(errors)

    def test_load_HFIR_peaks_file(self):
        e = self.exp #@type e Experiment
        e.crystal.lattice_lengths = (10, 10, 10)
        e.crystal.lattice_angles_deg = (90.0, 90.0, 90.0)
        e.crystal.make_ub_matrix()
        e.crystal.calculate_reciprocal()
        e.initialize_reflections()
        errors = e.load_HFIR_peaks_file("../model/data/HFIR_oxalic.int", append=False)
        #@type ref Reflection
        ref = e.get_reflection(0, 1, 1)
        assert len(ref.real_measurements)==1, "Found 1 real measurements for (0, 1, 1)"

#==================================================================
class TestExperimentFourCircle(unittest.TestCase):

    def setUp(self):
        instrument.inst = instrument.InstrumentFourCircle()
        instrument.inst.set_goniometer(goniometer.HB3AGoniometer() )
        self.exp = Experiment(instrument.inst)
        e = self.exp
        e.inst.d_min = 0.5
        e.crystal.lattice_lengths = (6.06, 3.55, 11.926)
        e.crystal.lattice_angles_deg = (90.0, 106.63, 90.0)
        #UB and reciprocal
        e.crystal.make_ub_matrix()
        e.crystal.calculate_reciprocal()
        e.initialize_reflections()

    def test_get_angles_to_measure_hkl(self):
        e = self.exp #@type e Experiment
        print e.get_angles_to_measure_hkl(1,2,3, 0.01)
        e.fourcircle_measure_all_reflections()

    def test_ub_matrix(self):
        e = self.exp #@type e Experiment
        e.crystal.read_HFIR_ubmatrix_file("data/HFIR_UBmatrix.dat", "data/HFIR_lattice.dat")
        e.crystal.calculate_reciprocal()
        e.range_h = [-3,3]
        e.range_k = [-3,3]
        e.range_l = [-6,6]
        e.range_automatic = False
        e.initialize_reflections()
        # Measure all HKL
        e.fourcircle_measure_all_reflections(None)
        #@type ref Reflection
        for (h,k,l) in [ (1,0,1), (0,0,2), (-1,0,-1), (0,0,-2) ]:
            ref = e.get_reflection(h, k, l)
            if not ref is None:
                if len(ref.measurements) > 0:
                    poscovid = ref.measurements[0][0]
                    poscov = e.inst.get_position_by_id(poscovid)
                    print "HKL", h,k,l, " : ", np.array(poscov.angles)*57.3

if __name__ == "__main__":
#    unittest.main()
    
    tst = TestExperimentFourCircle('test_get_angles_to_measure_hkl')
    tst.setUp()
    tst.test_ub_matrix()

   
