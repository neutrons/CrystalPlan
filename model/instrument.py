""" Instrument Class module.

Holds the Instrument class, settings and info about the instrument
being used.
"""
import os.path

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
from ctypes import ArgumentError
import sys
import os
from cPickle import loads, dumps
import warnings
import messages
import numpy as np
from numpy import array, sin, cos, pi
#import scipy.optimize
import scipy
import weave
import time
import csv

#--- Model Imports ---
import numpy_utils
from numpy_utils import column, rotation_matrix, vector_length, normalize_vector, index_evenly_spaced, az_elev_direction
import crystal_calc
from crystal_calc import getq, getq_python
import goniometer
from goniometer import Goniometer, TopazInHouseGoniometer
from detectors import Detector, FlatDetector, CylindricalDetector
import config
import utils

#--- Multiprocessing check ---
try:
    import multiprocessing
    from multiprocessing import Process, Pool, Queue
    multiprocessing_installed = True
except ImportError, e:
    #Python version is likely < 2.6, therefore multiprocessing is not available
    multiprocessing_installed = False


#========================================================================================================
#============== FUNCTIONS FOR MULTIPROCESSING ===========================================================
#========================================================================================================
def detector_coverage_process_3D( args):
    """Calculate 3D coverage of a single detector."""
    #Error checking
    (inst, det, set_value) = args
    if det is None: return None
    if inst is None: return None
    message = "Calculating coverage of detector '%s'..." % det.name
    #Statusbar update
    #TODO: This notification does nothing in multiprocessing because this function runs in a separate process!
    messages.send_message(messages.MSG_UPDATE_MAIN_STATUSBAR, message)
    print message
    #This does the calculation
    coverage = inst.detector_coverage_3D_by_lines(det, set_value)
    #We have the coverage map, return it
    return coverage




#========================================================================================================
#========================================================================================================
#========================================================================================================
class PositionCoverage:
    """Class holding the results of the calculated detector coverage at a given position(orientation).
    A list of these will be a member of the Instrument class."""

    def __init__(self, angles, coverage, sample_U_matrix):
        """Constructor, initialize angles and coverage.

        Parameters:
            angles: Angles of the sample, (a list). Length needs to match instrument.angles.
                Typically, will be in radians.
            coverage: 3D array with the coverage; indices are x,y,z
            sample_U_matrix: 3x3 matrix describing the sample mounting orientation.

        """
        #Angles of the sample, (a list). Length needs to match instrument.angles
        if isinstance(angles, np.ndarray):
            #Convert to a list
            self.angles = list(angles.flatten())
        else:
            self.angles = angles
            
        #3D array with the coverage; indices are x,y,z
        self.coverage = coverage
        #3x3 matrix describing the sample mounting orientation.
        self.sample_U_matrix = sample_U_matrix
        #Criterion for stopping measurement, a string
        #Value (float) associated with the criterion. Could be runtime in seconds, etc.
        try:
            (self.criterion, self.criterion_value) = inst.get_default_stopping_criterion()
        except:
            (self.criterion, self.criterion_value) = ("runtime", 0)
        #A comment string
        self.comment = ""

    def get_sample_rotation_matrix(self):
        """Return the sample rotation matrix for the angles of this positionCoverage object."""
        (phi, chi, omega) = self.angles
        return rotation_matrix(phi, chi, omega)

    #========================================================================================================
    #======================================= PICKLING =====================================
    #========================================================================================================
    def __getstate__(self):
        """Return a dictionary containing all the stuff to pickle."""
        #Exclude all these attributes.
        exclude_list = ['coverage']
        return utils.getstate_except(self, exclude_list)

    #========================================================================================================
    def __setstate__(self, d):
        """Set the state of experiment, using d as the settings dictionary."""
        for (key, value) in d.items():
            setattr(self, key, value)
        self.coverage = None #You need to calculate this later.

    #========================================================================================================
    def __eq__(self, other):
        return utils.equal_objects(self, other)
        


#========================================================================================================
#========================================================================================================
#========================================================================================================

#For traits
from traits.api import HasTraits,Int,Float,Str,Property
from traitsui.api import View,Item,Label,Heading, Spring, Handler
from traitsui.menu import OKButton, CancelButton,RevertButton


#========================================================================================================
class InstrumentParameters(HasTraits):
    """This class holds parameters of an instrument that would require the entire calculation to
    be redone if changed.
    The parameters are copied into the Instrument object, they are not used directly.
    """

    #Minimum detectable wavelength in Angstroms
    wl_min = Float(1)
    #And max
    wl_max = Float(3.6)

    #The limits to d (interplanar spacing) to consider when modeling space. In A^-1
    d_min = Float(1)
    #d_max does not really have an effect here.
    d_max = Float(5.0)
        

#========================================================================================================
#========================================================================================================
#========================================================================================================
class Instrument:
    """ SCD Instrument class:
    Holds all relevant information for a SCD instrument.
        - Goniometer: for the control motors
        - Detector: list of detector positions
    """


    #The extents of defined q-space are -qlim to +qlim in the 3 axes, will be calculated from d_min
    @property
    def qlim(self):
        """Limit in defined q-space."""
        return 2*np.pi / self.d_min

    #========================================================================================================
    def initizalize (self):
        #List of all installed detectors
        self.detectors = []

        #This 3D array holds the q-radius (i.e. |q| ) at all points of the grid. Indices are x,y,z
        #   It is used to quickly do slices.
        self.qspace_radius = None

        #List of the calculated PositionCoverage's
        self.positions = list()

        #Other stuff, mostly used by GUIs
        self.last_sort_ascending = False
        self.last_sort_angle_num = -1

        #Default as verbose
        self.verbose = True

        #Minimum detectable wavelength in Angstroms
        self.wl_min = 0.4
        #And max
        self.wl_max = 3.4

        #The limits to d (interplanar spacing) to consider. In A^-1
        self.d_min = 0.7
        self.d_max = 5

        #Resolution of the coverage map of q-space, in Angstroms^-1
        self.q_resolution = 0.2

        self.verbose = True

        
    #========================================================================================================
    def __init__ (self, filename=None, params=dict()):
        """Create an instrument. Will load the geometry from a supplied CSV file."""
        #Create the default members
        self.initizalize()
        
        if not (filename is None):
            self.load_detectors_file(filename)
        else:
            #Save it, for reloading
            self.detector_filename = filename

        #Now set the params
        self.set_parameters(params)

        #Goniometer
        self.set_goniometer( goniometer.TopazAmbientGoniometer(), False )


    #========================================================================================================
    def __eq__(self, other):
        return utils.equal_objects(self, other)




    #========================================================================================================
    #======================================= PICKLING =====================================
    #========================================================================================================
    def __getstate__(self):
        """Return a dictionary containing all the stuff to pickle."""
        #Exclude all these attributes. Make sure to exclude properties!!!
        exclude_list = ['qspace_radius', 'detectors', 'qlim']
        return utils.getstate_except(self, exclude_list)

    #========================================================================================================
    def __setstate__(self, d):
        """Set the state of experiment, using d as the settings dictionary."""
        self.initizalize()

        #Things to NOT load; fix old files.
        exclude_list = ['qlim']
        
        for (key, value) in d.items():
            if not (key in exclude_list):
                setattr(self, key, value)

        #Fix the goniometer
        self.set_goniometer(self.goniometer)

        #Now, re-load the detectors
        #TODO: Check that the file still exists!
        if not self.detector_filename is None:
            self.load_detectors_file(self.detector_filename)
        #Generate your q-space stuff
        self.make_qspace()
        #Re-calculate all the positions (with all detectors enabled)
        for poscov in self.positions: #@type poscov PositionCoverage
            poscov.coverage = self.calculate_coverage(self.detectors, poscov.angles, poscov.sample_U_matrix, use_inline_c=True)


#    #---------------------------------------------------------------------------------------------
#    def _load_nxs_file(self, filename):
#        """Load the detector geometry from a supplied NXS file."""
#        import nxs
#        _nxs = nxs.open(filename)
#        for entry_name,entry_nxclass in _nxs.entries():
#            #First should be the "entry" field
#            if entry_nxclass=='NXentry':
#                for instrument_name, instrument_nxclass in _nxs.entries():
#                    #Next we should look for an instrument class
#                    if instrument_nxclass=='NXinstrument':
#                        for detector_name, detector_nxclass in _nxs.entries():
#                            if detector_nxclass=='NXdetector':
#                                #Load the detector, add it if it is okay
#                                newdetector = Detector(detector_name, _nxs)
#                                if newdetector != None:
#                                    self.detectors.append(newdetector)
#                        #Stop after the 1st (and normally only) instrument
#                        break
#                #Stop after the 1st (and normally only, if there is a single run) entry
#                break


    #---------------------------------------------------------------------------------------------
    def load_detectors_file(self, filename):
        """Load the detector geometry from a file; picks the method based on the file extension."""
        if not os.path.exists(filename):
            print "Error! The supplied detector filename '%s' does not exist. No detectors were loaded." % filename
            return

        #Save it, for reloading later.
        self.detector_filename = filename

        (main, ext) = os.path.splitext(filename)
        ext = ext.lower()
        if ext == ".detcal":
            self.load_detectors_detcal_file(filename)
        else:
            self.load_detectors_csv_file(filename)


    #---------------------------------------------------------------------------------------------
    def load_detectors_csv_file(self, filename):
        """Load the detector geometry from a CSV file."""
        #Initialize members
        old_list = self.detectors
        self.detectors = list()

        try:
            reader = csv.reader( open(filename) )
            row = reader.next()
            cylindrical = False
            if row[0].startswith("#Cyl"):
                cylindrical = True
                
            # Skip other comment rows
            while True:
                row = reader.next()
                if len(row) > 0:
                    if len(row[0]) > 0:
                        if row[0][0] != '#': break
                    else:
                        break
                else:
                    break
            
            count = 1
            for row in reader:
                if cylindrical:
                    # -------- Cylindrical Detector ---------
                    name = row[0].strip()
                    if name == "":  name = "d%d" % number
                    det = CylindricalDetector( name )
                    det.origin[0] = float(row[1])
                    det.origin[1] = float(row[2])
                    det.origin[2] = float(row[3])
                    det.radius = float(row[4])
                    det.height = float(row[5])
                    det.angle_start = float(row[6])
                    det.angle_end = float(row[7])
                    #Calculate the pixel angles
                    det.calculate_pixel_angles()
                    self.detectors.append(det)
                    
                else:
                    # -------- Flat Detector ---------
                    number = round(float(row[0]))
                    name = row[1].strip()
                    if name == "":
                        name = "d%d" % number
                    det = FlatDetector( name )
                    #Distance in mm
                    det.distance = float(row[5]) * 10
                    det.elevation_center = float(row[3])
                    det.azimuth_center = float(row[2])
                    det.rotation = float(row[4])
                    det.width = float(row[6]) * 10
                    det.height = float(row[7]) * 10
                    # Is the detector in use?
                    try:
                        inuse = (float(row[8]) != 0)
                    except:
                        inuse = False
                        
                    if (inuse):
                        #Calculate the pixel angles
                        det.calculate_pixel_angles()
                        self.detectors.append(det)
                count = count + 1
        except:
            #Uh-oh! Let's go back to the original list
            self.detectors = old_list
            #And handle the error normally.
            raise


    #---------------------------------------------------------------------------------------------
    def load_detectors_detcal_file(self, filename):
        """Load the detector geometry from an ISAW .detcal file."""

        #Initialize members
        old_list = self.detectors
        self.detectors = list()

        try:
            f = open(filename, 'r')
            count = 0
            for row in f.readlines():
                if len(row) > 0:
                    if row[0]=='5':
                        #Only lines starting with 5 are real.
                        #Convert to a list of floats
                        values = [float(x) for x in row.split()]
                        #Create the detector
                        name = "%d" % values[1]
                        #@type det FlatDetector
                        det = FlatDetector( name )
                        det.xpixels = int(values[3])
                        det.ypixels = int(values[2])
                        #Convert cm to mm
                        det.width = values[4] * 10
                        det.height = values[5] * 10
                        det.distance = values[7] * 10
                        #Vectors determining the center, and base, and up vectors
                        center = np.array(values[8:11])*10.0
                        #Looks like I need to flip the horizontal vector direction.
                        base = -np.array(values[11:14])*10.0
                        up = np.array(values[14:17])*10.0

                        #Calculate the pixel angles using these vectors
                        det.calculate_pixel_angles_using_vectors(center,base,up)
                        self.detectors.append(det)
                count = count + 1
        except:
            #Uh-oh! Let's go back to the original list
            self.detectors = old_list
            #And handle the error normally.
            raise


    #========================================================================================================
    def set_goniometer(self, gonio, different_angles=False):
        """Change the goniometer used by the instrument.
        
        Parameters:
            gonio: a new Goniometer() class object.
            different_angles: bool, set to True if the angles are different in the new goniometer.
                That means the position coverage objects have to be cleared.
        """
        #The goniometer class holds functions for calculating the necessary motor position, etc.
        self.goniometer = gonio

        #List of the names of the sample orientation angles to use.
        #   Different instruments may have more angles available.
        self.angles = gonio.get_angles()

        #Do we need to clear all the positions?
        if different_angles:
            self.positions = []
                

    #========================================================================================================
    def change_qspace_size(self, params):
        """Call this function to change the size and/or resolution of q-space you are modeling.
        Calculations of spatial coverage will still need to be redone."""
        #Set these parameters
        self.set_parameters(params)
        print "change_qspace_size", params
        
        #Start by recalculating the q-space array here.
        self.make_qspace()

    #========================================================================================================
    def set_parameters(self, params):
        """Set some of the parameters in the instrument.

        Parameters:
            params: a dictionary with the key as the name of the attribute."""
        for (param, value) in params.items():
            #Only set an attribute if the name exists already.
            if hasattr(self, param):
                setattr(self, param, value)

    #========================================================================================================
    def sort_positions_by(self, angle_num):
        """Sort the list of positions.

        Parameters:
            angle_num: index into self.angles of which angle to sort."""
        #Valid index?
        if angle_num < 0 or angle_num >= len(self.angles):
            return

        #Choose the direction of sort and save it for next time.
        ascending = True
        if self.last_sort_angle_num == angle_num:
            ascending = not self.last_sort_ascending
            self.last_sort_ascending = ascending
        self.last_sort_angle_num = angle_num

        #Sort by all the angles, with the selected one at top
        decorated = []
        for (i, poscov) in enumerate(self.positions):
            sort_list = [poscov.angles[angle_num]]
            if angle_num > 0: sort_list += poscov.angles[0:(angle_num-1)]
            if angle_num < len(self.angles)-1: sort_list += poscov.angles[(angle_num+1):]
            #Also add the original index to keep the sort stable.
            sort_list += [i]
            decorated.append( tuple(sort_list + [poscov]) )
            
        decorated.sort(reverse=(not ascending))
        #Save it back in the array
        self.positions[:] = [x[-1] for x in decorated]



    #========================================================================================================
    def recalculate(self, poscov, new_sample_U_matrix=None):
        """Recalculates one positionCoverage object.
        Used for example when q-space size or resolution changes.
        Parameters:
            poscov: PositionCoverage
            sample_U_matrix: None if you are to reuse the one in the instrument.
                or, set to a 3x3 matrix to change it in the calculation.
        """
        #Find the u matrix
        if new_sample_U_matrix is None:
            new_sample_U_matrix = poscov.sample_U_matrix
        #Calculate
        poscov.coverage = self.calculate_coverage(self.detectors, poscov.angles,
                            sample_U_matrix=new_sample_U_matrix)
        #And make sure to save the U matrix (changed or not)
        poscov.sample_U_matrix = new_sample_U_matrix
        

    #========================================================================================================
    def evaluate_position_list(self, angles_lists, ignore_gonio):
        """Given lists of phi, chi, omega angles to loop through, check
        that each combination is possible given the goniometer.
        
            angles_lists: list of lists of angles. There are as many lists as there are angles in
                instrument.angles: e.g. 3 for phi, chi, omega.
                Angles are in internal units.
            ignore_gonio: set to True to ignore any goniometer limitations.
            
        returns (valid, redundant, invalid, invalid_reason):
            valid: list of [phi, chi, omega] angles that are valid
            redundant: list of redundant angles
            invalid: list of invalid (unreachable) angles
            invalid_reason: list of strings describing the reason of each invalid angle
                
        """
        
        valid = list()
        invalid = list()
        redundant = list()
        invalid_reason = list()

        #The number of angles
        num_angles = len(angles_lists)
        #Indices to each arrays
        indices = np.zeros( num_angles, dtype=int )
        #Length of each list
        list_length = np.zeros( num_angles, dtype=int )
        for i in range(num_angles):
            list_length[i] = len(angles_lists[i])

        while True:
            #Make the list of angles for these indices
            angles = np.zeros(num_angles)
            for i in range(num_angles):
                angles[i] = angles_lists[i][indices[i]]
            #Check it!
            (allowed, reason) = self.goniometer.are_angles_allowed(angles, return_reason=True)
            if ignore_gonio or allowed:
                #Yes, it is possible
                valid.append( angles )
                #TODO: Check for redundancy
            else:
                invalid.append( angles )
                invalid_reason.append(reason)

            #Okay, now move on to the next spot in the list. We have to iterate through all the
            #   combinations from all lists. The first list is iterated first, then the 2nd, etc.
            i = 0
            while (i < num_angles):
                indices[i] += 1
                if indices[i] >= list_length[i]:
                    #Reached the end of this index. Increment the next one instead.
                    indices[i] = 0
                    i += 1
                else:
                    break
                    
            #If all the indices are zero, we have wrapped back to the start. Stop looping
            if all( indices==0 ):
                break

     
        return (valid, redundant, invalid, invalid_reason)



    #========================================================================================================
    def make_blank_qspace(self, datatype, number_of_ints=None):
        """Generate an evenly-spaced q-space map, initialized to zero,
        using the settings and limits given in the object.
        make_qspace() should have been called first.
            datatype: specify the datatype to use, e.g. numpy.int64
            Returns: the 3D array.
        """
        if self.qx_list is None:  
            warnings.warn("instrument.make_blank_qspace(): called before instrument was properly initialized. Call make_qspace() first!")
            return None
        #Find the size and make the array
        n = self.qx_list.size
        if not number_of_ints is None:
            return np.zeros( (n,n,n,number_of_ints), dtype=datatype )
        else:
            return np.zeros( (n,n,n), dtype=datatype )
    

    #-------------------------------------------------------------------------------
    def make_qspace(self):
        """Generate an evenly-spaced q-space map using the settings and limits 
        given in the object."""

        #Limit to q volume given the d_min
        qlim = self.qlim

        #Create the lists giving the q values
        self.qx_list = np.arange(-qlim, qlim, self.q_resolution)

        #Generate the radius 3D matrix
        ##spacing = self.q_resolution
        lims = qlim/self.q_resolution #Integer limit
        #ogrid can only produce integer grids
        (qx, qy, qz) = np.ogrid[-lims:lims, -lims:lims, -lims:lims]
        #So we do the radius calculation in units of "self.q_resolution"
        self.qspace_radius = np.sqrt( qx**2 + qy**2 + qz**2 ) * self.q_resolution


    #========================================================================================================
    # C-code for calculate_coverage
    _code_vector_length = """
        double vector_length(py::tuple vector)
        {
            double length = 0;
            for (int i=0; i < vector.length(); i++)
                length += double(vector[i])*double(vector[i]);
            return sqrt(length);
        }
        """

    _code_shrink_q_vector = """void shrink_q_vector(py::tuple &q, double limit)
            {
                double length = vector_length(q);
                if (length <= 0)
                    return;

                //py::tuple q_out(q.length());

                for (int i=0; i < q.length(); i++)
                    {
                    if (length > limit)
                        { q[i] = double(q[i]) * (limit / length);
                        }
                    else
                        { q[i] = double(q[i]);
                        }
                    }
                //return q_out;
            }"""
            
    _code_calculate_coverage = """
            //Loop through pixels using the list given before.
            for (int iix = 0; iix < xlist.length(); iix++)
            {
                int ix = xlist[iix];
                for (int iiy = 0; iiy < ylist.length(); iiy++)
                {
                    int iy = ylist[iiy];

                    //Angles of the detector pixel positions
                    double az, elev;
                    az = AZIMUTHAL_ANGLE2(iy, ix);
                    elev = ELEVATION_ANGLE2(iy, ix);

                    //Get the two limits.
                    py::tuple q_min = getq(wl_min, az, elev, pi, rot_matrix);
                    py::tuple q_max = getq(wl_max, az, elev, pi, rot_matrix);
                    //Limit them to the modeled size
                    shrink_q_vector(q_min, qlim);
                    shrink_q_vector(q_max, qlim);

                    //Find out how long of a line that is
                    double q_length = (vector_length(q_max) - vector_length(q_min));
                    if (q_length<0) q_length = -q_length;

                    //How many steps will we take. The multiplication factor here is a fudge to make sure it covers fully.
                    int numfrac = (1.25 * q_length) / (q_resolution);

                    // if (numfrac == 0) printf("numfrac is %d, q_length %f, qmin and max are %f and %f, abs is %f \\n", numfrac, q_length, vector_length(q_min), vector_length(q_max),   (vector_length(q_max) - vector_length(q_min))  );

                    if (numfrac > 0)
                    {
                        //There is something to measure.

                        //Size in q-space of each step
                        double dx, dy, dz;
                        dx = (double(q_max[0]) - double(q_min[0])) / numfrac;
                        dy = (double(q_max[1]) - double(q_min[1])) / numfrac;
                        dz = (double(q_max[2]) - double(q_min[2])) / numfrac;

                        /*
                        printf("%f, %f\\n",vector_length(q_min),vector_length(q_max));
                        printf("%d\\n", numfrac);
                        printf("%f, %f, %f\\n", dx, dy, dz);
                        */

                        long index;
                        double qx, qy, qz;
                        long iqx, iqy, iqz;
                        // unsigned int* coverage_int = (unsigned int*) coverage;

                        double lim_min = -qlim;
                        double lim_max = +qlim;

                        double q_min_x = double(q_min[0]);
                        double q_min_y = double(q_min[1]);
                        double q_min_z = double(q_min[2]);
                        //printf("Setvalue1 is %d\\n", set_value1);

                        //Okay now we draw the line from q_min to q_max.
                        long i;
                        for (i=0; i<numfrac; i++)
                        {
                            //All of these qx checks might not be necessary anymore...?
                            qx = q_min_x + i*dx;
                            iqx = round((qx - lim_min) / q_resolution);
                            if ((iqx >=0) && (iqx < stride))
                            {
                                qy = q_min_y + i * dy;
                                iqy = round((qy - lim_min) / q_resolution);
                                if ((iqy >=0) && (iqy < stride))
                                {
                                    qz = q_min_z + i * dz;
                                    iqz = round((qz - lim_min) / q_resolution);
                                    if ((iqz >=0) && (iqz < stride))
                                    {
                                        if (number_of_ints==2)
                                        {
                                            COVERAGE4(iqx,iqy,iqz,0) |= set_value1;
                                            COVERAGE4(iqx,iqy,iqz,1) |= set_value2;
                                        }
                                        else
                                        {
                                            COVERAGE4(iqx,iqy,iqz,0) |= set_value1;
                                        }
                  
                                    }
                                }
                            }
                        } //for i in numfrac
                    } //numfrac > 0 so we can draw the line

                } //for iiy

            }"""


    #========================================================================================================
    def get_coverage_number_of_ints(self):
        """Return the number of 32-bit ints needed to have the detector mask."""
        if len(self.detectors)>=32:
            return 2
        else:
            return 1

    #========================================================================================================
    def get_wavelength_range(self, angles):
        """Return the wavelength range, either from goniometer settings, or from
        instrument default."""
        (wl_min, wl_max) = self.goniometer.get_wavelength_range(angles)
        if wl_min is None:
            (wl_min, wl_max) = (self.wl_min, self.wl_max)
        return (wl_min, wl_max)


    #========================================================================================================
    def calculate_coverage(self, det_list, angles, sample_U_matrix=np.identity(3), use_inline_c=True):
        """This method finds the coverage of all detectors in a full 3D matrix by drawing
        straight lines from the min. wavelength to the max. wavelength. This is, so far, the most
        efficient way I have found to calculate it.
        Optimized for speed using inline C code.

        Parameters:
            det_list: list of all the detectors (objects) to calculate.
            angles: sample orientation angles, in radians (usually), as a list.
            sample_U_matrix: U matrix of the sample mount angles.

        Returns:
            coverage: the q-space 3D array of coverage. Each point is a 64-bit-mask identifying each
                detector.
        """

        def shrink_q_vector(q):
            """Helper function returns a shortened q-vector, capped
            to self.qlim but with the same direction."""
            q_length = vector_length(q)
            if q_length > self.qlim:
                return (q / q_length) * self.qlim
            else:
                return q

        if not isinstance(det_list, list):
            raise ArgumentError("Did not supply a list as det_list.")
        if not (len(angles) == len(self.angles)):
            raise ArgumentError("You supplied a list of angles of a length that does not match what the instrument is setup for.")

        #For passing out messages
        angles_string = self.make_angles_string(angles)

        #Calculate the q-rotation matrix
        rot_matrix = self.goniometer.make_q_rot_matrix(angles)
        #Multiply by the inverse of the sample's U matrix.
        # qR = R * U^-1
        #   so that we undo the goniometer angle rotation 1st, then the sample mounting rotation
        #   = the opposite order of the rotations in the first case.
        rot_matrix = np.dot(np.linalg.inv(sample_U_matrix), rot_matrix)

        last_time = time.time()

        #Start with zeros
        number_of_ints = self.get_coverage_number_of_ints()
        coverage = self.make_blank_qspace(np.uint32, number_of_ints=number_of_ints)
        count = 0
        if self.verbose: sys.stdout.write( "For angles [%s], calculating coverage of detectors... " % angles_string)
        for det in det_list:
            #Make sure the detector object is valid
            if det is None:
                count = count+1
                continue

            #Output and statusbar messages, if it's been long enough
            messages.send_message_optional(self, messages.MSG_UPDATE_MAIN_STATUSBAR, "Calculating coverage of detector '%s' at %s" % (det.name, angles_string))
            if (time.time() - last_time) > 0.33:
                last_time = time.time()
                if self.verbose: sys.stdout.write(det.name + ", ")
                if self.verbose: sys.stdout.flush()

            #The binary flag to use here.
            if count < 31:
                set_value1 = (2**count)
                set_value2 = (0)
            else:
                set_value1 = (0)
                set_value2 = (2**(count-31))
            count = count+1

            #Find the wavelength range to use
            (wl_min, wl_max) = self.get_wavelength_range(angles)
            
            #Two nearby pixels
            q0 = getq(det.azimuthal_angle[0, 0], det.elevation_angle[0, 0], wl_min, rot_matrix)
            q_xmax = getq(det.azimuthal_angle[0, -1], det.elevation_angle[0, -1], wl_min, rot_matrix)
            q_ymax = getq(det.azimuthal_angle[-1, 0], det.elevation_angle[-1, 0], wl_min, rot_matrix)

            #Make sure they aren't too long
            q0 = shrink_q_vector(q0)
            q_xmax = shrink_q_vector(q_xmax)
            q_ymax = shrink_q_vector(q_ymax)

            #Project them to all the same length (whichever is the longest)
            length = max( vector_length(q_xmax),  vector_length(q_ymax),  vector_length(q0) )
            q0 = normalize_vector(q0, length)
            q_xmax = normalize_vector(q_xmax, length)
            q_ymax = normalize_vector(q_ymax, length)

            #The pixels to look at (dont look at all of them)
            nx = (vector_length(q_xmax - q0) / self.q_resolution) * 1.5
            ny = (vector_length(q_ymax - q0) / self.q_resolution) * 1.5
            if (nx < 1): nx = 1
            if (ny < 1): ny = 1

            #So this is the list of the pixels in the detectors that we pick to calculate where they are in q-space
            #   (convert to int and take only unique values; convert back to list).
            xlist = [x for x in set(np.linspace(0, det.xpixels-1, nx, endpoint=True).astype(int))]
            xlist.sort()
            ylist = [x for x in set(np.linspace(0, det.ypixels-1, ny, endpoint=True).astype(int))]
            ylist.sort()

            if use_inline_c and not config.cfg.force_pure_python:
                #------- Inline C ---------
                #Get some variables ready
                azimuthal_angle = det.azimuthal_angle
                elevation_angle = det.elevation_angle
                #Set up several functions used in the code
                support = "#include <math.h>\n"
                support += crystal_calc.getq_code_header + crystal_calc.getq_code + crystal_calc.getq_code_footer
                support += self._code_vector_length
                support += self._code_shrink_q_vector
                #Dimensions of the array
                s = coverage.shape
                stride = s[0]
                max_index = s[0]*s[1]*s[2] #largest index into the array +1
                #Make the list of local vars to use.
                varlist = ['xlist', 'ylist', 'azimuthal_angle', 'elevation_angle']
                varlist += ['pi', 'rot_matrix']
                varlist += ['set_value1', 'set_value2', 'number_of_ints']
                varlist += ['coverage', 'stride', 'max_index']
                varlist += ['wl_min', 'wl_max']
                #Dump these  in the locals namespace
                attribute_list = ['qlim', 'q_resolution']
                for var in attribute_list: locals()[var] = getattr(self, var)
                varlist += attribute_list
                #Run the C code (see between function declarations for the actual code).
                weave.inline(self._code_calculate_coverage, varlist, compiler='gcc', support_code=support) # , libraries = ['m'])

            else:
                #-------- Pure Python ---------
                for ix in xlist:
                    for iy in ylist:
                        #The angles we find are the azimuth and elevation angle of the scattered beam.
                        az = det.azimuthal_angle[iy, ix]
                        elev = det.elevation_angle[iy, ix]

                        #Now we find the reciprocal vector r=1/d which corresponds to this point on the ewald sphere.
                        q_min = getq(az, elev, wl_min, rot_matrix)
                        q_max = getq(az, elev, wl_max, rot_matrix)
                        #Cap to qlim
                        q_min = shrink_q_vector(q_min)
                        q_max = shrink_q_vector(q_max)
                        #Length to walk through
                        q_diff = q_max-q_min

                        #Calculate an optimal # of points to use in the q direction.
                        numfrac = int(1.25 * vector_length(q_diff) / self.q_resolution)
                        if numfrac > 0:
                            #If we get numfrac==0, that means nothing can be detected.
                            for frac in np.arange(0.0, 1.0, 1.0/numfrac):
                                #This is the intermediate q
                                q = q_min + frac*q_diff
                                #Find the indices in q-space that correspond
                                iqx = index_evenly_spaced(-self.qlim, len(self.qx_list), self.q_resolution, q[0])
                                if iqx is None: continue
                                iqy = index_evenly_spaced(-self.qlim, len(self.qx_list), self.q_resolution, q[1])
                                if iqy is None: continue
                                iqz = index_evenly_spaced(-self.qlim, len(self.qx_list), self.q_resolution, q[2])
                                if iqz is None: continue
                                #All indices are valid, hurrah!
                                coverage[iqx, iqy, iqz, 0] |= set_value1
                                if number_of_ints == 2:
                                    coverage[iqx, iqy, iqz, 1] |= set_value2

        if self.verbose:  print " done!"

        return coverage


    #========================================================================================================
    def make_angles_string(self, angles):
        """Pretty-prints a list of angles.

        Parameters:
            angles: list of angle (or other values in AngleInfo).

        Returns:
            a string.
        """
        import string
        return string.join( [self.angles[i].pretty_print(angles[i], add_unit=True) for i in range(len(angles))], ", " )

    #========================================================================================================
    def get_default_stopping_criterion(self):
        """Return a default stopping criterion for a new PositionCoverage object.
        Finds the last one given and uses that.

        Return
            criterion, value: string, float
        """
        #Some default numbers
        criterion = "runtime"
        value = 60
        if len(self.positions)>0:
            if not self.positions[-1] is None:
                criterion = self.positions[-1].criterion
                value = self.positions[-1].criterion_value
        return (criterion, value)

    
    #========================================================================================================
    def get_position_num(self, num):
        """Return the PositionCoverage object # num in the list of saved positions, or None if 
        out of bounds."""
        if num < 0 or num >= len(self.positions):
            return None
        return self.positions[num]
    
    #========================================================================================================
    def get_position_by_id(self, poscovid):
        """Return the PositionCoverage object that matches the given poscovid id number,
        or None if not found."""
        for poscov in self.positions:
            if id(poscov) == poscovid:
                return poscov
        return None

    #========================================================================================================
    def simulate_position(self, angles, sample_U_matrix=np.identity(3), use_multiprocessing=False):
        """Function to simulate coverage for a given sample orientation, and save the results in the
        list of positions in the instrument.

        Parameters:
            angles: list of angles in radians.
            sample_U_matrix: sample mounting angles matrix U
            use_multiprocessing: will attempt to use multiprocessing.

        Returns:
            pos: The PositionCoverage object that was just calculated.
        """
        angles_string = self.make_angles_string(angles)

        ump = ""
        if use_multiprocessing: ump = " using multiprocessing"

        #Send messages, but not too frequently.
        messages.send_message_optional(self, messages.MSG_UPDATE_MAIN_STATUSBAR, "Calculating %s%s..." % (angles_string,ump))
            
        t1 = time.time()
        coverage = self.calculate_coverage(self.detectors, angles, sample_U_matrix=sample_U_matrix)
        if self.verbose:
            print "instrument.simulate_position done in %s sec." % (time.time()-t1)

        #Create a PositionCoverage object that holds both the position and the coverage
        pos = PositionCoverage(angles, coverage, sample_U_matrix=sample_U_matrix)

        #Add it to the list.
        self.positions.append(pos)
        #Statusbar update
        messages.send_message_optional(self, messages.MSG_UPDATE_MAIN_STATUSBAR, "Calculation of %s complete." % angles_string)

        return pos


    #========================================================================================================
    def total_coverage(self, detectors_used, orientations_used, use_inline_c=True):
        """Calculate the total coverage, with some options, as bool arrays:

        Parameters:
            detectors_used: a bool list, giving which of the measured detectors are considered. if None, all are used.
            orientations_used: a list of PositionCoverage objects to add up together.
                If None, all the entries saved in the instrument are used.
            use_inline_c: bool, True to use inline C routine, false for pure Python.

        Returns:
            coverage: a 3D array with the # of times each voxel was measured.
        """
        #print "instrument.total_coverage( %s, %s)" % ( detectors_used, orientations_used)
        #print "instrument.total_coverage starting..."

        t1 = time.time()
        #Create the q-space
        coverage = self.make_blank_qspace(np.int16)
        
        #Error checking
        if self.positions is None: return None

        #Create a bitwise AND mask that will only show the detectors we want
        mask = 0
        mask1 = 0
        mask2 = 0
        if detectors_used is None:
            mask = np.uint64(-1) #Will show all the bits.
            mask1 = 2**31-1
            mask2 = 2**31-1
        else:
            for i in range(len(detectors_used)):
                if detectors_used[i]:
                    if i < 62:
                        mask = np.uint64(mask + 2**i) #64-bit mask
                        #Also make two 32-bit masks
                        if i < 31:
                            mask1 = mask1 + 2**i
                        else:
                            mask2 = mask2 + 2**(i-31)
                    else:
                        print "cannot compute total_coverage for detector #%d as it exceed the max of 62 detectors." % i

        #Make sure the list of positions makes sense
        if orientations_used is None:
            orientations_used = self.positions #Just refer to the same list saved here.

        #Look through each orientation saved and add them up.
        coverage_list = list()
        for pos_cov in orientations_used:
            if pos_cov is None: continue
            if pos_cov.coverage is None: continue
            #Add the coverage of the position used to the list
            coverage_list.append(pos_cov.coverage)

        number_of_ints = self.get_coverage_number_of_ints()

        #Now we add up the coverages together
        if config.cfg.force_pure_python or not use_inline_c:
            #--- Pure Python Version ---
            for one_coverage in coverage_list:
                #By applying the mask and the >0 we take away any unwanted detectors.
                if number_of_ints==1:
                    #coverage = coverage + ((one_coverage & mask) != 0)
                    coverage += ((one_coverage[:,:,:,0] & mask1) != 0)
                else:
                    coverage += (((one_coverage[:,:,:,0] & mask1) | (one_coverage[:,:,:,1] & mask2)) != 0)

        else:
            #--- Inline C version ---
            #   about 70x faster than the python version.
            coverage_size = coverage.size 
            num_coverage = len(coverage_list)
            support = ""
            code = """
            int i, j;
            PyArrayObject* each_coverage[num_coverage];
            for (j=0; j<num_coverage; j++)
                each_coverage[j] = (PyArrayObject*) PyList_GetItem(coverage_list, j);

            // npy_uint* one_coverage;
            npy_uint low, high;

            for (j=0; j<num_coverage; j++)
            {
                npy_intp* Sone_coverage = each_coverage[j]->strides;

                for (int ix=0; ix < Ncoverage[0]; ix++)
                {
                    for (int iy=0; iy < Ncoverage[1]; iy++)
                    {
                        for (int iz=0; iz < Ncoverage[2]; iz++)
                        {
                            if (number_of_ints==2)
                            {
                                //NOTE: This code assumes LSB-first ints.
                                low = (*((npy_uint*)(each_coverage[j]->data + (ix)*Sone_coverage[0] + (iy)*Sone_coverage[1] + (iz)*Sone_coverage[2] + 0)));
                                high = (*((npy_uint*)(each_coverage[j]->data + (ix)*Sone_coverage[0] + (iy)*Sone_coverage[1] + (iz)*Sone_coverage[2] + 1*Sone_coverage[3])));
                                if ((low & mask1) || (high & mask2))
                                {
                                    COVERAGE3(ix,iy,iz)++;
                                }
                            }
                            else
                            {
                                //Only 1 int
                                //Index into the coverage array
                                low = (*((npy_uint*)(each_coverage[j]->data + (ix)*Sone_coverage[0] + (iy)*Sone_coverage[1] + (iz)*Sone_coverage[2])));
                                // if (low != 0) { printf("low is %d\\n", low);}
                                if (low & mask1)
                                {
                                    COVERAGE3(ix,iy,iz)++;
                                }
                            }
                        }// for iy
                    }
                } //for ix
            }"""
            varlist = ['number_of_ints', 'coverage', 'coverage_size', 'mask1', 'mask2', 'num_coverage', 'coverage_list']
            weave.inline(code, varlist, compiler='gcc', support_code = support)


        if self.verbose: print "instrument.total_coverage done in %s sec." % (time.time()-t1)

        return coverage













#========================================================================================================
#========================================================================================================
#========================================================================================================
class InstrumentInelastic(Instrument):
    """Inelastic scattering instrument; a subclass of Instrument.
    - Inelastic instruments are assumed to have a monochromatic input beam, and
        we consider scattering events with energy changes.
    - The wl_min and wl_max fields indicate the limits that the detectors can measure.
    - We build a 3D array representing reciprocal space. Each element will contain a list
        of the sample's energy change for that Q.
    """

    #Input neutron wavelength.
    wl_input = 1.0

    #-------------------------------------------------------------------------------------------
    def __init__ (self, filename=None, params=dict()):
        """Create an inelastic instrument. Will load the geometry from a supplied CSV file."""
        Instrument.__init__(self, filename, params)



    #========================================================================================================
    # C-code for calculate_coverage

    _code_calculate_coverage = """
            double kfz, kf_squared, E;

            double lim_min = -qlim;
            double lim_max = +qlim;
            
            //Loop through pixels using the list given before.
            for (int iix = 0; iix < xlist.length(); iix++)
            {
                int ix = xlist[iix];
                for (int iiy = 0; iiy < ylist.length(); iiy++)
                {
                    int iy = ylist[iiy];

                    //Angles of the detector pixel positions
                    double az, elev;
                    az = AZIMUTHAL_ANGLE2(iy, ix);
                    elev = ELEVATION_ANGLE2(iy, ix);

                    //Get the two limits.
                    py::tuple q_max_both = getq_inelastic(wl_input, wl_min, az, elev, pi, rot_matrix);
                    py::tuple q_min_both = getq_inelastic(wl_input, wl_max, az, elev, pi, rot_matrix);

                    // Rotated qs
                    double q_max[3], q_min[3], q_max_unrot[3], q_min_unrot[3];
                    double q_max_length = 0;
                    double q_min_length = 0;
                    for (int i=0; i<3; i++)
                    {
                        q_max[i] = q_max_both[i];
                        q_max_unrot[i] = q_max_both[i+3];
                        q_max_length += q_max[i]*q_max[i];

                        q_min[i] = q_min_both[i];
                        q_min_unrot[i] = q_min_both[i+3];
                        q_min_length += q_min[i]*q_min[i];
                    }
                    q_max_length = sqrt(q_max_length);
                    q_min_length = sqrt(q_min_length);

                    /*
                        //Limit them to the modeled size -- SHRINK VECTOR ---
                        double reduceby_max = 1.0;
                        double reduceby_min = 1.0;
                        if (q_max_length > qlim)
                            { reduceby_max = q_max_length/qlim; }
                        if (q_min_length > qlim)
                            { reduceby_min = q_min_length/qlim; }
                        for (int i=0; i<3; i++)
                        {
                            q_max[i] /= reduceby_max;
                            q_min[i] /= reduceby_min;
                            q_max_unrot[i] /= reduceby_max;
                            q_min_unrot[i] /= reduceby_min;
                        }
                    */

                    //Vector difference max-min
                    double q_diff[3];
                    double q_diff_unrot[3];
                    double q_length = 0.0;
                    for (int i=0; i<3; i++)
                    {
                        // We change the sign, because for inelastic, Q = ki-kf, rather than the opposite sign for elastic
                        q_diff[i] = q_max[i] - q_min[i];
                        q_diff_unrot[i] = q_max_unrot[i] - q_min_unrot[i];
                        q_length += q_diff[i]*q_diff[i];
                    }
                    
                    //Find out how long of a line that is
                    q_length = sqrt(q_length);

                    //How many steps will we take. The multiplication factor here is a fudge to make sure it covers fully.
                    long numfrac = (1.25 * q_length) / (q_resolution);

                    if (numfrac > 0)
                    {
                        //There is something to measure.

                        //Size in q-space of each step
                        double dx, dy, dz;
                        dx = q_diff[0] / numfrac;
                        dy = q_diff[1] / numfrac;
                        dz = q_diff[2] / numfrac;
                        
                        double dx_unrot, dy_unrot, dz_unrot;
                        dx_unrot = q_diff_unrot[0] / numfrac;
                        dy_unrot = q_diff_unrot[1] / numfrac;
                        dz_unrot = q_diff_unrot[2] / numfrac;

                        long index;
                        double qx, qy, qz;
                        double qx_unrot, qy_unrot, qz_unrot;
                        long iqx, iqy, iqz;

                        //Okay now we draw the line from q_min to q_max.
                        double i;
                        for (i=0; i<numfrac; i++)
                        {
                            //All of these qx checks might not be necessary anymore...?
                            qx = q_min[0] + i*dx;
                            qx_unrot = q_min_unrot[0] + i*dx_unrot;
                            iqx = long(round((qx - lim_min) / q_resolution));
                            if ((iqx >=0) && (iqx < stride))
                            {
                                qy = q_min[1] + i*dy;
                                qy_unrot = q_min_unrot[1] + i*dy_unrot;
                                iqy = long(round((qy - lim_min) / q_resolution));
                                if ((iqy >=0) && (iqy < stride))
                                {
                                    qz = q_min[2] + i*dz;
                                    qz_unrot = q_min_unrot[2] + i*dz_unrot;
                                    iqz = long(round((qz - lim_min) / q_resolution));
                                    if ((iqz >=0) && (iqz < stride))
                                    {
                                        //Calculate the neutron energy gain
                                        // But we need to get back the z component of kf
                                        // We kept Q = kf - ki
                                        kfz = (ki + qz_unrot);

                                        // Okay, now calculate kf^2
                                        kf_squared = qx_unrot*qx_unrot + qy_unrot*qy_unrot + kfz*kfz;

                                        // Get the energy. The constant sets the units (to meV)
                                        E = energy_constant * (kf_squared - ki_squared);

                                        COVERAGE3(iqx,iqy,iqz) = E; 
                                    }
                                }
                            }
                            
                        } //for i in numfrac
                    } //numfrac > 0 so we can draw the line

                } //for iiy
            }"""
            
    #========================================================================================================
    def calculate_coverage(self, det_list, angles, sample_U_matrix=np.identity(3), use_inline_c=True):
        """This method finds the coverage of all detectors in a full 3D matrix by drawing
        straight lines from the min. wavelength to the max. wavelength. This is, so far, the most
        efficient way I have found to calculate it.
        Optimized for speed using inline C code.

        Parameters:
            det_list: list of all the detectors (objects) to calculate.
            angles: sample orientation angles, in radians (usually), as a list.
            sample_U_matrix: U matrix of the sample mount angles.

        Returns:
            coverage: the q-space 3D array of coverage. Each point is a float holding
                the neutron energy gain (in eV) at that q.
                If the voxel was not measured, it holds +inf.
        """

        #------
        def shrink_two_q_vector(q, q2):
            """Helper function returns a shortened q-vector, capped
            to self.qlim but with the same direction."""
            q_length = vector_length(q)
            if q_length > self.qlim:
                q2_length = vector_length(q2)
                return ((q / q_length) * self.qlim, (q2 / q2_length) * self.qlim)
            else:
                return (q, q2)
        #------
        def shrink_q_vector(q):
            """Helper function returns a shortened q-vector, capped
            to self.qlim but with the same direction."""
            q_length = vector_length(q)
            if q_length > self.qlim:
                return (q / q_length) * self.qlim
            else:
                return q

        if not isinstance(det_list, list):
            raise ArgumentError("Did not supply a list as det_list.")
        if not (len(angles) == len(self.angles)):
            raise ArgumentError("You supplied a list of angles of a length that does not match what the instrument is setup for.")

        #For passing out messages
        angles_string = self.make_angles_string(angles)

        #Calculate the q-rotation matrix
        rot_matrix = self.goniometer.make_q_rot_matrix(angles)
        #Multiply by the inverse of the sample's U matrix.
        # qR = R * U^-1
        #   so that we undo the goniometer angle rotation 1st, then the sample mounting rotation
        #   = the opposite order of the rotations in the first case.
        rot_matrix = np.dot(np.linalg.inv(sample_U_matrix), rot_matrix)


        #The input wavevector (along +z)
        wl_input = self.wl_input
        ki = 2*np.pi / wl_input
        ki_squared = ki**2

        #A constant for calculating energy
        #(h_bar^2 / (2*m) )
        # also, k = 2pi/lambda, with lambda in angstroms
        # and to convert from A^-2 to m^-2, you have to x 1e20
        # and to convert from Joules to eV, you / 1.602177e-19
        # Finally we convert from eV to meV
        energy_constant = 1.6599e-42 * 1e20  / 1.602177e-19 * 1000 #Resulting unit = meV

        input_energy = energy_constant * ki_squared

        #Find the wavelength range to use
        (wl_min, wl_max) = self.get_wavelength_range(angles)

        #For updating
        last_time = time.time()

        #Start with +(Large Number) everywhere
        coverage = self.make_blank_qspace(np.float) + 1e6
        
        count = 0
        if self.verbose:
            sys.stdout.write( "For angles [%s], calculating coverage of detectors... " % angles_string)
            
        for det in det_list:
            #Make sure the detector object is valid
            if det is None:
                continue

            #Output and statusbar messages, if it's been long enough
            messages.send_message_optional(self, messages.MSG_UPDATE_MAIN_STATUSBAR, "Calculating coverage of detector '%s' at %s" % (det.name, angles_string))
            if (time.time() - last_time) > 0.33:
                last_time = time.time()
                if self.verbose: sys.stdout.write(det.name + ", ")
                if self.verbose: sys.stdout.flush()


            #Two nearby pixels
            q0 = getq(det.azimuthal_angle[0, 0], det.elevation_angle[0, 0], wl_min, rot_matrix, wl_input=wl_input)
            q_xmax = getq(det.azimuthal_angle[0, -1], det.elevation_angle[0, -1], wl_min, rot_matrix, wl_input=wl_input)
            q_ymax = getq(det.azimuthal_angle[-1, 0], det.elevation_angle[-1, 0], wl_min, rot_matrix, wl_input=wl_input)

            #Make sure they aren't too long
            q0 = shrink_q_vector(q0)
            q_xmax = shrink_q_vector(q_xmax)
            q_ymax = shrink_q_vector(q_ymax)

            #Project them to all the same length (whichever is the longest)
            length = max( vector_length(q_xmax),  vector_length(q_ymax),  vector_length(q0) )
            q0 = normalize_vector(q0, length)
            q_xmax = normalize_vector(q_xmax, length)
            q_ymax = normalize_vector(q_ymax, length)

            #The pixels to look at (dont look at all of them)
            nx = (vector_length(q_xmax - q0) / self.q_resolution) * 1.5
            ny = (vector_length(q_ymax - q0) / self.q_resolution) * 1.5

            #So this is the list of the pixels in the detectors that we pick to calculate where they are in q-space
            #   (convert to int and take only unique values; convert back to list).
            xlist = [x for x in set(np.linspace(0, det.xpixels-1, nx, endpoint=True).astype(int))]
            xlist.sort()
            ylist = [x for x in set(np.linspace(0, det.ypixels-1, ny, endpoint=True).astype(int))]
            ylist.sort()

            if use_inline_c and not config.cfg.force_pure_python:
                #------- Inline C ---------
                #Get some variables ready
                azimuthal_angle = det.azimuthal_angle
                elevation_angle = det.elevation_angle
                #Set up several functions used in the code
                support = "#include <math.h>\n"
                support += crystal_calc.getq_inelastic_code_header + crystal_calc.getq_inelastic_code + crystal_calc.getq_inelastic_code_footer
                #support += self._code_vector_length
                #support += self._code_shrink_q_vector
                #Dimensions of the array
                s = coverage.shape
                stride = s[0]
                max_index = s[0]*s[1]*s[2] #largest index into the array +1
                #Make the list of local vars to use.
                varlist = ['wl_input', 'xlist', 'ylist', 'azimuthal_angle', 'elevation_angle']
                varlist += ['pi', 'rot_matrix', 'energy_constant', 'ki_squared', 'ki']
                varlist += ['coverage', 'stride', 'max_index']
                #Dump these  in the locals namespace
                attribute_list = ['wl_min', 'wl_max', 'qlim', 'q_resolution']
                for var in attribute_list: locals()[var] = getattr(self, var)
                varlist += attribute_list
                #Run the C code (see between function declarations for the actual code).
                weave.inline(self._code_calculate_coverage, varlist, compiler='gcc', support_code=support) # , libraries = ['m'])

                print "C-code: neutron energy gain min", np.min(coverage), "; max", np.max(coverage[coverage <1e6])

            else:
#            if True:
                #-------- Pure Python ---------
                for ix in xlist:
                    for iy in ylist:
                        #The angles we find are the azimuth and elevation angle of the scattered beam.
                        az = det.azimuthal_angle[iy, ix]
                        elev = det.elevation_angle[iy, ix]

                        #For inelastic, we want Q = ki-kf = momentum transfer from neutron TO sample
                        (q_min, q_min_unrot) = getq(az, elev, self.wl_min, rot_matrix, wl_input=self.wl_input)
                        (q_max, q_max_unrot) = getq(az, elev, self.wl_max, rot_matrix, wl_input=self.wl_input)

#                        #Cap to qlim
#                        (q_min, q_min_unrot) = shrink_two_q_vector(q_min, q_min_unrot)
#                        (q_max, q_max_unrot) = shrink_two_q_vector(q_max, q_max_unrot)

                        #Length to walk through
                        q_diff = q_max-q_min
                        q_diff_unrot = q_max_unrot - q_min_unrot

                        #Calculate an optimal # of points to use in the q direction.
                        numfrac = int(1.25 * vector_length(q_diff) / self.q_resolution)
                        if numfrac > 0:
                            #If we get numfrac==0, that means nothing can be detected.
                            for i in xrange(numfrac):
                                #This is the intermediate q
                                q = q_min + i*q_diff/numfrac
                                q_unrot = q_min_unrot + i*q_diff_unrot/numfrac

                                #Find the indices in q-space that correspond
                                iqx = index_evenly_spaced(-self.qlim, len(self.qx_list), self.q_resolution, q[0])
                                if iqx is None: continue
                                iqy = index_evenly_spaced(-self.qlim, len(self.qx_list), self.q_resolution, q[1])
                                if iqy is None: continue
                                iqz = index_evenly_spaced(-self.qlim, len(self.qx_list), self.q_resolution, q[2])
                                if iqz is None: continue
                                #All indices are valid, hurrah!
                                #Now we calculate the energy gained by the NEUTRON.
                                # E = h_bar * omega = (h_bar^2 / (2*m) )*(kf^2 - ki^2)

                                #But we need to get back the z component of kf
                                #If Q = kf-ki (Busing Levy 1967 convention)
                                kfz = (ki + q_unrot[2])

                                #Okay, now calculate kf^2
                                kf_squared = q_unrot[0]**2 + q_unrot[1]**2 + kfz**2

                                #Get the energy. Units are in meV
                                E = energy_constant * (kf_squared - ki_squared)

                                #Set that energy in the array
                                coverage[iqx, iqy, iqz] = E
                print "Python-code: neutron energy gain min", np.min(coverage), "; max", np.max(coverage[coverage <1e6])

        if self.verbose: print " done!"

        return coverage



    #========================================================================================================
    def total_coverage(self, detectors_used, orientations_used, use_inline_c=True, slice_min=-100.0, slice_max=100.0):
        """Calculate the total coverage for an inelastic instrument, with some options, as bool arrays:

        Parameters:
            detectors_used: IGNORED for inelastic instrument!
            orientations_used: a list of PositionCoverage objects to add up together.
                If None, all the entries saved in the instrument are used.
            use_inline_c: bool, True to use inline C routine, false for pure Python.
            slice_min and slice_max: energy slice to take.

        Returns:
            coverage: a 3D array with the # of times each voxel was measured.
        """
        #print "instrument.total_coverage starting..."

        t1 = time.time()
        #Create the q-space
        coverage = self.make_blank_qspace(np.int16)

        #Error checking
        if self.positions is None: return None

        #Make sure the list of positions makes sense
        if orientations_used is None:
            orientations_used = self.positions #Just refer to the same list saved here.

        #Look through each orientation saved and add them up.
        coverage_list = list()
        for pos_cov in orientations_used:
            if pos_cov is None: continue
            if pos_cov.coverage is None: continue
            #Add the coverage of the position used to the list
            coverage_list.append(pos_cov.coverage)

        #Now we add up the coverages together
        if config.cfg.force_pure_python or not use_inline_c or True:
            #--- Pure Python Version ---
            for one_coverage in coverage_list:
                #Add one to the voxels where the energy is within the given range.
                coverage += (one_coverage[:,:,:] >= slice_min) & (one_coverage[:,:,:] <= slice_max)

        else:
            #--- Inline C version ---
            #   about 70x faster than the python version.
            coverage_size = coverage.size
            num_coverage = len(coverage_list)
            support = ""
            code = """
            int i, j;
            PyArrayObject* each_coverage[num_coverage];
            for (j=0; j<num_coverage; j++)
                each_coverage[j] = (PyArrayObject*) PyList_GetItem(coverage_list, j);

            float low, high;

            for (j=0; j<num_coverage; j++)
            {
                npy_intp* Sone_coverage = each_coverage[j]->strides;

                for (int ix=0; ix < Ncoverage[0]; ix++)
                {
                    for (int iy=0; iy < Ncoverage[1]; iy++)
                    {
                        for (int iz=0; iz < Ncoverage[2]; iz++)
                        {
                            //Only 1 int
                            //Index into the coverage array
                            low = (*((float*)(each_coverage[j]->data + (ix)*Sone_coverage[0] + (iy)*Sone_coverage[1] + (iz)*Sone_coverage[2])));
                            if (low < 1e6)
                            {
                                COVERAGE3(ix,iy,iz)++;
                                //COVERAGE3(ix,iy,iz) = low;
                            }
                        }// for iy
                    }
                } //for ix


            }"""
            varlist = ['number_of_ints', 'coverage', 'coverage_size', 'num_coverage', 'coverage_list']
            weave.inline(code, varlist, compiler='gcc', support_code = support)


        if self.verbose: print "instrument.total_coverage done in %s sec." % (time.time()-t1)

        return coverage






#========================================================================================================
#========================================================================================================
#========================================================================================================
class InstrumentFourCircle(Instrument):
    """ Class for a four-circle diffractometer (also could be applied to
    triple-axis.
    Features:
        - Fixed incoming wavelength
            - Possible contamination wavelengths
        - Movable detector (typically single pixel)
    """

    def __init__(self):
        """Constructor"""

        #Init the base instrument with no file
        Instrument.__init__(self, None, dict())
        
        # Input wavelength in angstroms
        self.wl_input = 1.01

        self.wl_min = 0.95
        self.wl_max = 1.05
        
        # Add the single pixel detector
        #Dimensions in mm
        #@type det FlatDetector
        det = FlatDetector("single-pixel")
        det.distance = 300
        det.elevation_center = 0
        det.azimuth_center = np.pi/4
        det.rotation = 0
        det.width = 10
        det.height = 10
        det.xpixels = 10
        det.ypixels = 10
        #Calculate the pixel angles
        det.calculate_pixel_angles()
        self.detectors.append(det)

    def total_coverage(self, *args, **kwargs):
        print "total_coverage not implemented for the Four-Circle instrument"

#    def simulate_position(self, *args, **kwargs):
#        print "simulate_position not implemented for the Four-Circle instrument"

    def calculate_coverage(self, *args, **kwargs):
        print "calculate_coverage not implemented for the Four-Circle instrument"
        return np.array([])



             
    
#==============================================================================
inst = Instrument





#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest




#==================================================================
class TestInelasticInstrument(unittest.TestCase):
    """Unit test for the InstrumentInelastic class."""
    def setUp(self):
        config.cfg.force_pure_python = False
        self.tst_inst = InstrumentInelastic("../instruments/TOPAZ_geom_all_2011.csv")
        ti = self.tst_inst #@type ti InstrumentInelastic
        ti.set_goniometer(goniometer.Goniometer())
        ti.d_min = 1.0
        ti.q_resolution = 0.5
        ti.wl_min = 0.9
        ti.wl_max = 10.0
        ti.wl_input = 1.0
        ti.make_qspace()

    def test_coverage(self):
        ti = self.tst_inst #@type ti InstrumentInelastic
        assert hasattr(ti, 'wl_input'), "wl_input field exists."
        angles = [0., 0., 0.]
        cov_python = ti.calculate_coverage(ti.detectors[:1], angles, use_inline_c=False)
        cov_C = ti.calculate_coverage(ti.detectors[:1], angles, use_inline_c=True)
        tot_python = np.sum( cov_python < 2e6)
        tot_C = np.sum( cov_C < 2e6)
        assert tot_python==tot_C, "Same # covered found with C and python. %d vs %d" % (tot_C, tot_python)
        diff =  cov_C - cov_python
        print "sum of diff",np.sum(diff)
        print "# diff",np.sum(abs(diff) > 1e-10)
        print diff[abs(diff) > 1e-10]
#        assert np.allclose(cov_C, cov_python), "Energy values found with C and python are close within float error."
#        old_min = np.min(cov_C)
#
#        angles = [0.3, 1.20, -1.23]
#        cov_python = ti.calculate_coverage(ti.detectors[:1], angles, use_inline_c=False)
#        cov_C = ti.calculate_coverage(ti.detectors[:1], angles, use_inline_c=True)
#        tot_python = np.sum( cov_python < 2e6)
#        tot_C = np.sum( cov_C < 2e6)
#        assert tot_python==tot_C, "Same # covered found with C and python. %d vs %d" % (tot_C, tot_python)
#        assert np.allclose(cov_C, cov_python), "Energy values found with C and python are close within float error."
#        assert np.allclose(np.min(cov_C), old_min), "Same minima found after a rotation."

    def test_simulate_position(self):
        ti = self.tst_inst #@type ti InstrumentInelastic
        angles = [0., 0., 0.]
        ti.simulate_position(angles, sample_U_matrix=np.eye(3), use_multiprocessing=False)
        total = ti.total_coverage(ti.detectors, ti.positions, use_inline_c=True)
        cov = ti.positions[0].coverage
        total_measured = np.sum( total > 0)
        total_coverage = np.sum( cov < 1e6)
        print cov.size
        assert total_measured==total_coverage, "Measured the same # of voxels. We found %d in total_coverage, vs %d in the individual coverage" % (total_measured, total_coverage)




#==================================================================
class TestInstrumentWithDetectors(unittest.TestCase):
    """Unit test for the Instrument class."""
    def setUp(self):
        self.tst_inst = Instrument("../instruments/TOPAZ_detectors_2010.csv")
        self.tst_inst.set_goniometer(goniometer.Goniometer())
        self.tst_inst.d_min = 0.7
        self.tst_inst.q_resolution = 0.15
        self.tst_inst.wl_min = 0.7
        self.tst_inst.wl_max = 3.6

    def test_creation(self):
        """test_creation: Full test of Instrument creation and .calculate_coverage()"""
        empty_instrument = Instrument("file_does_not_exist.csv")
        assert len(empty_instrument.detectors)==0, "Create an Instrument with a file not found, and you get no error but an empty detectors list."
        tst_inst = self.tst_inst
        assert len(tst_inst.detectors) == 14, "Correct # of detectors"
        assert isinstance(tst_inst.detectors[0], Detector), "Detectors created."
        #-- Check q-space creation --
        tst_inst.make_qspace()
        assert not hasattr(tst_inst, 'qspace'), "Instrument should not have a qspace field."
        self.assertAlmostEquals(tst_inst.qlim, 2*np.pi/0.7, 10, "Correct qlim")
        #-- Check the radius array --
        assert tst_inst.qspace_radius.shape == (120, 120, 120), "Correct qspace_radius shape."
        assert np.allclose(tst_inst.qspace_radius[0,0,0], np.sqrt(3 * tst_inst.qlim**2)) , "qspace_radius value tested"

        #Keep going
        self.do_calculate_coverage(more_det=False)
        

    def test_double_creation(self):
        assert len(self.tst_inst.detectors) == 14, "Correct # of detectors after first creation"
        self.tst_inst = Instrument("../instruments/TOPAZ_geom_all_2011.csv")
        assert len(self.tst_inst.detectors) == 14, "Correct # of detectors after second creation"


#    def test_calculate_coverage_more_detectors(self):
#        self.tst_inst = Instrument("../instruments/TOPAZ_detectors_all.csv")
#        self.tst_inst.set_goniometer(goniometer.Goniometer())
#        self.tst_inst.d_min = 0.7
#        self.tst_inst.q_resolution = 0.15
#        self.tst_inst.wl_min = 0.7
#        self.tst_inst.wl_max = 3.6
#        self.tst_inst.make_qspace()
#        assert len(self.tst_inst.detectors) == 48, "Correct # of detectors after second creation"
#        self.do_calculate_coverage(more_det=True)

    def do_calculate_coverage(self, more_det=False):
        tst_inst = self.tst_inst
        angles = [0.1, 0.2, 0.3]
        self.assertRaises(ArgumentError, tst_inst.calculate_coverage, [], [1, 2])
        self.assertRaises(ArgumentError, tst_inst.calculate_coverage, None, [1, 2, 3, 4])
        ret = tst_inst.calculate_coverage([None], angles)
        assert not np.any(ret), "calculate_coverage returns 0-initialized array for no detectors."
        #Make a mostly empty list
        det_list = [None for x in xrange(48)]
        my_nums = [0,1]
        if more_det: my_nums += [30, 31, 32, 45]
        for x in my_nums:
            det_list[x] = tst_inst.detectors[x]
            
        for use_inline_c in [True, False]:
            #Calculate it (using C)
            msg = ["(python only)", "(inline C"][use_inline_c]
            ret = tst_inst.calculate_coverage(det_list, angles, use_inline_c=use_inline_c)
            assert np.any(ret == 1), "Coverage %s has some bits equal to 1." % msg
            assert np.any(ret == 2), "Coverage %s has some bits equal to 2." % msg
            assert not np.any(ret == 4), "Coverage %s has no bits equal to 4." % msg
            if more_det: assert np.any(ret == 2**(30)), "Coverage %s has some bits equal to 2**30." % msg
            if more_det: assert np.any(ret == 2**(45-31)), "Coverage %s has some bits equal to 2**45-31." % msg
            if use_inline_c: c_ret = ret
        #Find differences
        (x,y,z,byte) = np.nonzero(ret != c_ret)
        #assert abs(np.sum(ret > 0) - np.sum(c_ret > 0)) < 10, "Number of points found (C vs Python) match within 10. C: %d,  "
        assert len(x) < 20, "Coverage calculated by C and python match within 20 differences. We found %s differences" % len(x)
        #Do a full list at 0
        ret = tst_inst.calculate_coverage(tst_inst.detectors, [0, 0, 0], use_inline_c=True)
        found = np.sum(ret > 0)
        wanted = 55870
        if more_det: wanted = 167077
        assert found == wanted, "%d points found with %d detectors. We expected %d points" % (found, len(tst_inst.detectors), wanted)

    def test_simulate_and_total(self):
        """test_simulate_and_total: test of simulate_coverage() and total_coverage()"""
        tst_inst = self.tst_inst
        tst_inst.positions = []
        tst_inst.make_qspace()
        assert not hasattr(tst_inst, 'qspace'), "Instrument should not have a qspace field (anymore)."
        
        phi_list = np.deg2rad([0, 45, 90, 135])
        (valid, redundant, invalid, invalid_reason) = tst_inst.evaluate_position_list([ phi_list, [0], [0] ], ignore_gonio=True)
        assert len(valid) == 4, "evaluate_position_list valid entries."
        assert len(redundant) == 0, "evaluate_position_list redundant entries."
        assert len(invalid) == 0, "evaluate_position_list invalid entries."
        for angles in valid:
            print "test: simulating angles", angles
            tst_inst.simulate_position(angles, use_multiprocessing=False)
        assert len(tst_inst.positions)==4, "There should be 4 positions saved. There are %d" % len(tst_inst.positions)
        total = 0
        for (index, poscov) in enumerate(tst_inst.positions):
            assert isinstance(poscov, PositionCoverage), "PositionCoverage object instance"
            assert np.all( poscov.angles == valid[index]), "Matching angles saved."
            total += np.sum(poscov.coverage != 0)
            assert isinstance(poscov, PositionCoverage), "PositionCoverage object instance"
        #Do the total coverage
        print "Total coverage test 1"
        cov = tst_inst.total_coverage([False], tst_inst.positions)
        assert np.sum(cov)==0, "total_coverage() should return an empty array when no detectors are used."
        print "Total coverage test 2"
        cov = tst_inst.total_coverage(None, [])
        assert np.sum(cov)==0, "total_coverage() should return an empty array when no positions are used."
        print "Total coverage test 3"
        cov = tst_inst.total_coverage(None, tst_inst.positions)
        cov_sum = np.sum(cov)
        expected = 223107
        assert cov_sum==expected, "total_coverage() total should be %d for these settings, but we got %s." % (expected, cov_sum)
        assert np.sum(cov)==total, "total_coverage() total (%s) should match the total those of each individual position (%s)." % (cov_sum, total)
        #Compare without inline_c
        print "Total coverage test 4"
        cov_python = tst_inst.total_coverage(None, tst_inst.positions, use_inline_c=False)
        assert np.all(cov == cov_python), "total_coverage() gives the same using pure Python as with inline C. There are %d differences." % (np.nonzero(cov == cov_python)[0].size)

#    def test_simulate_and_total_more_detectors(self):
#        self.tst_inst = Instrument("../instruments/TOPAZ_geom_all_2011.csv")
#        tst_inst = self.tst_inst
#        tst_inst.set_goniometer(goniometer.Goniometer())
#        tst_inst.d_min = 0.7
#        tst_inst.q_resolution = 0.15
#        tst_inst.wl_min = 0.7
#        tst_inst.wl_max = 3.6
#        tst_inst.make_qspace()
#        tst_inst.positions = []
#        assert len(self.tst_inst.detectors) == 48, "Correct # of detectors for test_simulate_and_total_more_detectors"
#        angles = [0,0,0]
#        tst_inst.simulate_position(angles, use_multiprocessing=False)
#        assert len(tst_inst.positions)==1, "There should be 1 position saved. There are %d" % len(tst_inst.positions)
#        #Total coverage of all detectors
#        cov = tst_inst.total_coverage([True]*48, tst_inst.positions)
#        cov_sum = np.sum(cov)
#        position_coverage_sum = np.sum( (tst_inst.positions[0].coverage[:,:,:,0] | tst_inst.positions[0].coverage[:,:,:,1])   != 0 )
#        assert cov_sum==position_coverage_sum, "The sum of total_coverage() and that of of all non-zero elements in the PositionCoverage object matches."
#        cov_python = tst_inst.total_coverage([True]*48, tst_inst.positions, use_inline_c=False)
#        cov_python_sum = np.sum(cov)
#        assert cov_python_sum==position_coverage_sum, "The sum of total_coverage(use_inline_c=False) and that of of all non-zero elements in the PositionCoverage object matches."
#        #Feed it too many detector positions to check!
#        cov = tst_inst.total_coverage([True]*100, None)
#        #Same thing if you give it None for the detector list
#        cov = tst_inst.total_coverage(None, None)

    def test_pickle(self):
        tst_inst = self.tst_inst
        #Do a couple of calcs
        tst_inst.make_qspace()
        tst_inst.simulate_position([1,2,3], use_multiprocessing=False)
        tst_inst.simulate_position([4,5,6], use_multiprocessing=False)
        datas = dumps(tst_inst)
        print "Length of dumped is ", len(datas)
        tst_inst2 = loads(datas)
        assert tst_inst==tst_inst2, "Matching instruments before and after file load."

    def test_load_detcal(self):
        self.tst_inst = Instrument("../instruments/TOPAZ_2010_06_08.detcal")

        



#==================================================================
class TestFourCircleInstrument(unittest.TestCase):
    """Unit test for the 4-circle class."""
    def setUp(self):
        config.cfg.force_pure_python = False
        self.tst_inst = InstrumentFourCircle()
        #@type ti InstrumentFourCircle
        ti = self.tst_inst
        ti.set_goniometer(goniometer.HB3AGoniometer())
        ti.d_min = 1.0
        ti.q_resolution = 0.5
        ti.make_qspace()

    def test_creation(self):
        #@type ti InstrumenFourCircle
        ti = self.tst_inst
        assert len(ti.detectors) == 1
        
#==================================================================
class TestImagine(unittest.TestCase):
    """Unit test for the 4-circle class."""
    def setUp(self):
        self.tst_inst = Instrument("../instruments/IMAGINE_detectors.csv")
        self.tst_inst.set_goniometer(goniometer.ImagineGoniometer())
        self.tst_inst.d_min = 0.7
        self.tst_inst.q_resolution = 0.15
        self.tst_inst.wl_min = 0.7
        self.tst_inst.wl_max = 3.6

    def test_creation(self):
        #@type ti InstrumenFourCircle
        ti = self.tst_inst
        assert len(ti.detectors) == 4
        
    def test_calculate_coverage(self):
        tst_inst = self.tst_inst
        self.assertRaises(ArgumentError, tst_inst.calculate_coverage, [], [1, 2])
        self.assertRaises(ArgumentError, tst_inst.calculate_coverage, None, [1, 2, 3, 4])

        # Intialize        
        tst_inst.make_qspace()

        #Only one detector
        det_list = [tst_inst.detectors[0]]
        # Phi = 0
        angles = [0]
            
        for use_inline_c in [False, True]:
            #Calculate it (using C)
            msg = ["(python only)", "(inline C)"][use_inline_c]
            ret = tst_inst.calculate_coverage(det_list, angles, use_inline_c=use_inline_c)
            assert np.any(ret == 1), "Coverage %s has some bits equal to 1." % msg
            if use_inline_c: c_ret = ret
            
        #Find differences
        (x,y,z,byte) = np.nonzero(ret != c_ret)
        assert len(x) < 20, "Coverage calculated by C and python match within 20 differences. We found %s differences" % len(x)

#---------------------------------------------------------------------
if __name__ == "__main__":
#    #Test just the inelastic one
    suite = unittest.makeSuite(TestImagine)
    unittest.TextTestRunner().run(suite)

#    tst = TestInstrumentWithDetectors('test_load_detcal')
#    tst.setUp()
#    tst.test_load_detcal()

#    unittest.main()

#    test_setup()
#    test_calculate_coverage()

