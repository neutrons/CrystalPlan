"""
Class definitions for detector geometries.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import numpy as np
import weave

#--- Model Imports ---
import numpy_utils
import utils
from numpy_utils import rotation_matrix, x_rotation_matrix, column, vector_length


#========================================================================================================
#========================================================================================================
#========================================================================================================
class Detector:
    """Base class for detector geometry."""

    #---------------------------------------------------------------------------------------------
    def __init__(self, detector_name):
        """Constructor for a FlatDetector object.
        """
        #Name of the detector
        self.name = detector_name

        #List of 3-D vectors giving the coordinates of each corner of the detector,
        #or other points of the outline for non-rectangular detectors.
        #   Used for plotting and others.
        self.corners = list()

        #Pixels in x and y (width and height)
        self.xpixels = 256
        self.ypixels = 256

        #Time-of-flight pixels
        self.tof_pixels = 2000

        #Azimuthal angle (along the horizontal plane) of each pixel
        # Coordinates are [Y, X]
        self.azimuthal_angle = None

        #Elevation angle (above the horizontal plane) of each pixel
        # Coordinates are [Y, X]
        self.elevation_angle = None

        #Time of flight "pixels" in ??? seconds
        self.time_of_flight = None

        #3 by Y by X, where the first dimension gives a 3D vector of the pixel XYZ position
        self.pixels = None

    #========================================================================================================
    def __eq__(self, other):
        return utils.equal_objects(self, other)

    #-------------------------------------------------------------------------------
    def get_pixel_direction(self, horizontal, vertical):
        """Return a 3x1 column vector giving the direction of a pixel
        on the face of the detector.

        Parameters:
            horizontal, vertical: position in mm on the face of the detector.
                0,0 = center of the detector.

        Returns:
            3x1 column vector of the beam direction, normalized."""
        raise NotImplementedError("Detector base class virtual method get_pixel_direction was called.")

#========================================================================================================
#========================================================================================================
#========================================================================================================
class FlatDetector(Detector):
    """Base class for a flat (planar) detector."""

    #---------------------------------------------------------------------------------------------
    def __init__(self, detector_name):
        """Constructor for a FlatDetector object.
        """
        Detector.__init__(self, detector_name)
        #Azimuth angle  (along the horizontal plane, relative to positive Z direction) of the center. Radians
        self.azimuth_center = 0

        #elevation angle of the center (above the XZ plane). Radians
        self.elevation_center = 0

        #Distance from center (mm)
        self.distance = 400

        #3D Vector, columnar, defining the position of the center of the detector.
        #The detector face is assumed to be normal to this vector.
    #    self.center_vector = column([0, 0, 1000])

        #Width and height of the detector (mm)
        self.width = 100
        self.height = 100

        #Angle (rad) between the "width" dimension and the horizon (X-Z) plane.
        #aka: the rotation of the detector around the center_vector.
        self.rotation = 0

        #Vector normal to the surface, pointing away from the sample.
        #   along with one corner, defines the plane equation.
        self.normal = None

        #Vector defining the horizontal (or X-axis) direction in the plane (normalized)
        self.horizontal = None

        #Vector defining the vertical (or Y-axis) direction in the plane (normalized)
        self.vertical = None

        #XYZ position of the CENTER of the detector.
        self.base_point = None

        #Rotation matrix to use to convert detector horiz, vert to XYZ in real space
        self.pixel_rotation_matrix = np.identity(3)


    #-------------------------------------------------------------------------------
    def edge_avoidance(self, h, v, edge_x, edge_y):
        """Returns a value from 0.0 to 1.0 where 1.0 = away from edge and 0 = just at the edge.
        Assumes h,v are actually on the detector otherwise.

        Parameters:
            h,v: horizontal and vertical position on detector. 0 = center
            edge_x, edge_y: edge size, in mm, on either side of x and y.
        """
        value = 1.0
        if edge_x > 0:
            edge_x = 1.0 * edge_x
            wid = self.width/2-edge_x
            if h > wid:
                value -= (h-wid)/edge_x
            if h < -wid:
                value -= (-h-wid)/edge_x

        if edge_y > 0:
            edge_y = 1.0 * edge_y
            hei = self.height/2-edge_y
            if v > hei:
                value -= (v-hei)/edge_y
            if v < -hei:
                value -= (-v-hei)/edge_y
        #Return the value, clipping to 0
        if value < 0:
            return 0
        else:
            return value

    #-------------------------------------------------------------------------------
    def detector_coord(self, az, elev):
        """Returns the coordinates of the point(s) touched on the detector (if any).
            az, elev: azimuth and elevation angle of the scattered beam direction.
                    Can be arrays, whose shapes need to match.
        Returns: (x,y,d)
            x,y: position in mm, in the detector's planar coordinates, relative to the base_point.
            d: distance from the origin to that point on the detector.
        """

        #Find the normalized direction vector(s) for this direction
        (bx, by, bz) = az_elev_direction(az, elev)
        #This beam coincides with the origin (0,0,0)
        #Therefore the line equation is x/beam[0] = y/beam[1] = z/beam[2]

        #We find where az, elev intersects the plane.
        (nx, ny, nz) = self.normal.reshape(3)
        (x0, y0, z0) = self.base_point.reshape(3)

        # A minimal value to check for non-zeroness
        min = 1e-6

        #Now we look for the intersection between the plane of normal nx,ny,nz and the given angle.
        n_dot_base = nx*x0 + ny*y0 + nz*z0

        print "bz is zero?", np.sum(bz<1e-10)

        if type(bx) == np.ndarray:
            #Vectorized way
            z = n_dot_base / (nx*bx/bz + ny*by/bz + nz)
            temp = (z / bz)
            y = by * temp
            x = bx * temp
            #TODO: Fix any nans?

        else:
            #Avoid divide by zero errors by using one of 3 possible methods to find the intersect
            if abs(bz) > min and abs(nz) > min:
                z = n_dot_base / (nx*bx/bz + ny*by/bz + nz)
                temp = (z / bz)
                y = by * temp
                x = bx * temp
            elif abs(by) > min and abs(ny) > min:
                y = n_dot_base / (nx*bx/by + ny + nz*bz/by)
                temp = (y / by)
                x = bx * temp
                z = bz * temp
            elif abs(bx) > min and abs(nx) > min:
                x = n_dot_base / (nx + ny*by/bx + nz*bz/bx)
                temp = (x / bx)
                y = by * temp
                z = bz * temp
            else:
                #Huh-oh! No intersect (parallel line and plane)
                return (np.inf, np.inf, np.inf)

       #Difference between the detector base (0,0) point
        diff_x = x - self.base_point[0]
        diff_y = y - self.base_point[1]
        diff_z = z - self.base_point[2]

        #Project onto horizontal and vertical axes by doing a dot product
        h = self.horizontal[0]*diff_x + self.horizontal[1]*diff_y + self.horizontal[2]*diff_z
        v = self.vertical[0]*diff_x + self.vertical[1]*diff_y + self.vertical[2]*diff_z

        #Distance and x and y
        d = np.sqrt(x**2 + y**2 + z**2)
        return (h, v, d)



    #-------------------------------------------------------------------------------
    def get_detector_coordinates(self, beam, wl_min=0.0, wl_max=1e6):
        """Returns the coordinates of the point(s) touched on the detector (if any).

        Parameters:
        -----------
            beam: 3xN array with the first dimension giving a vector of the direction
                of the scattered beam. The length of the vector = 2*pi/wavelength of the neutron.
            wl_min, wl_max: float, minimum and maximum wavelength that the detector
                can measure, in Angstroms.
                
        Returns: (horizontal, vertical, wavelength, hits_it), which are all N-sized arrays.
        --------
            horizontal: horizontal position in the detector coordinates (in mm)
                        (relative to the base_point, the center of the detector)
            vertical: vertical position in the detector coordinates (in mm)
            wavelength: wavelength detected (in Angstroms)
            distance: distance between sample and spot on detector, in mm
            hits_it: a N-sized array of booleans, set to True for the points that hit
                the detector.

        Optimized using inline-C.
        """
        #Some checks
        if len(beam.shape) != 2 or beam.shape[0] != 3:
            raise ValueError("'beam' parameter has incorrect shape. Should be 3xN array")

        #This is used in a calculation below
        n_dot_base = np.dot(self.normal.flatten(), self.base_point.flatten())

        #output h,v, wl coordinate arrays
        (ignored, array_size) = beam.shape
        h_out = np.zeros( array_size )
        v_out = np.zeros( array_size )
        wl_out = np.zeros( array_size )
        distance_out = np.zeros( array_size )
        hits_it = np.zeros( array_size, dtype=bool )

        #The abs() call might be screwed up!
        support = """
        #define FLOAT float
        FLOAT absolute(FLOAT x)
        {
            if (x > 0.0)
            { return x; }
            else
            { return -x; }
        }
        """
        code = """
        FLOAT az, elev;
        FLOAT bx,by,bz;
        FLOAT x,y,z, temp;
        FLOAT h,v,d;
        FLOAT diff_x, diff_y, diff_z;

        //some vars
        FLOAT base_point_x = base_point[0];
        FLOAT base_point_y = base_point[1];
        FLOAT base_point_z = base_point[2];
        FLOAT horizontal_x = horizontal[0];
        FLOAT horizontal_y = horizontal[1];
        FLOAT horizontal_z = horizontal[2];
        FLOAT vertical_x = vertical[0];
        FLOAT vertical_y = vertical[1];
        FLOAT vertical_z = vertical[2];
        FLOAT nx = normal[0];
        FLOAT ny = normal[1];
        FLOAT nz = normal[2];
        FLOAT n_dot_base_f = FLOAT(n_dot_base);

        int i;
        int error_count = 0;
        int bad_beam = 0;
        FLOAT projection, beam_length,  wavelength;

        for (i=0; i<array_size; i++)
        {
            //Good beam, nice beam.
            bad_beam = 0;

            // Non-normalized beam direction
            bx=BEAM2(0,i);
            by=BEAM2(1,i);
            bz=BEAM2(2,i);

            // So we normalize it
            beam_length = sqrt(bx*bx + by*by + bz*bz);
            bx = bx/beam_length;
            by = by/beam_length;
            bz = bz/beam_length;

            //Check if the wavelength is within range
            wavelength = 6.2831853071795862/beam_length;

            //If there are any nan's in the beam direction, this next check will return false.
            if ((wavelength <= wl_max) && (wavelength >= wl_min))
            {
                //Wavelength in range! Keep going.

                //Make sure the beam points in the same direction as the detector, not opposite to it
                // project beam onto detector's base_point
                projection = (base_point_x*bx)+(base_point_y*by)+(base_point_z*bz);
                if (projection > 0)
                {
                    //beam points towards the detector

                    //This beam coincides with the origin (0,0,0)
                    //Therefore the line equation is x/bx = y/by = z/bz

                    //Now we look for the intersection between the plane of normal nx,ny,nz and the given angle.

                    //Threshold to avoid divide-by-zero
                    FLOAT min = 1e-6;
                    if ((absolute(bz) > min)) // && (absolute(nz) > min))
                    {
                        z = n_dot_base_f / ((nx*bx)/bz + (ny*by)/bz + nz);
                        temp = (z / bz);
                        y = by * temp;
                        x = bx * temp;
                    }
                    else if ((absolute(by) > min)) //  && (absolute(ny) > min))
                    {
                        y = n_dot_base_f / (nx*bx/by + ny + nz*bz/by);
                        temp = (y / by);
                        x = bx * temp;
                        z = bz * temp;
                    }
                    else if ((absolute(bx) > min)) //  && (absolute(nx) > min))
                    {
                        x = n_dot_base_f / (nx + ny*by/bx + nz*bz/bx);
                        temp = (x / bx);
                        y = by * temp;
                        z = bz * temp;
                    }
                    else
                    {
                        // The scattered beam is 0,0,0
                        error_count += 1;
                        bad_beam = 1;
                    }
                }
                else
                {
                    //The projection is <0
                    // means the beam is opposite the detector. BAD BEAM! No cookie!
                    bad_beam = 1;
                }
            }
            else
            {
                //Wavelength is out of range. Can't measure it!
                bad_beam = 1;
            }


            if (bad_beam)
            {
                //A bad beam means it does not hit, for sure.
                h_out[i] = NAN;
                v_out[i] = NAN;
                wl_out[i] = wavelength; //This may be NAN too, for NAN inputs.
                hits_it[i] = 0;
            }
            else
            {
                //Valid beam calculation
                //Difference between this point and the base point (the center)
                diff_x = x - base_point_x;
                diff_y = y - base_point_y;
                diff_z = z - base_point_z;

                //Project onto horizontal and vertical axes by doing a dot product
                h = diff_x*horizontal_x + diff_y*horizontal_y + diff_z*horizontal_z;
                v = diff_x*vertical_x + diff_y*vertical_y + diff_z*vertical_z;

                // Save to matrix
                h_out[i] = h;
                v_out[i] = v;

                // the scattered beam is 1/wl long.
                wl_out[i] = wavelength;

                //What was the distance to the detector spot?
                distance_out[i] = sqrt(x*x + y*y + z*z);

                // And do we hit that detector?
                // Detector is square and our h,v coordinates are relative to the center of it.
                hits_it[i] = (v > -height/2) && (v < height/2) && (h > -width/2) && (h < width/2);
            }
        }
        return_val = error_count;"""

        #Generate a list of the variables used
        varlist = ['base_point', 'horizontal', 'vertical', 'normal']
        #Dump them in the locals namespace
        for var in varlist: locals()[var] = getattr(self, var).flatten()
        width = self.width
        height = self.height
        varlist += ['h_out', 'v_out', 'wl_out', 'distance_out', 'hits_it']
        varlist += ['beam', 'array_size', 'n_dot_base', 'height', 'width', 'wl_min', 'wl_max']
        #Run the code
        error_count = weave.inline(code, varlist, compiler='gcc', support_code = support,libraries = ['m'])

        #if error_count>0: print "error_count", error_count
#        positions = np.concatenate( (h_out, v_out, wl_out), 0)
        return (h_out, v_out, wl_out, distance_out, hits_it)




    #-------------------------------------------------------------------------------
    def calculate_pixel_angles(self):
        """Given the center angle and other geometry of the detector, calculate the
        azimuth and elevation angle of every pixel."""

        x = np.linspace(-self.width/2, self.width/2, self.xpixels)
        y = np.linspace(-self.height/2, self.height/2, self.ypixels)
        #Starting x, y, z position
        (px, py) = np.meshgrid(x,y)

        #Start with a Z equal to the given distance
        pz = np.zeros( px.shape ) + self.distance

        #Reshape into a nx3 buncha columns, where the columns are the pixels XYZ positions
        num = self.xpixels * self.ypixels
        pixels = np.array( [ px.flatten(), py.flatten(), pz.flatten()] )

        #Ok, first rotate the detector around its center by angle.
        #Since this is rotation around z axis, it is a chi angle
        rot = rotation_matrix(0, self.rotation, 0)

        #Now do the elevation rotation, by rotating around the x axis.
        #   Elevation is positive above the horizontal plane - means the x_rotation angle has to be negative
        rot = np.dot(x_rotation_matrix(-self.elevation_center), rot)

        #Finally add an azimuthal rotation (around the y axis, or phi)
        rot = np.dot(rotation_matrix(self.azimuth_center, 0, 0), rot)

        #Save it for later
        self.pixel_rotation_matrix = rot

        #This applies to the pixels
        pixels = np.dot(rot, pixels)

        #Save em - for plotting, mostly. Indices go Z, Y, X
        self.pixels = np.reshape(pixels, (3, self.ypixels, self.xpixels) )

        #Save the corners
        self.corners = list()
        self.corners.append(self.pixels[:,  0,  0]) #One corner
        self.corners.append(self.pixels[:,  0, -1]) #The end in X
        self.corners.append(self.pixels[:, -1, -1]) #Max x and Y
        self.corners.append(self.pixels[:, -1,  0]) #The end in Y

        #This is the base point - the center of the detector
        base_pixel = column([0, 0, self.distance])
        #Do the same rotation as the rest of the pixels
        base_pixel = np.dot(rot, base_pixel)
        self.base_point = column(base_pixel)

        #Horizontal and vertical vector
        self.horizontal = column(self.corners[1] - self.corners[0])
        self.vertical = column(self.corners[3] - self.corners[0])
        #Normalize them
        self.vertical /= vector_length(self.vertical)
        self.horizontal /= vector_length(self.horizontal)

        #Normal vector: take a vector pointing in positive Z, and the do the same rotation as all the pixels.
        self.normal = column(np.cross(self.horizontal.flatten(), self.vertical.flatten() ))
        assert np.allclose(vector_length(self.normal), 1.0), "normal vector is normalized."
        #Another way to calculate the normal. Compare it
        other_normal = np.dot(rot,  column([0,0,1])).flatten()
        assert np.allclose(self.normal.flatten(), other_normal), "Match between two methods of calculating normals. %s vs %s" % (self.normal.flatten(), other_normal)

        #Now, we calculate the azimuth and elevation angle of each pixel.
        x=pixels[0,:]
        y=pixels[1,:]
        z=pixels[2,:]

        self.azimuthal_angle = np.reshape( np.arctan2(x, z), (self.ypixels, self.xpixels) )
        self.elevation_angle = np.reshape( np.arctan(y / np.sqrt(x**2 + z**2)), (self.ypixels, self.xpixels) )



    #-------------------------------------------------------------------------------
    def calculate_pixel_angles_using_vectors(self, center, base, up):
        """Calculate the pixels using specified vectors:
        Parameters
            center: position of center in mm
            base: horizontal direction
            up: vertical direction
        """
        #Normalize
        base /= vector_length(base)
        up /= vector_length(up)
        #Save the vectors
        self.base_point = column(center)
        self.horizontal = column(base)
        self.vertical = column(up)
        #Calculate the normal vector
        self.normal = np.cross(self.vertical.flatten(), self.horizontal.flatten())
        self.normal /= vector_length(self.normal)
        self.normal = column(self.normal)
        assert np.allclose(vector_length(self.normal), 1.0), "Normal vector is normalized to length 1.0"

        #Now let's make the pixels
        x = np.linspace(-self.width/2, self.width/2, self.xpixels)
        y = np.linspace(-self.height/2, self.height/2, self.ypixels)
        #Starting x, y, z position
        (px, py) = np.meshgrid(x,y)

        #XY is the horizontal, vertical position
        px = px.flatten()
        py = py.flatten()

        #Multiply by the base vectors and add the center
        pixels = px*self.horizontal + py*self.vertical + self.base_point

        #Save em - for plotting, mostly. Indices go Z, Y, X
        self.pixels = np.reshape(pixels, (3, self.ypixels, self.xpixels) )

        #Save the corners
        self.corners = list()
        self.corners.append(self.pixels[:,  0,  0]) #One corner
        self.corners.append(self.pixels[:,  0, -1]) #The end in X
        self.corners.append(self.pixels[:, -1, -1]) #Max x and Y
        self.corners.append(self.pixels[:, -1,  0]) #The end in Y

        #Now, we calculate the azimuth and elevation angle of each pixel.
        x=pixels[0,:]
        y=pixels[1,:]
        z=pixels[2,:]
        self.azimuthal_angle = np.reshape( np.arctan2(x, z), (self.ypixels, self.xpixels) )
        self.elevation_angle = np.reshape( np.arctan(y / np.sqrt(x**2 + z**2)), (self.ypixels, self.xpixels) )


    #---------------------------------------------------------------------
    def get_pixel_direction(self, horizontal, vertical):
        """Return a 3x1 column vector giving the direction of a pixel
        on the face of the detector.

        Parameters:
            horizontal, vertical: position in mm on the face of the detector.
                0,0 = center of the detector.

        Returns:
            3x1 column vector of the beam direction, normalized.
        """
        #Use the vectors to build the direction
        pixel = horizontal * self.horizontal + vertical * self.vertical + self.base_point

#        #Make a column array of the pixel position, with the z given by the
#        #   detector to sample distance
#        pixel = np.array([ [horizontal], [vertical], [self.distance] ])
#        #Perform the appropriate rotation, calculated before
#        pixel = np.dot(self.pixel_rotation_matrix , pixel)

        #Normalize
        pixel = pixel / vector_length(pixel)
        return pixel




#========================================================================================================
#========================================================================================================
#========================================================================================================
class CylindricalDetector(Detector):
    """Base class for a cylindrical detector."""
    
    # Distance = radius of the cylinder
    def get_distance(self):
        return self.radius
    distance = property(get_distance)
    
    # Width = arc length of the bottom of the cylinder
    def get_width(self):
        return self.radius * (self.angle_end - self.angle_start)
    width = property(get_width)

    #---------------------------------------------------------------------------------------------
    def __init__(self, detector_name):
        """Constructor for a CylindricalDetector object.
        """
        Detector.__init__(self, detector_name)
        self.origin = np.array([0,-225,0])
        self.radius = 200
        self.height = 450
        self.angle_start = 0
        self.angle_end = 1.57079638



    #-------------------------------------------------------------------------------
    def calculate_pixel_angles(self):
        """Given the center angle and other geometry of the detector, calculate the
        azimuth and elevation angle of every pixel."""

        # This is the "azimuthal angle"
        thetas = np.linspace(self.angle_start, self.angle_end, self.xpixels)
        # This is the height above 
        heights = np.linspace(0, self.height, self.ypixels)
        # Number of pixels
        num = self.xpixels * self.ypixels

        #Grid of angle and height
        (ptheta, pheight) = np.meshgrid(thetas,heights)
        # The circle is in the X-Z direction
        px = self.origin[0] + np.cos(ptheta) * self.radius
        pz = self.origin[2] + np.sin(ptheta) * self.radius
        # This is the height
        py = self.origin[1] + pheight
        
        #Reshape into a nx3 buncha columns, where the columns are the pixels XYZ positions
        pixels = np.array( [ px.flatten(), py.flatten(), pz.flatten()] )

        #Save em - for plotting, mostly. Indices go Z, Y, X
        self.pixels = np.reshape(pixels, (3, self.ypixels, self.xpixels) )

        #Save the corners
        self.corners = list()
        self.corners.append(self.pixels[:,  0,  0]) #One corner
        self.corners.append(self.pixels[:,  0, -1]) #The end in X
        self.corners.append(self.pixels[:, -1, -1]) #Max x and Y
        self.corners.append(self.pixels[:, -1,  0]) #The end in Y

        #Now, we calculate the azimuth and elevation angle of each pixel.
        x=pixels[0,:]
        y=pixels[1,:]
        z=pixels[2,:]

        self.azimuthal_angle = np.reshape( np.arctan2(x, z), (self.ypixels, self.xpixels) )
        self.elevation_angle = np.reshape( np.arctan(y / np.sqrt(x**2 + z**2)), (self.ypixels, self.xpixels) )


    #-------------------------------------------------------------------------------
    def get_detector_coordinates(self, beam, wl_min=0.0, wl_max=1e6):
        """Returns the coordinates of the point(s) touched on the detector (if any).

        Parameters:
        -----------
            beam: 3xN array with the first dimension giving a vector of the direction
                of the scattered beam. The length of the vector = 2*pi/wavelength of the neutron.
            wl_min, wl_max: float, minimum and maximum wavelength that the detector
                can measure, in Angstroms.
                
        Returns: (horizontal, vertical, wavelength, hits_it), which are all N-sized arrays.
        --------
            horizontal: horizontal position in the detector coordinates (in mm)
                        (relative to the center of the detector)
            vertical: vertical position in the detector coordinates (in mm, relative to center)
            wavelength: wavelength detected (in Angstroms)
            distance: distance between sample and spot on detector, in mm
            hits_it: a N-sized array of booleans, set to True for the points that hit
                the detector.

        Optimized using inline-C.
        """
        #Some checks
        if len(beam.shape) != 2 or beam.shape[0] != 3:
            raise ValueError("'beam' parameter has incorrect shape. Should be 3xN array")

#        #output h,v, wl coordinate arrays
#        (ignored, array_size) = beam.shape
#        h_out = np.zeros( array_size )
#        v_out = np.zeros( array_size )
#        wl_out = np.zeros( array_size )
#        distance_out = np.zeros( array_size )
#        hits_it = np.zeros( array_size, dtype=bool )
        
        # Beam directions
        x = beam[0,:]
        y = beam[1,:]
        z = beam[2,:]
        
        # Azimuthal angle of the beam, relative to the start of the detector
        az_angle = np.arctan2(x, z) - self.angle_start
        # Make sure all angles are above 0.
        below_zero = (az_angle < 0)
        az_angle[below_zero] += np.pi * 2.
        
        # Horizontal position = relative to the start angle, in mm 
        h_out = self.radius * az_angle - self.width / 2.
        
        # Height of the beam, relative to the bottom of the cylinder
        elev_angle = np.arctan(y / np.sqrt(x**2 + z**2))
        v_out = self.radius * np.tan(elev_angle) - self.origin[1] - self.height / 2.
        
        # Distance to the pixel
        pixely = self.radius * np.tan(elev_angle);
        distance_out = np.sqrt( self.radius**2 + pixely**2)
        
        # Now the wavelength
        beam_length = np.sqrt( x**2 + y**2 + z**2 )
        wl_out = 6.2831853071795862/beam_length;

        # Hits the detector if it is within the range
        H = self.height / 2.
        W = self.width / 2.
        hits_it = (v_out >= -H) & (v_out <= H) & (h_out >= -W) & (h_out <= W) \
                  & (wl_out >= wl_min) & (wl_out <= wl_max)
        
        # Return everything
        return (h_out, v_out, wl_out, distance_out, hits_it)


    #---------------------------------------------------------------------
    def get_pixel_direction(self, horizontal, vertical):
        """Return a 3x1 column vector giving the direction of a pixel
        on the face of the detector.

        Parameters:
            horizontal, vertical: position in mm on the face of the detector.
                0,0 = center of the detector

        Returns:
            3x1 column vector of the beam direction, normalized.
        """
        # Vertical position, relative to 0,0
        y = vertical + self.origin[1] + self.height/2.
        # Position in the XZ plane
        angle = horizontal / self.radius + self.angle_start + self.width/2.
        x = np.cos(angle) * self.radius
        z = np.sin(angle) * self.radius
        pixel = column([x, y, z])

        #Normalize
        pixel = pixel / vector_length(pixel)
        return pixel


    #---------------------------------------------------------------------
    def edge_avoidance(self, h, v, edge_x, edge_y):
        """Returns a value 1.0 since cylindrical detector is small and has no edges.
        With this function the default optimization works for this detector.
        Assumes h,v are actually on the detector otherwise.

        Parameters:
            h,v: horizontal and vertical position on detector. 0 = center
            edge_x, edge_y: edge size, in mm, on either side of x and y.
        """
        return 1.0



#================================================================================
#============================ UNIT TESTING ======================================
#================================================================================
import unittest

#==================================================================
class TestDetector(unittest.TestCase):
    """Unit test for the Detector base class."""
    def setUp(self):
        self.det = Detector("detector_name")

#==================================================================
class TestFlatDetector(unittest.TestCase):
    """Unit test for the FlatDetector base class."""
    #----------------------------------------------------
    def setUp(self):
        self.det = FlatDetector("my_name")
        assert self.det.name == "my_name"
        #Some default values
        assert self.det.xpixels == 256
        assert self.det.ypixels == 256
        assert self.det.distance == 400
        assert self.det.rotation == 0


    #----------------------------------------------------
    def test_calculate_pixel_angles(self):
        """Detector.calculate_pixel_angles() tests."""
        #Smaller size
        self.det.xpixels = 16
        self.det.ypixels = 10
        self.det.calculate_pixel_angles()

        #Check some shapes
        assert self.det.pixels.shape == (3, 10, 16)
        assert self.det.azimuthal_angle.shape == (10, 16)

        #What we expect
        az = np.fromstring("-0.12435499 -0.10791249 -0.0914112  -0.07485985 -0.0582673  -0.04164258 -0.02499479 -0.00833314  0.00833314  0.02499479  0.04164258  0.0582673  0.07485985  0.0914112   0.10791249  0.12435499", sep=" ")
        elev_x = np.fromstring("-0.12340447 -0.123639   -0.1238411  -0.12401028 -0.12414612 -0.12424829 -0.12431655 -0.12435072 -0.12435072 -0.12431655 -0.12424829 -0.12414612 -0.12401028 -0.1238411  -0.123639   -0.12340447", sep=" ")
        elev_y = np.fromstring("-0.12340447 -0.09617384 -0.06879943 -0.04132138 -0.01378076  0.01378076  0.04132138  0.06879943  0.09617384  0.12340447", sep=" ")
        assert np.allclose(az, self.det.azimuthal_angle[0, :]), "Azimuthal angles match."
        assert np.allclose(elev_x, self.det.elevation_angle[0, :]), "elevation angles match along x."
        assert np.allclose(elev_y, self.det.elevation_angle[:, 0]), "elevation angles match along y."
        #All azimuthal angle with same x index match
        for iy in xrange(1, self.det.ypixels):
            assert np.allclose(az, self.det.azimuthal_angle[iy, :]), "Azimuthal angles match in all y values."
        #But there is a difference in elev. angles
        assert np.all( abs(self.det.elevation_angle[:, 0] - self.det.elevation_angle[:, 1]) > 0), "Some small difference in elevation angles as you change x."

        #Corners is set
        assert np.allclose(self.det.corners[0], [ -50.,  -50.,  400.])
        assert np.allclose(self.det.corners[1], [  50.,  -50.,  400.])
        assert np.allclose(self.det.corners[2], [  50.,   50.,  400.])
        assert np.allclose(self.det.corners[3], [ -50.,   50.,  400.])

        assert np.allclose(self.det.base_point, column([0, 0, 400])), "Base point is in the center of the detector.";
        
        #Orientation vectors
        assert np.allclose( self.det.horizontal, column([1, 0, 0]) ), "Horizontal orientation vector is correct."
        assert np.allclose( self.det.vertical, column([0, 1, 0]) ), "Vertical orientation vector is correct."

    #----------------------------------------------------
    def test_calculate_pixel_angles_with_rotation(self):
        """Detector.calculate_pixel_angles() where the detector has a rotation angle."""
        #Smaller size
        self.det.xpixels = 16
        self.det.ypixels = 10
        self.det.rotation = np.deg2rad(45)
        self.det.calculate_pixel_angles()
        assert np.allclose( self.det.horizontal, np.sqrt(2)/2 * column([1, 1, 0]) ), "Orientation vector is correct, rotated 45 deg."
        assert np.allclose( self.det.vertical, np.sqrt(2)/2 * column([-1, 1, 0]) ), "Vertical orientation vector is correct, rotated."


    #----------------------------------------------------
    def test_calculate_pixel_angles_with_elevation(self):
        """Detector.calculate_pixel_angles() where the detector has an elevation angle."""
        #Smaller size
        self.det.xpixels = 16
        self.det.ypixels = 10
        self.det.elevation_center = np.deg2rad(45)
        self.det.rotation = np.deg2rad(0)
        self.det.calculate_pixel_angles()
        assert np.allclose( self.det.horizontal, column([1, 0, 0]) ), "Orientation vector is correct, elevation angle."
        assert np.allclose( self.det.vertical, np.sqrt(2)/2 * column([0, 1, -1]) ), "Vertical orientation vector is correct, elevation angle."


#==================================================================
from numpy import pi

class TestHitsFlatDetector(unittest.TestCase):
    """Unit test for the FlatDetector, checks what angles hit this detector"""
    #----------------------------------------------------
    def setUp(self):
        self.det = FlatDetector("my_name")
        assert self.det.name == "my_name"
        #Some default values
        assert self.det.xpixels == 256
        assert self.det.ypixels == 256
        assert self.det.distance == 400
        assert self.det.rotation == 0
        assert self.det.width == 100
        assert self.det.height == 100
        self.det.calculate_pixel_angles()

    def test_detector_coord(self):
        """FlatDetector.detector_coord()"""
        det = self.det
        func = det.get_detector_coordinates
        #Invalid parameter errors
        self.assertRaises(ValueError, func, np.zeros( (2,10) ))
        self.assertRaises(ValueError, func, np.zeros(3) )

        beam = column([10.0, 0.0, 0.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose( wl, 0.1), "Correct wavelength for one"
        beam = column([10.0, 20.0, 30.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose( wl, 1/np.sqrt(1400)), "Correct wavelength 2"

        beam = np.ones( (3, 15) )*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert h.shape == (15,), "Correct shape of h"
#
        beam = column([0.0, 0.0, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose( wl, 1.0), "Correct wavelength for one"

        beam = column([1.0, 0.0, 0.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.isnan(h), "H and V are nan since the beam is parallel to the detector plane."
        assert not np.any(hits_it), "... and it doesn't hit, of course."

        beam = column([0.0, 1.0, 0.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.isnan(h), "H and V are nan since the beam is parallel to the detector plane."
        assert not np.any(hits_it), "... and it doesn't hit, of course."

        beam = column([0.0, 0.0, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, 0.0), "H is right in the middle of the detector."
        assert np.allclose(v, 0.0), "V is right in the middle of the detector."
        assert np.allclose(distance, 400.0), "Distance is equal to the detector face distance."
        assert np.all(hits_it), "... and it hits it."

        beam = column([0.0, 0.05, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, 0.0), "H is right in the middle of the detector."
        assert np.all(v > 0.0), "V is above the middle of the detector."
        assert np.all(hits_it), "... and it hits it."
        
        beam = column([0.0, -0.05, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, 0.0), "H is right in the middle of the detector."
        assert np.all(v < 0.0), "V is below the middle of the detector."
        assert np.all(hits_it), "... and it hits it."

        beam = column([0.0, 0.13, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, 0.0), "H is right in the middle of the detector."
        assert np.all(v > 50.0), "V is above the detector edge."
        assert not np.all(hits_it), "... and it misses it."

        beam = column([0.05, 0.0, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.all(h > 0.0), "h is to the left of the middle of the detector."
        assert np.allclose(v, 0.0), "V is right in the middle of the detector."
        assert np.all(hits_it), "... and it hits it."

        beam = column([0.13, 0.0, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.all(h > 50.0), "H is to the left of the detector edge."
        assert np.allclose(v, 0.0), "V is right in the middle of the detector."
        assert not np.all(hits_it), "... and it misses it."

        beam = column([0.0, 0.0, 0.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.isnan(h), "Scattered beam is 0,0,0."
        assert not np.any(hits_it), "... and it doesn't hit, of course."
        
        #-- opposite direction ---
        beam = column([0.0, 0.0, -1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.isnan(h), "H misses since the beam points the other way."
        assert np.isnan(v), "H misses since the beam points the other way."
        assert not np.all(hits_it), "Does not hit because the beam is in the opposite direction."

        #--- Wavelength too small ----
        beam = column([0.0, 0.0, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam, wl_min=2.0, wl_max=10.0)
        assert not np.all(hits_it), "Misses detector, wl is too small."

        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam, wl_min=0.1, wl_max=0.5)
        assert not np.all(hits_it), "Misses detector, wl is too large."

    def test_get_pixel_direction(self):
        det = self.det #@type det Detector
        #Higher tolerance because it has to round to pixel positions
        dir = det.get_pixel_direction(0.0, 0.0)
        assert np.allclose(dir.flatten(), np.array([0,0,1]), atol=1e-2), "Dead center"
        dir = det.get_pixel_direction(-50.0, 0.0)
        assert np.allclose(dir.flatten(), np.array([-0.124,0,0.992]), atol=1e-2), "Left"
        dir = det.get_pixel_direction(-50.0, -50.0)
        assert np.allclose(dir.flatten(), np.array([-0.124,-0.124,0.984]), atol=1e-2), "Left-bottom"
        
    def test_edge_avoidance(self):
        det = self.det #@type det Detector
        assert np.allclose(det.edge_avoidance(45, 0, 10, 10), 0.5), "edge avoidance tests."
        assert np.allclose(det.edge_avoidance(-48, 0, 10, 10), 0.2), "edge avoidance tests."
        assert np.allclose(det.edge_avoidance(0, 45, 10, 10), 0.5), "edge avoidance tests."
        assert np.allclose(det.edge_avoidance(0, -48, 10, 10), 0.2), "edge avoidance tests."
        assert np.allclose(det.edge_avoidance(22, 33, 10, 10), 1.0), "edge avoidance tests."
        assert np.allclose(det.edge_avoidance(40, -40, 10, 10), 1.0), "edge avoidance tests."
        assert np.allclose(det.edge_avoidance(60, -80, 10, 10), 0.0), "edge avoidance tests."



#==================================================================
class TestCylindricalDetector(unittest.TestCase):
    """Unit test for the CylindricalDetector base class."""
    #----------------------------------------------------
    def setUp(self):
        self.det = CylindricalDetector("my_name")
        assert self.det.name == "my_name"
        #Some default values
        assert self.det.xpixels == 256
        assert self.det.ypixels == 256

    #----------------------------------------------------
    def test_calculate_pixel_angles(self):
        """Detector.calculate_pixel_angles() tests."""
        #Smaller size
        self.det.xpixels = 16
        self.det.ypixels = 10
        self.det.calculate_pixel_angles()

        #Check some shapes
        assert self.det.pixels.shape == (3, 10, 16)
        assert self.det.azimuthal_angle.shape == (10, 16)
        print self.det.azimuthal_angle
        print self.det.elevation_angle


    def test_detector_coord(self):
        """CylindricalDetector.get_detector_coordinates()"""
        det = self.det
        func = det.get_detector_coordinates
        #Invalid parameter errors
        self.assertRaises(ValueError, func, np.zeros( (2,10) ))
        self.assertRaises(ValueError, func, np.zeros(3) )

        beam = column([10.0, 0.0, 0.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(wl, 0.1), "Correct wavelength"

        beam = column([1.0, 0.0, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, 0 * np.pi / 4, atol=1e-5), "H is right in the middle of the detector."
        assert np.allclose(v, 0.0), "V is right in the middle of the detector."
        assert np.allclose(distance, 200.0), "Distance is equal to the detector radius."
        assert np.all(hits_it), "... and it hits it."
        
        beam = column([-1.0, 0.0, 1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, +6 * 200 * np.pi / 4), "H is off to the right"
        assert np.allclose(v, 0.0), "V is right in the middle of the detector."
        assert not np.all(hits_it), "... and it misses."

        beam = column([1.0, 0.0, -1.0])*2*pi
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, 2 * 200 * np.pi / 4), "H is off to the left"
        assert np.allclose(v, 0.0), "V is right in the middle of the detector."
        assert not np.all(hits_it), "... and it misses."
        
        
    def test_detector_coord_vertical(self):
        """CylindricalDetector.get_detector_coordinates()"""
        det = self.det

        beam = column([0., 1., 1.]) 
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, -200 * np.pi / 4), "H is on the edge"
        assert np.allclose(v, 200.0), "V is 200 mm above horizontal"
        assert np.allclose(distance, np.sqrt(2*200.**2)), "Distance is correct"
        assert np.all(hits_it), "... and it hits it."

        beam = column([0., -1., 1.]) 
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, -200 * np.pi / 4), "H is on the edge"
        assert np.allclose(v, -200.0), "V is 200 mm below horizontal"
        assert np.allclose(distance, np.sqrt(2*200.**2)), "Distance is correct"
        assert np.all(hits_it), "... and it hits it."

        beam = column([0., 2., 1.]) 
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, -200 * np.pi / 4), "H is on the edge"
        assert np.allclose(v, 400.0), "V is 400 mm above horizontal"
        assert not np.all(hits_it), "... and it misses."
        
        beam = column([0., -2., 1.]) 
        (h, v, wl, distance, hits_it) = det.get_detector_coordinates(beam)
        assert np.allclose(h, -200 * np.pi / 4), "H is on the edge"
        assert np.allclose(v, -400.0), "V is 400 mm below horizontal"
        assert not np.all(hits_it), "... and it misses."
        
        

#==================================================================
if __name__ == "__main__":

    suite = unittest.makeSuite(TestCylindricalDetector)
    unittest.TextTestRunner().run(suite)
    
#    unittest.main()


