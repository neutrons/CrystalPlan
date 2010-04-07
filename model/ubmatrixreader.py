""" ubmatrixreader: Program to read in a UB Matrix text file.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import os.path
import sys
import numpy as np
from math import *
import optparse

#--- Model Imports ---
import crystal_calc


#-----------------------------------------------------------------
def read_ubmatrix_file(filename, verbose):
    """Open and read a UB matrix file.

    Parameters:
        filename: string, path to the file to load
        verbose: True for console output

    Returns:
        lattice_lengths: tuple of 3 lattice dimensions in angstroms.
        lattice_angles: tuple of 3 lattice angles in degrees.
        ub_matrix: the UB matrix read from the file
        niggli: calculated niggli matrix
    """

    if not (os.path.exists(filename)):
        raise IOError, ("The file %s cannot be found" % filename)
        return None
    
    # Read the file.
    #try:
    if True:
        if verbose: print 'Opening file', filename
        f = open(filename)

        #Read the transposed UB matrix. 3x3, first 3 lines
        UBtransposed = np.zeros([0,3])

        for i in range(0, 3):
            s = f.readline()
            temp = np.fromstring(s, float, 3, ' ')
            UBtransposed = np.vstack((UBtransposed, temp))

        if verbose: print "Transposed UB matrix is:\n", UBtransposed

        #Fourth line: is the unit cell which describes the smallest repeatable unit that can build up
        #the sample in three dimensions:  the first three numbers are a, b, c in Angstroms (10^-8cm)
        # followed by the three angles (in degrees) followed by the volume in Angstroms cubed.
        s = f.readline()
        temp = np.fromstring(s, float, 7, ' ')
        if verbose: print "Lattice parameters in file are (a,b,c, alpha,beta,gamma, volume):\n", temp

        #Unit cell sizes in angstroms
        a = temp[0]
        b = temp[1]
        c = temp[2]
        #Angles
        alpha = temp[3]
        beta = temp[4]
        gamma = temp[5]

        #Calculate the niggli matrix
#        niggli = crystal_calc.make_niggli_matrix(a,b,c,alpha,beta,gamma)

        #Return params
        lattice_lengths = (a,b,c)
        lattice_angles = (alpha,beta,gamma)
        ub_matrix = UBtransposed.transpose()

        return (lattice_lengths, lattice_angles, ub_matrix)
    
    #Error checking here
    #except:
        print "Error reading UB matrix file:", sys.exc_info()[0]
    #finally:
        #Clean up in case of error.
        f.close()

    #There was an exception if we reached this point
    return None


#============================ MAIN CODE ==========================
if __name__ == "__main__":
    #Parse the command line arguments
    parser = optparse.OptionParser(usage="Usage: %prog [options] filename")
    parser.add_option("-v", "--verbose", action="store_true", default=False, help="verbose output.")
    (options, args) = parser.parse_args()

   #We need a single argument, the filename
    if len(args) != 1:
        parser.error("Please specify the filename!")
        parser.print_help()

    #Arguments were sufficient.
    filename = args[0]
    (lattice_lengths, lattice_angles, ub_matrix) = read_ubmatrix_file(filename, options.verbose)

    #Perhaps format ouptput differently.
    print "Resulting ub_matrix Matrix is:\n", ub_matrix

   

