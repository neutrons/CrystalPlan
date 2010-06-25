""" General utils module.

Provides useful functions.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: numpy_utils.py 1325 2010-06-09 20:39:26Z 8oz $

import numpy as np

#------------------------------------------------------------------------------
def getstate_except(obj, exclude_list):
    """Return a dictionary with all the attributes in the object EXCEPT:
        - those in the exclude_list
        - any that start with "_" (normally private ones)
        - any that can't pickle, i.e. callable objects.

    This is used when pickling, from the __getstate__() method.
    """
    d = {}

    #Make the dictionary
    for key in dir(obj):
        if not key.startswith("_"):
            if not key in exclude_list:
                value = getattr(obj, key)
                #No callable (don't pickle methods"
                if not hasattr(value, '__call__'):
                    d[key] = value
                    #print key
    return d

#------------------------------------------------------------------------------
def equal_values(value1, value2):
    """Compare values, handle numpy arrays, and lists."""
    if (value1 is None) and (value2 is None):
        return True
    if (value1 is None) or (value2 is None):
        return False
    
    if isinstance(value1, np.ndarray):
        return np.allclose(value1, value2)
    elif hasattr(value1, "__iter__"):
        #Iterable:
        if len(value1) != len(value2):
            return False
        else:
            for (val1, val2) in zip(value1, value2):
                if not equal_values(val1, val2):
                    return False
            return True
    else:
        #Default!
        return (value1 == value2)

#------------------------------------------------------------------------------
def equal_objects(first, second):
    """Compare all the attributes of the first and second object, and
    return true if all were equal. Except:
        - any that start with "_" (normally private ones)
        - any that can't pickle, i.e. callable objects.
    """
#    print "equal_objects called for %s==%s" % (first, second)
    #Make the dictionary
    for key in dir(first):
        if not key.startswith("_"):
            value = getattr(first, key)
            #No callable (don't pickle methods"
            if not hasattr(value, '__call__'):
#                print "Key %s" % key
                if hasattr(second, key):
                    value2 = getattr(second, key)
                    if not equal_values(value, value2):
                        #No match!
                        print "... no match at %s. \n %s \n -------- vs --------- \n %s" % (key, value, value2)
                        return False
    #If you get to this point, all were equal, return true.
    return True