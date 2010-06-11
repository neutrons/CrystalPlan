""" General utils module.

Provides useful functions.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: numpy_utils.py 1325 2010-06-09 20:39:26Z 8oz $


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