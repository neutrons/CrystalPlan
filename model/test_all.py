"""Routine to automate unit tests over every module in the library."""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: __init__.py 1127 2010-04-01 19:28:43Z 8oz $

module_list = ['config', 'crystal_calc', 'crystals', 'detectors', 'experiment', 'goniometer',
    'instrument', 'messages', 'numpy_utils', 'optimize_coverage', 'reflections',
    'ubmatrixreader', 'system_tests']

import unittest
import string

def get_all_tests():
    """Returns a unittest.TestSuite containing all the tests in all the model modules."""
    #@type all_tests TestSuite
    all_tests = unittest.TestSuite()

    for module_name in module_list:
        #print "------------------ Testing Module %s ---------------------------" % module_name
        #Import it!
        module = __import__(module_name)

        test_classes = []
        for member_name in dir(module):
            if len(member_name) > 4:
                if member_name[0:4]=="Test":
                    #Starts with Test, it is a test class
                    test_classes += [member_name]

        if len(test_classes) == 0:
            print "No tests found in module %s" % (module_name)

        for test_class_name in test_classes:
            #Create an instance of it
            #print "making", test_class_name
            test_class = getattr(module, test_class_name)
            for test_method_name in dir(test_class):
                if len(test_method_name) > 4:
                    if test_method_name[0:4].lower() == "test":
                        test_instance = test_class(test_method_name)
                        #print "Adding test: %s.%s" % (test_class_name, test_method_name)
                        all_tests.addTests([test_instance])
                        #test_instance.runTest()

    return all_tests

def main():
    all_tests = get_all_tests()
    unittest.TextTestRunner(verbosity=2).run(all_tests)

if __name__=="__main__":
    main()
