"""Version information for CrystalPlan"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$


#import os
#import re
#
#def get_svn_revision():
#    """Get revision # from svn command line on repository"""
#    revision = 0
#    fileRev = os.popen("svn info .", "r")
#    for line in fileRev:
#        print line
#        if 'Revision' in line:
#            revision = re.match("Revision: (\d+)", line).group(1)
#    return revision

version = '1.6'
package_name = 'CrystalPlan'
description = \
"""CrystalPlan is an experiment planning tool for crystallography. You can choose an instrument and supply your sample's lattice parameters to simulate which reflections will be measured, by which detectors and at what wavelengths."""

license = """Enter license text here."""
author = "Janik Zikovsky"
author_email = "zikovskyjl@ornl.gov"
url = 'http://neutronsr.us'
copyright = '(C) 2010'

#Path to icons, relative to gui scripts path
icon_file = "icons/CrystalPlan_icon.png"
icon_file_config = "icons/CrystalPlan_icon_config.png"
icon_file_3d = "icons/CrystalPlan_icon_3d.png"
icon_file_optimizer = "icons/CrystalPlan_icon_optimizer.png"

if __name__ == "__main__":
    print package_name, version
