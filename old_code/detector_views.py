"""Detector views module: holds the View classes that display GUI elements related
to the detectors."""

#$Id$

import wx
#from pubsub import pub


import model

class DetectorView3D:
    """Class to display all the detectors in a 3D view."""
    
    def display(self):
        """Plot all the detectors in real space."""
        mlab.figure("Detectors", size=(600, 500))
        mlab.clf
        mlab.options.offscreen = False
        #Make a simple color map
        c = 0
        for det in model.instrument.inst.detectors:
            c = c+1
            n=3.0
            r = (c%n)/n
            g = ((c/n)%n)/n
            b = ((c/(n*n))%n)/n
            col = (r, g, b)
            print "plotting " + det.name
            det.plot(color=col)

        mlab.orientation_axes()
        mlab.title("Detectors in real space", size=0.3, height=0.98 )
        mlab.show()

        
class DetectorListController:
    """Displays the detectors as a checked list."""
    lst = None
    
    def __init__(self, checkListBox):
        #UI list box
        self.lst = checkListBox
##        model.messages.subscribe(
        self.update()
        
    def update(self):
        #Make the list
        items = list()
        for det in model.instrument.inst.detectors:
            items.append(det.name)
        #Sets the list
        self.lst.Set(items)
        #Check the boxes as appropriate
        for x in range(len(model.instrument.inst.detectors_used)):
            self.lst.Check(x, model.instrument.inst.detectors_used[x])

