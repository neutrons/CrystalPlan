"""Module to hold the DetectorPlot widget, which is a plot of a detector
and a spot (peak) on it."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import  wx.lib.newevent
import numpy as np

#--- GUI Imports ---
import gui_utils

import model


#Custom event
DetectorClicked, EVT_DETECTOR_CLICKED = wx.lib.newevent.NewEvent()
DetectorClickMoved, EVT_DETECTOR_CLICK_MOVED = wx.lib.newevent.NewEvent()

class DetectorPlot(wx.Window):
    """Window plots a detector and the positions of reflections on it.
    """

    
    #---------------------------------------------------------------------------
    def __init__(self, parent=None, **kwargs):
        """Constructor.

        Parameters:
            center_horizontal, center_vertical: bool, to set the centering of the plot.
            show_coordinates: show the coordinates of the mouse
        """
        wx.Window.__init__(self, parent=parent)
        self.SetBackgroundColour("white")
        #Bind all the events
        self.Bind(wx.EVT_PAINT, self.detectorPlotPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnPlotEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnPlotResize)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)
        #Other members
        self.meas = None
        self.detector = None
        self.scaling=1.
        (self.xoffset, self.yoffset) = (0, 0)
        self.mouse_pos = (np.nan, np.nan)
        (self.det_width, self.det_height) = (150., 150.)
        self.background_image = None

        #Settings
        self.center_horizontal = kwargs.get("center_horizontal", False)
        self.center_vertical = kwargs.get("center_vertical", False)
        self.align_right = kwargs.get("align_right", False)
        self.show_coordinates = kwargs.get("show_coordinates", False)

    #---------------------------------------------------------------------------
    def set_measurement(self, meas):
        """Set a measurement to be plotted"""
        self.meas = meas
        if (meas.detector_num >= 0) and (meas.detector_num < len(model.instrument.inst.detectors)):
            self.set_detector(model.instrument.inst.detectors[meas.detector_num])
        self.Refresh()

    #---------------------------------------------------------------------------
    def set_detector(self, detector):
        """Set the detector to be plotted."""
        self.detector = detector
        self.Refresh()

    #---------------------------------------------------------------------------
    def detectorPlotPaint(self, event):
        """Called when the detectorPlot needs to be redrawn."""
        #Get the real detector size
        if self.detector is None:
            (self.det_width, self.det_height) = (150., 150.)
        else:
            (self.det_width, self.det_height) = (self.detector.width, self.detector.height)

        #Figure out the scaled size that maintains aspect ratio
        (self.plot_size, self.scaling) = gui_utils.scale_to_fit(wx.Size(self.det_width, self.det_height), self.GetSize())

        #@type dc, BufferedDC
        dc = wx.BufferedPaintDC(self)

        #Clear the background
        backColour = self.GetBackgroundColour()
        backBrush = wx.Brush(backColour, wx.SOLID)
        dc.SetBackground(backBrush)
        dc.Clear()
        #Draw a rectangle
        dc.SetPen( pen=wx.Pen(colour="black", width=1, style=wx.SOLID) )
        dc.SetBrush( brush=wx.Brush(colour="white", style=wx.SOLID) ) #wx.Colour(200,200,255)
        # To center vertically
        self.yoffset = 0
        if self.center_vertical:
            self.yoffset = (self.GetSize()[1] - self.plot_size.height)  / 2
        #Horizontally
        self.xoffset = 0
        if self.center_horizontal:
            self.xoffset = (self.GetSize()[0] - self.plot_size.width)  / 2
        elif self.align_right:
            self.xoffset = (self.GetSize()[0] - self.plot_size.width)


        dc.DrawRectangle(self.xoffset, self.yoffset, self.plot_size.width, self.plot_size.height)

        if not self.background_image is None:
            #First, scale the image into
            scaled_image = self.background_image.Scale(self.plot_size.width, self.plot_size.height)
            #Make into a device-dependant bitmap
            bmp = wx.BitmapFromImage(scaled_image)
            #Draw it!
            dc.DrawBitmap(bmp, self.xoffset, self.yoffset)

        if not self.meas is None:
            self.plot_measurement(dc)

        if self.show_coordinates:
            dc.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))
            if np.any(np.isnan(self.mouse_pos)):
                dc.DrawText("N/A", 2, 2)
            else:
                dc.DrawText("%.2f, %.2f" % self.mouse_pos, 2, 2)


    #---------------------------------------------------------------------------
    def plot_measurement(self, dc):
        """Plot a reflection measurement."""
        #Plot where the peak was found
        x = self.xoffset + self.scaling * (self.det_width/2 + self.meas.horizontal)
        #We flip the y measurement value because we want negative y to be at the bottom, on screen.
        y = self.yoffset + self.scaling * (self.det_height/2 - self.meas.vertical)
        #Make cross-hairs, size relative to size of plot
        length = self.plot_size.width/10
        #Get the radius in pixels from the peak width in mm
        r = self.plot_size.width * self.meas.peak_width / self.det_width
        if r <= 0: r = 1

        def crosshairs():
            #Make crosshairs
#            dc.SetPen( pen=wx.Pen(colour="blue", width=3, style=wx.SOLID) )
#            dc.DrawLine(x+length,y, x-length,y)
#            dc.DrawLine(x,y+length, x,y-length)
            dc.SetPen( pen=wx.Pen(colour="red", width=1, style=wx.SOLID) )
            dc.DrawLine(x+length,y, x-length,y)
            dc.DrawLine(x,y+length, x,y-length)

        def spot_size():
            #Plot a circle of the right size over
            dc.SetPen( pen=wx.Pen(colour="black", width=1, style=wx.SOLID) )
            dc.SetBrush( brush=wx.Brush(colour="black", style=wx.SOLID) )
            dc.DrawCircle(x,y,r)

        #Swap the draw order depending on size.
        if length/r > 1.15:
            crosshairs()
            spot_size()
        else:
            spot_size()
            crosshairs()

    #---------------------------------------------------------------------------
    def OnPlotEraseBackground(self, event):
        pass

    def OnPlotResize(self, event):
        """Handle resizing event - redraw the control."""
        self.Refresh()

    #---------------------------------------------------------------------------
    def get_detector_coords(self, x, y):
        """Return the (mx, my), the detector coordinates of the mouse pointer."""
        w = self.det_width/2
        h = self.det_height/2

        mx = (x - self.xoffset)/self.scaling - w
        my = -(y - self.yoffset)/self.scaling + h
        if mx < -w or mx > w:
            mx = np.nan
        if my < -h or my > h:
            my = np.nan
        return (mx, my)

    #---------------------------------------------------------------------------
    def OnMouseMove(self, event):
        self.mouse_pos = self.get_detector_coords(event.X, event.Y)
        #Send a continuous event?
        if event.LeftIsDown() and not np.any(np.isnan(self.mouse_pos)):
            clickEvent = DetectorClickMoved(data=self.mouse_pos)
            self.GetEventHandler().ProcessEvent(clickEvent)
            
        if self.show_coordinates:
            self.Refresh()

    #---------------------------------------------------------------------------
    def OnLeftUp(self, event):
        """Called when clicking on the detector face."""
        self.mouse_pos = self.get_detector_coords(event.X, event.Y)
        if not np.any(np.isnan(self.mouse_pos)):
            clickEvent = DetectorClicked(data=self.mouse_pos)
            self.GetEventHandler().ProcessEvent(clickEvent)



#===========================================================================
if __name__ == "__main__":
    import gui_utils
    (app, pnl) = gui_utils.test_my_gui(DetectorPlot)
    app.frame.SetClientSize(wx.Size(500, 700))
    app.MainLoop()
