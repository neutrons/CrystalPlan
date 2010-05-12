"""This module only holds the SliceControl custom control.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import numpy as np

#--- GUI Imports ---
import config_gui

#--- Model Imports ---



#-------------------------------------------------------------------------------
#CONSTANTS FOR SLICE CONTROL
MOUSE_OVER_NOTHING = 0
MOUSE_OVER_SLICE_MIN = 1
MOUSE_OVER_SLICE_MAX = 2
MOUSE_OVER_SLICE_MIDDLE = 3

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class SliceControl(wx.PyControl):
    """Custom control for displaying the slicer.
    Shows a graph of coverage, and has a movable slider bar to adjust the size of the slice.
    """

    #-------------------------------------------------------------------------------
    def __init__(self, parent, use_slice=False, apply_slice_method=None,
            id=wx.ID_ANY, pos=wx.DefaultPosition,
            size=wx.DefaultSize, validator=wx.DefaultValidator,
            name="SliceControl"):
        """
        Default class constructor.

        Parameters:
        -----------
            parent: Parent window. Must not be None.
            use_slice: Does the control start in "use slice" mode, showing the slice changer?
            apply_slice_method: method that will be called when the slice changes.
                The method should accept 3 arguments:
                    use_slice: boolean.
                    slice_min and slice_max: floats, the limits of the slice.
            id: identifier. 
            pos: position. If the position (-1, -1) is specified
                    then a default position is chosen.
            size: size. If the default size (-1, -1) is specified
                     then a default size is chosen.
            validator: Window validator.
            name: Window name.
        """
        
        #Using the slicer mode?
        self.use_slice = use_slice
        #Method to call for events
        self._apply_slice_method = apply_slice_method
        #Some defaults
        self.scale_x = 1.0
        self.scale_y = 1.0

        #Start and end positions of the slicers.
        self.slice_min = 0.5
        self.slice_max = 1.5

        #X/Y data to plot
        self.data_x = None
        self.data_y = list()

        #Update the view in real time
        self.realtime = False

        #Current drag mode
        self._current_drag = 0
        #Starting point of the drag, in plot X coordinates
        self._drag_start_x = 0
        self._drag_start_min = 0
        self._drag_start_max = 0

        #For the timer
        self._last_slice = (0, 0)

        #For energy slice type
        self.energy_mode = False


        #Init the base pyControl
        wx.PyControl.__init__(self, parent, id, pos, size, wx.TAB_TRAVERSAL, validator, name)

        # Bind the events related to our control: first of all, we use a
        # combination of wx.BufferedPaintDC and an empty handler for
        # wx.EVT_ERASE_BACKGROUND (see later) to reduce flicker
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

        # Then we want to monitor user clicks
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        if wx.Platform == '__WXMSW__':
            # MSW Sometimes does strange things...
            self.Bind(wx.EVT_LEFT_DCLICK,  self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        #As well as mouse move
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)

        #Handle resizing
        self.Bind(wx.EVT_SIZE, self.OnResize)

        #Handle keystrokae
        self.Bind(wx.EVT_CHAR, self.OnChar)


    #-------------------------------------------------------------------------------
    def ApplySlice(self):
        """Applies the change of slice parameters by setting the proper parameters.
        This calls the "_apply_slice_method" set in the constructor.
        """
        #Call it.
        if not self._apply_slice_method is None:
            if callable(self._apply_slice_method):
                self._apply_slice_method(self.use_slice, self.slice_min, self.slice_max)
            else:
                raise ValueError("SliceControl's _apply_slice_method is not callable!")


    #-------------------------------------------------------------------------------
    def SetUseSlice(self, value):
        """Is the control set to use the slice, or just show the graph?
        For example, a checkbox that is external to the control could set this
        value.

        Fires the event when changed.

        Parameters:
            value: boolean, True indicating that the slice will be used.
        """
        if self.use_slice != value:
            self.use_slice = value
            #Tell the experiment to recalculate the slice.
            self.ApplySlice()

            #Redraw - the slicer needs to go away, or be drawn again.
            self.Refresh()


    #-------------------------------------------------------------------------------
    def OnChar(self, event):
        """Handles key press event (EVT_CHAR)"""
        keycode = event.GetKeyCode()
        #Key pressed to move the slider?
        move = 0
        if (keycode == wx.WXK_LEFT):
            move = -1.0
        elif (keycode == wx.WXK_RIGHT):
            move = +1.0
        elif (keycode == wx.WXK_PAGEDOWN):
            move = +5.0
        elif (keycode == wx.WXK_PAGEUP):
            move = -5.0

        if (event.ShiftDown()):
            move = move * 5

        #print "Keyboard move by %s" % move

        if (move != 0):
            qmove = move/10.0
            if not self.data_x is None:
                #Ensure we stay within the limits
                qmax = self.data_x[-1]
                qmin = self.data_x[0]
                if qmove > (qmax-self.slice_max):
                    #We would go past the end
                    qmove = (qmax-self.slice_max)
                if qmove < (qmin-self.slice_min):
                    #We would go past the start
                    qmove = (qmin-self.slice_min)
                if qmove != 0:
                    #Shift the slice over
                    self.slice_min += qmove
                    self.slice_max += qmove
                    self.CheckSlice()
                    #Redraw
                    #self.OnPaint(None)
                    self.Refresh()
                    #Tell the experiment to recalculate the slice.
                    self.ApplySlice()

        event.Skip();

##
##    #-------------------------------------------------------------------------------
##    def OnTick(self, event):
##        """Handles the timer events."""
##        if self.realtime:
##            if self._last_slice != (self.slice_min, self.slice_max):
##                #Something changed in the slice
##                model.experiment.exp.change_slice(self.slice_min, self.slice_max)
##                #Save it so we don't update unnecessarily
##                self._last_slice = (self.slice_min, self.slice_max)


    #-------------------------------------------------------------------------------
    def OnResize(self, event):
        """Handle resizing event - redraw the control."""
        self.Refresh()

    #-------------------------------------------------------------------------------
    def WhereIsTheMouse(self, event):
        """Where is the mouse (from the event) located relative to the slicer
        controls?"""

        #If we're not in slice mode, never return that we are over it.
        if not self.use_slice:
            return MOUSE_OVER_NOTHING

        #Slice pos
        xmin = self.GetX(self.slice_min)
        xmax = self.GetX(self.slice_max)
        mouse_x = event.GetX()

        #Margin
        margin = 5
        if (xmax-xmin) <= margin+1:
            #Overlap of the min and max ones
            if (mouse_x - self.GetX(0)) < margin*3:
                #The slicer is little and on the left side
                #So we need to allow the max slicer
                return MOUSE_OVER_SLICE_MAX
            else:
                #Otherwise, the min size is the one to move
                return MOUSE_OVER_SLICE_MIN
        if abs(mouse_x-xmin) <= margin:
            #Clicking the min slicer
            return MOUSE_OVER_SLICE_MIN
        elif abs(mouse_x-xmax) <= margin:
            #Clicking the MAX slicer
            return MOUSE_OVER_SLICE_MAX
        elif (mouse_x < xmax) and (mouse_x > xmin):
            #Clicking between the side
            return MOUSE_OVER_SLICE_MIDDLE
        else:
            return MOUSE_OVER_NOTHING


    #-------------------------------------------------------------------------------
    def OnMouseMove(self, event):
        """ Handles the mouse moving event. """
        if not self.IsEnabled():
            # Nothing to do, we are disabled
            return

        if self._current_drag > 0:
            #!We are dragging something
            #How much have we moved, in plot coordinates?
            offset = self.GetPlotX( event.GetX() ) - self._drag_start_x
            #Adjust the slice positions as required
            if (self._current_drag == MOUSE_OVER_SLICE_MIN):
                self.slice_min = self._drag_start_min + offset
                if self.slice_min > self.slice_max: self.slice_min = self.slice_max
            if (self._current_drag == MOUSE_OVER_SLICE_MAX):
                self.slice_max = self._drag_start_max + offset
                if self.slice_max < self.slice_min: self.slice_max = self.slice_min
            if (self._current_drag == MOUSE_OVER_SLICE_MIDDLE):
                self.slice_min = self._drag_start_min + offset
                self.slice_max = self._drag_start_max + offset
            #Redraw the window
            self.Refresh()

            #Tell the experiment to recalculate the slice.
            if self.realtime:
                self.ApplySlice()

        else:
            #Not dragging

            #Where is the mouse, relative to the slice
            where = self.WhereIsTheMouse(event)
            if (where == MOUSE_OVER_SLICE_MIN or where == MOUSE_OVER_SLICE_MAX):
                self.SetCursor( wx.StockCursor(wx.CURSOR_SIZEWE ) )
            elif where == MOUSE_OVER_SLICE_MIDDLE:
                self.SetCursor( wx.StockCursor(wx.CURSOR_HAND))
            else:
                self.SetCursor( wx.StockCursor(wx.CURSOR_ARROW))


    #-------------------------------------------------------------------------------
    def StartDrag(self, event, where):
        """Begin a slicer dragging operation."
            event: the mouse event.
            where: return value of WhereIsTheMouse. """

        self._current_drag = where
        self._drag_start_x = self.GetPlotX(event.GetX() )
        self._drag_start_min = self.slice_min
        self._drag_start_max = self.slice_max

    #-------------------------------------------------------------------------------
    def OnMouseDown(self, event):
        """ Handles the wx.EVT_LEFT_DOWN event. """
        if not self.IsEnabled():
            # Nothing to do, we are disabled
            return

        #Where is the mouse, relative to the slice
        where = self.WhereIsTheMouse(event)
        if where > 0:
            self.StartDrag(event, where)

        event.Skip()


    #-------------------------------------------------------------------------------
    def OnMouseUp(self, event):
        """ Handles the wx.EVT_LEFT_DOWN event. """
        if not self.IsEnabled():
            # Nothing to do, we are disabled
            return
        #Stop any drag operations
        if self._current_drag > 0:
            self._current_drag = 0
            self.CheckSlice()
            self.Refresh()
            #Redo the slice calculation
            self.ApplySlice()

        event.Skip()


    #-------------------------------------------------------------------------------
    def CheckSlice(self):
        """Makes sure that the current values of slice_min and slice_max make sense."""
        if not self.data_x is None:
            if self.slice_min < self.data_x[0]: self.slice_min = self.data_x[0]
            if self.slice_max < self.data_x[0]: self.slice_max = self.data_x[0]
            if self.slice_min > self.data_x[-1]: self.slice_min = self.data_x[-1]
            if self.slice_max > self.data_x[-1]: self.slice_max = self.data_x[-1]


    #-------------------------------------------------------------------------------
    def OnPaint(self, event):
        """ Handles the wx.EVT_PAINT event. """

        # If you want to reduce flicker, a good starting point is to
        # use wx.BufferedPaintDC.
        dc = wx.BufferedPaintDC(self)

        # It is advisable that you don't overcrowd the OnPaint event
        # (or any other event) with a lot of code, so let's do the
        # actual drawing in the Draw() method, passing the newly
        # initialized wx.BufferedPaintDC
        self.Draw(dc)

    #-------------------------------------------------------------------------------
    def GetX(self, x):
        """Returns the x pixel position for the given plot x position."""
        return (x-self.data_x[0])*self.scale_x + self.plot_x_offset

    def GetY(self, y):
        """Returns the y pixel position for the given plot y position."""
        return y*self.scale_y + self.plot_y_offset

    def GetPlotX(self, x_pixel):
        """Returns the x plot coordinate given the pixel position.
        Limits to 0 at minimum."""
        if self.scale_x == 0: return 0
        x = (x_pixel - self.plot_x_offset) / self.scale_x
        if x < 0:
            return 0
        else:
            return x

    #-------------------------------------------------------------------------------
    def DrawTick(self, dc, x, y, label, horizontal=True):
        """Draw a tick and label onto the plot.
            x,y: in plot coordinates"""
        tick = 3
        px = self.GetX(x)
        py = self.GetY(y)
        dc.SetPen( wx.Pen('black', width=1) )
        if horizontal:
            #Go along the horizontal (x) axis
            dc.DrawLine( px, py, px, py+tick+1)
            dc.DrawLabel(label,
                wx.Rect( px, py+tick+1),
                alignment=wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_TOP )
        else:
            dc.DrawLine( px, py, px-tick-1, py)
            dc.DrawLabel(label,
                wx.Rect( self.GetX(x)-tick-1, self.GetY(y)),
                alignment=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL )


    #-------------------------------------------------------------------------------
    def Draw(self, dc):
        """
        Actually performs the drawing operations, for the bitmap and
        for the text, positioning them centered vertically.
        """

        # Get the actual client size of ourselves
        width, height = self.GetClientSize()
        self.plot_x_offset = 28
        self.plot_width = width- self.plot_x_offset
        self.plot_height = height-25
        self.plot_y_offset = height-20

        if not width or not height:
            # Nothing to do, we still don't have dimensions!
            return

        # Initialize the wx.BufferedPaintDC, assigning a background
        # colour and a foreground colour (to draw the text)
        backColour = self.GetBackgroundColour()
        backColour = 'white'
        backBrush = wx.Brush(backColour, wx.SOLID)
        dc.SetBackground(backBrush)
        dc.Clear()

        if self.IsEnabled():
            dc.SetTextForeground(self.GetForegroundColour())
        else:
            dc.SetTextForeground(wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT))

        dc.SetFont(self.GetFont())

        #Now draw the data
        x = self.data_x
        xrange = 1.
        yrange = 1.
        #Pen and brush colors
        pen_colors = ['black', 'dark green', 'orange', 'dark orange']
        brush_colors = ['cyan', 'light green', 'yellow', [235, 166, 5] ]

        if not (x is None):
            #Lets figure out how much to scale in x and y
            xrange = x[-1] - x[0]
            self.scale_x = self.plot_width * 1. / xrange

            plot_num = 0
            for y in self.data_y:
                #Go through each plot

                if plot_num == 0:
                    #Now scale in y
                    yrange = np.max(y) #y will be from 0 to max(y)
                    if yrange < 0.1: yrange = 100
                    self.scale_y = -self.plot_height / yrange

                polygon = list()
                #Starting point
                polygon.append( wx.Point( self.GetX(0), self.GetY(0)) )
                #And go up to the first data point to make the graph look better.
                polygon.append( wx.Point( self.GetX(0), self.GetY(y[0])) )
                #Each data point
                for i in range(len(x)):
                    polygon.append( wx.Point( self.GetX(x[i]), self.GetY(y[i]) ) )
                #Point at the end
                polygon.append( wx.Point( self.GetX(x[-1]), self.GetY(0)) )
                #And go back to the starting point
                polygon.append( wx.Point( self.GetX(0), self.GetY(0) ) )

                #Make the polygon pen (for outline)
                dc.SetPen( pen=wx.Pen(colour=pen_colors[plot_num], width=2, style=wx.SOLID) )
                dc.SetBrush( brush=wx.Brush(colour=brush_colors[plot_num], style=wx.SOLID) )
                dc.DrawPolygon(points=polygon, xoffset=0, yoffset=0)

                plot_num += 1

        #Now make the Y axis line
        dc.SetPen( pen=wx.Pen(colour='black', width=1, style=wx.SOLID) )
        dc.DrawLine( self.plot_x_offset, self.plot_y_offset, self.plot_x_offset, -1)


        #Add some labels
        d_mode = config_gui.cfg.show_d_spacing

        #--- Vertical Axis ----
        self.DrawTick(dc, self.data_x[0], 0, '0', horizontal=False)
        self.DrawTick(dc, self.data_x[0], yrange, ("%d" % np.round(yrange)), horizontal=False)
        dc.DrawLabel('%' , wx.Rect( self.GetX(self.data_x[0])-2, self.GetY(yrange/2)), alignment=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL )

        steps = 1
        steps = int( 1.0+ xrange * 1. / (self.plot_width/40) ) #Around 40 pixel between ticks
        if steps > 20: steps = 50
        elif steps > 10: steps = 20
        elif steps > 5: steps = 10
        elif steps > 2: steps = 5
        if steps < 1: steps = 1

        if self.energy_mode:
            # --- Label the energy axis ---
            dc.DrawLabel('E' , wx.Rect( 2, height), alignment=wx.ALIGN_LEFT|wx.ALIGN_BOTTOM )
            #Show energy ticks
            for x_label in range(int(self.data_x[0]), int(self.data_x[-1]+1), steps):
                self.DrawTick(dc, x_label, 0, ("%d" % x_label), horizontal=True)

        else:
            # --- Q-space slice ---
            if d_mode:
                #Show d-spacing
                for x_label in range(0, int(xrange+1), steps):
                    if x_label > 0:
                        label = ("%.1f" % (2*np.pi/(1.0*x_label)))
                    else:
                        label = u"\u221E" #infinity
                    self.DrawTick(dc, x_label, 0, label, horizontal=True)
                dc.DrawLabel('d' , wx.Rect( 2, height), alignment=wx.ALIGN_LEFT|wx.ALIGN_BOTTOM )
            else:
                #Show q-values
                for x_label in range(0, int(xrange+1), steps):
                    self.DrawTick(dc, x_label, 0, ("%d" % x_label), horizontal=True)
                #Label
                dc.DrawLabel('q' , wx.Rect( 2, height), alignment=wx.ALIGN_LEFT|wx.ALIGN_BOTTOM )


##        dc.DrawText( ("%d" % np.round(yrange)) , self.GetX(0), self.GetY(yrange))
##        dc.DrawRotatedText( '%', self.GetX(0),  self.GetY(yrange/2), angle=90 )
        if self.use_slice:
            #Draw the slicer thingie
            xmin = self.GetX(self.slice_min)
            xmax = self.GetX(self.slice_max)
            dc.SetPen( pen=wx.Pen(colour='dark red', width=2, style=wx.SOLID) )
            dc.SetBrush( brush=wx.Brush(colour=wx.Colour( 255,0,0), style=wx.CROSSDIAG_HATCH) )
            sl_width = (xmax-xmin)
            if sl_width < 4:
                sl_width = 4
                xmin = (xmax+xmin)/2 - 2
            dc.DrawRectangle(x=xmin, y=-10, width=sl_width, height=height+20)



    #-------------------------------------------------------------------------------
    def OnEraseBackground(self, event):
        """ Handles the wx.EVT_ERASE_BACKGROUND event for CustomCheckBox. """

        # This is intentionally empty, because we are using the combination
        # of wx.BufferedPaintDC + an empty OnEraseBackground event to
        # reduce flicker
        pass


    #-------------------------------------------------------------------------------
    def SetData(self, data_x, data_y):
        """ Sets the data to plot in the coverage. Does not redraw.
            data_x: numpy array of x values for the points.
            data_y: list of numpy arrays with the y values. data_y[0] is measured once or more, data_y[1] is measured twice, etc."""
        self.data_x = data_x
        self.data_y = data_y
        #Ensure the slice is still valid.
        self.CheckSlice()




if __name__ == '__main__':
    import gui_utils
    (app, sc) = gui_utils.test_my_gui(SliceControl)
    sc.use_slice = True
    sc.energy_mode = True
    data_x = np.arange(-50, 20, 5)
    print data_x
    data_y = []
    for i in xrange(4):
        data_y.append( data_x*0 )
    sc.SetData(data_x, data_y)
    sc.Refresh()
    
    app.frame.SetClientSize(wx.Size(700,500))
    app.MainLoop()


