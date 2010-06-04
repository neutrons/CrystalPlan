"""Module to take screenshots and generate documentation,
and run scripts/GUI unittests.
General to wxPython applications.
"""

import os.path
import wx
import numpy as np

#Path to save screenshots to
base_screenshot_path = "../docs/screenshots/"

#For linux (UBUNTU) correct sizes of full frames:
title_bar_height = 24
frame_width = 8
frame_height = 4

#--------------------------------------------------------------------------------
def take_screenshot(rect):
    """ Takes a screenshot of the screen at given pos & size (rect).
    
    Parameters:
        rect: rectangle definining what to grab, in pixels.
    """
    print "Taking a screenshot at", rect
    #@type bmp Bitmap
    
    #Create a DC for the whole screen area
    dcScreen = wx.ScreenDC()

    #Create a Bitmap that will later on hold the screenshot image
    #Note that the Bitmap must have a size big enough to hold the screenshot
    #-1 means using the current default colour depth
    bmp = wx.EmptyBitmap(rect.width, rect.height)

    #Create a memory DC that will be used for actually taking the screenshot
    memDC = wx.MemoryDC()

    #Tell the memory DC to use our Bitmap
    #all drawing action on the memory DC will go to the Bitmap now
    memDC.SelectObject(bmp)

    #Blit (in this case copy) the actual screen on the memory DC
    #and thus the Bitmap
    memDC.Blit( 0, #Copy to this X coordinate
        0, #Copy to this Y coordinate
        rect.width, #Copy this width
        rect.height, #Copy this height
        dcScreen, #From where do we copy?
        rect.x, #What's the X offset in the original DC?
        rect.y  #What's the Y offset in the original DC?
        )

    #Select the Bitmap out of the memory DC by selecting a new
    #uninitialized Bitmap
    memDC.SelectObject(wx.NullBitmap)

    return bmp



#--------------------------------------------------------------------------------
def screenshot_frame(frame, filename, top_only=0):
    """Take a screenshot of an entire wx.Frame object on screen.

    Parameters
        top_only: take only the first (top_only) pixels; if 0, take everything.
    """

    rect = frame.GetScreenRect()
    rect.width += frame_width
    if top_only > 0:
        rect.height = top_only / 2
        margin = [0, 0, 0, top_only/2]
    else:
        margin = 0
    rect.height += title_bar_height + frame_height
    screenshot_of(rect, filename, margin=margin, gradient_edge=0)
#    bmp = take_screenshot(rect)
#
#    #Save to PNG file
#    bmp.SaveFile(os.path.join(base_screenshot_path, os.path.splitext(filename)[0] + ".png"), wx.BITMAP_TYPE_PNG)


#--------------------------------------------------------------------------------
def get_screen_rect(widget):
    """Get the ScreenRect of a window or sizer."""
    if isinstance(widget, wx.Window):
        return widget.GetScreenRect()
    elif isinstance(widget, wx.Sizer):
        #Find the offset of the parent window
        frame_pos = widget.GetContainingWindow().GetScreenPosition()
        #Adjust the rectangle to get the screenRect
        pos = widget.GetPosition() + frame_pos
        size = widget.GetSize()
        rect = wx.Rect(pos.x, pos.y, size.width, size.height)
        return rect
    else:
        raise NotImplementedError("cannot get a ScreenRect for the object %s" % widget)


#--------------------------------------------------------------------------------
def screenshot_of(window, filename, margin=0, gradient_edge=0, minheight=False):
    """Take a screenshot of any wx.Window object on screen.

    Parameters:
        window: wx.Window to grab. If it is a list,
            a rectangle containing all controls is grabbed.
            can also be a wx.Rect directly
        filename: PNG file to save to; path will be relative to the base_screenshot_path.
        margin: margin, in pixels, to add to all sides.
            - single scalar: applied to all sides
            - list: [left right top bottom]
        gradient_edge: define a gradient to fade out the edges of the screenshot.
            - scalar: the # will be SUBTRACTED from the margins on all sides to find the gradient size.
                this leaves this many pixels normal before fade commences
        minheight: grab a screenshot of the minimum height the control can be. Only for a single window

    """

    #@type rect Rect

    #Find the rectangle
    if isinstance(window, wx.Rect):
        rect = wx.Rect(*window) #Make a copy
    else:
        #Windows
        if hasattr(window, "__iter__"):
            for (i, wnd) in enumerate(window):
                if i == 0:
                    rect = get_screen_rect(wnd)
                else:
                    #Adding rectangles calculates the smallest rect containing all. Yay!
                    rect += get_screen_rect(wnd)
        else:
            #Single window/sizer etc.
            rect = get_screen_rect(window)
            if minheight:
                rect.Height = window.GetMinSize()[1] #Minimum height

    #Make a 4-element list for margins
    if not hasattr(margin, "__iter__"):
        margin = [margin, margin, margin, margin]

    #Adjust for margins
    if margin != 0:
        rect.Top -= margin[2]
        rect.Height += margin[2] + margin[3]
        rect.Left -= margin[0]
        rect.Width += margin[0] + margin[1]
        
    bmp = take_screenshot(rect)

    #Make the gradient pixel size list
    if gradient_edge is None:
        gradient = margin
    else:
        gradient = [x-gradient_edge for x in margin]

    #--- Make nice smoothed edges ----
    #@type img wx.Image
    img = bmp.ConvertToImage()
    img.InitAlpha()
    
    # Buffer, 1st dimension = y axis, 2nd = x axis.
    alpha_buffer = np.zeros( (rect.Height, rect.Width), dtype=int ) + 255
    min = 0
    #Left gradient
    if gradient[0]>0:
        alpha_buffer[:,0:gradient[0]] = np.linspace(min, 255, gradient[0])
    #Right gradient
    if gradient[1]>0:
        alpha_buffer[:,-gradient[1]:] = np.linspace(255, min, gradient[1])
    #Top gradient
    if gradient[2]>0:
        alpha_buffer[0:gradient[2], :] = alpha_buffer[0:gradient[2], :] - np.linspace(255, min, gradient[2]).reshape(gradient[2],1)
    #Bottom
    if gradient[3]>0:
        alpha_buffer[-gradient[3]:, :] = alpha_buffer[-gradient[3]:, :] - np.linspace(min, 255, gradient[3]).reshape(gradient[3],1)

    #Convert to byte channel, clip to 0 alpha
    alpha_buffer[alpha_buffer<0] = 0
    alpha_buffer = alpha_buffer.astype(np.byte)
    
    #Set the alpha channel
    img.SetAlphaData(alpha_buffer.data)

    #Save to PNG file
    img.SaveFile(os.path.join(base_screenshot_path, os.path.splitext(filename)[0] + ".png"), wx.BITMAP_TYPE_PNG)


#------------------------------------------------------------
def animated_screenshot(scene, filename):
    files = []
    for (i, az) in enumerate(np.arange(0, 360, 10)):
        print "azimuth", az
        (azim, elev, dist, focalpoint) = scene.mlab.view()
        scene.mlab.view(az, -45, dist, focalpoint)
        fname = "/tmp/frame"+str(i)+".png"
        files.append(fname)
        scene.mlab.savefig(fname)
        #screenshot_frame(frame, "frame" + str(i))
    #Call program to assemble them
    os.system("../doc_maker/apngasm " + filename + " " + " ".join(files) + " 1 20")
    #Erase the temp files
    for fname in files:
        os.remove(fname)

if __name__=="__main__":
    app = wx.PySimpleApp()
    frame = wx.Frame(None, title='Testing screenshot')
    
    def onClose(event):
        screenshot_of(wx.Rect(20,20,200,100), "test", margin=20, gradient_edge=5)
        event.Skip()

    frame.Bind(wx.EVT_CLOSE, onClose)
    app.frame = frame
    frame.Show()
    app.MainLoop()
    
    