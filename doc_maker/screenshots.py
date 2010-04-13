"""Module to take screenshots and generate documentation,
and run scripts/GUI unittests.
General to wxPython applications.
"""
import os.path
import wx

base_screenshot_path = "/home/janik/Code/GenUtils/trunk/python/CrystalPlan/docs/screenshots/"
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
def screenshot_frame(frame, filename):
    """Take a screenshot of an entire wx.Frame object on screen."""
    rect = frame.GetScreenRect()
    rect.width += frame_width
    rect.height += title_bar_height + frame_height
    bmp = take_screenshot(rect)

    #Save to PNG file
    bmp.SaveFile(os.path.join(base_screenshot_path, os.path.splitext(filename)[0] + ".png"), wx.BITMAP_TYPE_PNG)

#--------------------------------------------------------------------------------
def screenshot_of(window, filename, margin=0):
    """Take a screenshot of any wx.Window object on screen.

    Parameters:
        window: wx.Window to grab. If it is a list,
            a rectangle containing all controls is grabbed.
            can also be a wx.Rect directly
        filename: PNG file to save to; path will be relative to the base_screenshot_path.
        margin: margin, in pixels, to add to all sides.
            - single scalar: applied to all sides
            - list: [left right top bottom]

    """
    #@type rect Rect

    #Find the rectangle
    if isinstance(window, wx.Rect):
        rect = window
    else:
        #Windows
        if hasattr(window, "__iter__"):
            for (i, wnd) in enumerate(window):
                if i == 0:
                    rect = wnd.GetScreenRect()
                else:
                    #Adding rectangles calculates the smallest rect containing all. Yay!
                    rect += wnd.GetScreenRect()
        else:
            #Single window
            rect = window.GetScreenRect()

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
    #TODO: Nice faded edges?

    #Create a memory DC of the BMP
    #@type memDC MemoryDC
    memDC = wx.MemoryDC()
    memDC.SelectObject(bmp)
    #Left gradient
    for i in xrange(margin[0]):
        alpha=(i*255.)/margin[0]
        print "alpha", alpha
        x = i
        memDC.SetPen( wx.Pen(wx.Color(255,255,255, alpha=alpha), width=1, style=wx.SOLID) )
        memDC.DrawLine(x, 0, x, rect.Height)
    #Select the Bitmap out of the memory DC
    memDC.SelectObject(wx.NullBitmap)

    #Save to PNG file
    bmp.SaveFile(os.path.join(base_screenshot_path, os.path.splitext(filename)[0] + ".png"), wx.BITMAP_TYPE_PNG)



if __name__=="__main__":
    app = wx.PySimpleApp()
    frame = wx.Frame(None, title='Testing screenshot')
    
    def onClose(event):
        screenshot_of(wx.Rect(20,20,100,100), "test", margin=20)
        event.Skip()

    frame.Bind(wx.EVT_CLOSE, onClose)
    app.frame = frame
    frame.Show()
    app.MainLoop()
    
    