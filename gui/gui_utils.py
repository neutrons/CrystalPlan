"""Collection of methods useful for GUI building using WxPython."""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
import os
import os.path
import sys
from cPickle import loads, dumps
import xml.etree.ElementTree as ET

#--- GUI Imports ---
import display_thread
import config_gui

#--- Model Imports ---
import model


# ===========================================================================================
def save_configuration():
    """Save the current configuration to a file in the home folder of the user."""
    homedir = os.getenv('HOME')
    filename = os.path.join(homedir, ".crystalplan_cfg.xml")

    root = ET.Element('html')
    #The gui configuration
    xml_config_gui = ET.SubElement(root,'config_gui')
    xml_config_gui.text = dumps(config_gui.cfg)
    #The model configuration
    xml_config = ET.SubElement(root,'config')
    xml_config.text = dumps(model.config.cfg)
    
    tree = ET.ElementTree(root)
    tree.write(filename)


# ===========================================================================================
def jiggle_window(window):
    """Jiggle the size of a window to force it to layout properly."""
    h,w = window.GetSize()
    window.SetSize((h+1, w+1))
    window.SetSize((h, w))
    window.Refresh()

#------------------------------------------------------------------
def is_mac():
    """Return True if the program is running on Mac OS"""
    return sys.platform=="darwin"

#------------------------------------------------------------------
def is_windows():
    """Return True if the program is running on Windows"""
    return sys.platform=="win32"

#------------------------------------------------------------------
def inelastic_mode():
    """Return true if the instrument being simulated is for
    inelastic scattering."""
    return isinstance(model.instrument.inst, model.instrument.InstrumentInelastic)

#------------------------------------------------------------------
def fourcircle_mode():
    """Return true if the instrument being simulated is a
    four-circle diffractometer."""
    return isinstance(model.instrument.inst, model.instrument.InstrumentFourCircle)

#------------------------------------------------------------------
def print_large_number(n,width=0,delim=',',decimal='.'):
    # Copyright 2007 Regents of the University of California
    # Written by David Isaacson at the University of California, Davis
    # BSD License
    """
    Converts a float to a string with appropriately placed commas.

    Floats will be shown with 'width' digits right of the decimal.
    'delim' specifies the thousands delimiter.
    'decimal' specifies the decimal character.
    """
    if width >= 0: s = "%.*f" %(width,n)
    else: s = str(n)
    dec = s.find(decimal)
    if dec == -1: dec = len(s)
    threes = int((dec-1)/3) #we don't need a comma at the start
    for i in xrange(threes):
        loc = dec-3*(i+1)
        s = s[:loc] + delim + s[loc:]
    return s


# ===========================================================================================
#Save the latest path used as global
last_csv_path = ''
def dialog_to_save_experiment_to_CSV(parent):
    """Opens a dialog asking the user where to save the experiment plan."""
    filters = 'CSV files (*.csv)|*.csv|All files (*)|*'
    if is_mac(): filters = '' #Filters tend to crash on mac
    global last_csv_path
    (path, filename) = os.path.split(last_csv_path)
    dialog = wx.FileDialog ( parent, defaultFile=filename, defaultDir=path, message='Save the experiment plan to CSV file', wildcard=filters, style=wx.SAVE )
    if dialog.ShowModal() == wx.ID_OK:
        filename = dialog.GetPath()
        last_csv_path = filename
        dialog.Destroy()
    else:
        #'Nothing was selected.
        dialog.Destroy()
        return
    #Save the CSV file
    model.experiment.exp.save_sample_orientations_to_CSV_file(filename)

last_experiment_path = ''
def save_experiment_file_dialog(parent):
    """Opens a dialog asking the user where to save the experiment."""
    filters = 'CrystalPlan Experiment files (*.exp)|*.exp|All files (*)|*'
    if is_mac(): filters = '' #Filters tend to crash on mac
    global last_experiment_path
    (path, filename) = os.path.split(last_experiment_path)
    dialog = wx.FileDialog ( parent, defaultFile=filename, defaultDir=path, message='Save the experiment plan to EXP file', wildcard=filters, style=wx.SAVE )
    if dialog.ShowModal() == wx.ID_OK:
        filename = dialog.GetPath()
        last_experiment_path = filename
        dialog.Destroy()
    else:
        #'Nothing was selected.
        dialog.Destroy()
        return None
    #Save the file
    model.experiment.save_to_file(model.experiment.exp, filename)
    return filename

def load_experiment_file_dialog(parent):
    """Opens a dialog asking the user where to load the experiment."""
    filters = 'CrystalPlan Experiment files (*.exp)|*.exp|All files (*)|*'
    if is_mac(): filters = '' #Filters tend to crash on mac
    global last_experiment_path
    (path, filename) = os.path.split(last_experiment_path)
    dialog = wx.FileDialog ( parent, defaultFile=filename, defaultDir=path, message='Load an experiment plan from an EXP file', wildcard=filters, style=wx.OPEN )
    if dialog.ShowModal() == wx.ID_OK:
        filename = dialog.GetPath()
        last_experiment_path = filename
        dialog.Destroy()
    else:
        #'Nothing was selected.
        dialog.Destroy()
        return None
    #Load it
    model.experiment.exp = model.experiment.load_from_file(filename)
    #Now we need to fix a lot of stuff
    model.instrument.inst = model.experiment.exp.inst
    #This hopefully redraws everything
    display_thread.handle_change_of_qspace()
    #Make sure we select it all, by default.
    display_thread.select_position_coverage(poscov_list=model.instrument.inst.positions, update_gui=True)
    return filename

last_integrate_path = ''
def load_integrate_file_dialog(parent, sequential_detector_numbers):
    """Opens a dialog asking the user where to load the integrate."""
    filters = 'ISAW .integrate or .peaks files (*.peaks;*.integrate)|*.peaks;*.integrate|All files (*)|*'
    if is_mac(): filters = '' #Filters tend to crash on mac
    global last_integrate_path
    (path, filename) = os.path.split(last_integrate_path)
    dialog = wx.FileDialog ( parent, defaultFile=filename, defaultDir=path, message='Load an ISAW .integrate or .peaks file', wildcard=filters, style=wx.OPEN )
    if dialog.ShowModal() == wx.ID_OK:
        filename = dialog.GetPath()
        last_integrate_path = filename
        dialog.Destroy()
    else:
        #'Nothing was selected.
        dialog.Destroy()
        return None
    #Now, ask if the file peaks should be appended
    res = wx.MessageDialog(parent, "Do you wish to REPLACE the peaks to the current list of measured peaks?\nClick NO to append to the list.", "Replace Measured Peaks List?", wx.YES_NO | wx.YES_DEFAULT).ShowModal()
    do_append = (res == wx.ID_NO)
    print "do_append", do_append
    #Load it
    model.experiment.exp.load_peaks_file(filename, append=do_append, sequential_detector_numbers=sequential_detector_numbers)
    #This hopefully redraws everything
    display_thread.handle_change_of_qspace()
    return filename


last_hfir_int_path = ''
def load_HFIR_int_file_dialog(parent):
    """Opens a dialog asking the user where to load the int file."""
    filters = 'HFIR .int files (*.int)|*.int|All files (*)|*|'
    if is_mac(): filters = '' #Filters tend to crash on mac
    global last_hfir_int_path
    (path, filename) = os.path.split(last_hfir_int_path)
    dialog = wx.FileDialog ( parent, defaultFile=filename, defaultDir=path, message='Load a HFIR .int file', wildcard=filters, style=wx.OPEN )
    if dialog.ShowModal() == wx.ID_OK:
        filename = dialog.GetPath()
        last_hfir_int_path = filename
        dialog.Destroy()
    else:
        #'Nothing was selected.
        dialog.Destroy()
        return None
    #Now, ask if the file peaks should be appended
    res = wx.MessageDialog(parent, "Do you wish to REPLACE the peaks to the current list of measured peaks?\nClick NO to append to the list.", "Replace Measured Peaks List?", wx.YES_NO | wx.YES_DEFAULT).ShowModal()
    do_append = (res == wx.ID_NO)
    print "do_append", do_append
    #Load it
    model.experiment.exp.load_HFIR_peaks_file(filename, append=do_append)
    #This hopefully redraws everything
    display_thread.handle_change_of_qspace()
    return filename


# ===========================================================================================
def do_recalculation_with_progress_bar(new_sample_U_matrix=None):
    """Perform a recalculation of reciprocal space coverage. Show a progress bar.

    Parameters:
        new_sample_U_matrix: set to a 3x3 matrix to change the sample's U-matrix
    """
    global count, prog_dlg
    #Steps in calculation
    max = len(model.instrument.inst.positions)*2+1

    prog_dlg = wx.ProgressDialog( "Recalculating all sample orientations.",        "Calculation progress:",
        max, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME |
                     wx.PD_ESTIMATED_TIME | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE)
    #Make it wider
    prog_dlg.SetSize((500, prog_dlg.GetSize()[1]))
    #Initial show
    count = 0
    prog_dlg.Update(count)

    keep_going = True
    for i in xrange(len(model.instrument.inst.positions)):
        try:
            #Do the recalculation
            poscov = model.instrument.inst.positions[i]
            model.instrument.inst.recalculate(poscov, new_sample_U_matrix=new_sample_U_matrix)
            #Check in the dialog
            count += 1
            (keep_going, skip) = prog_dlg.Update(count, "Recalculating orientation %s of %s..." % (i+1, len(model.instrument.inst.positions)))
        except:
            #We destroy the dialog so it doesn't get stuck.
            prog_dlg.Destroy()
            # We re-raise the exception
            # We are in the main loop, so the sys.excepthook will catch it and display a dialog.
            raise

    #Now recalc the reflections.
    def progress_callback(poscov):
        global count, prog_dlg
        count += 1
        # @type poscov PositionCoverage
        (keep_going, skip) = prog_dlg.Update(count, "Recalculating reflections at %s..." % model.instrument.inst.make_angles_string(poscov.angles))
        
    try:
        model.experiment.exp.recalculate_reflections(None, calculation_callback=progress_callback)
    except:
        #We destroy the dialog so it doesn't get stuck, and re-raise
        prog_dlg.Destroy()
        raise

    #Should we recalculate reflections here?

    #Clean up dialog
    prog_dlg.Destroy()


# ===========================================================================================
def do_calculation_with_progress_bar(poscov_list):
    """Perform a calculation of reciprocal space coverage. Show a progress bar.

    Parameters:
        poscov_list: list of PositionCoverage objects where the angles are set,
            but the coverage array is not.

    """
    #Steps in calculation
    max = len(poscov_list)+1

    prog_dlg = wx.ProgressDialog( "Calculating sample orientations.",        "Calculation progress:",
        max, style = wx.PD_CAN_ABORT | wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME |
                     wx.PD_ESTIMATED_TIME | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE)
    #Make it wider
    prog_dlg.SetSize((500, prog_dlg.GetSize()[1]))
    #Initial show
    count = 0
    prog_dlg.Update(count)

    keep_going = True
    for pos_cov_empty in poscov_list:
        if not pos_cov_empty is None:
            try:
                #Do the calculation
                poscov = model.instrument.inst.simulate_position(pos_cov_empty.angles, sample_U_matrix=pos_cov_empty.sample_U_matrix, use_multiprocessing=False)
                #Check in the dialog
                count += 1
                (keep_going, skip) = prog_dlg.Update(count, "Calculating orientation %s of %s..." % (count, len(poscov_list)))
                if not keep_going: break
            except:
                #We destroy the dialog so it doesn't get stuck.
                prog_dlg.Destroy()
                # We re-raise the exception
                # We are in the main loop, so the sys.excepthook will catch it and display a dialog.
                raise

    #Should we recalculate reflections here?

    #Clean up dialog
    prog_dlg.Destroy()




#---------------------------------------------------------------------------
#CONSTANTS FOR COLORS
TEXT_BACKGROUND_COLOUR_GOOD = "white"
TEXT_BACKGROUND_COLOUR_BAD = wx.Colour(255, 200, 200)

#---------------------------------------------------------------------------
def test_my_gui(WindowClass, *args, **kwargs):
    """General-purpose test for any wx window object. Here's how to use:
    
    (app, pnl) = test_my_gui(WindowClass, arguments)
    pnl.do_some_stuff_here()
    app.MainLoop()
    """
    app = wx.PySimpleApp()
    #Simple frame
    frame = wx.Frame(None, title='Testing a %s' % WindowClass.__name__)
    boxSizer = wx.BoxSizer(orient=wx.VERTICAL)
    frame.SetSizer(boxSizer)
    #Create the control!
    my_control = WindowClass(frame, *args, **kwargs)
    if isinstance(my_control, wx.Frame) or isinstance(my_control, wx.Dialog):
        #We are trying a frame
        my_control.Show()
        app.frame = my_control
    else:
        #Make it resize
        boxSizer.Add(my_control,1, flag=wx.EXPAND)
        frame.Show()
        app.frame = frame
    return (app, my_control)



#---------------------------------------------------------------------------
def scale_to_fit(source, target):
    """Scale a rectangle to fit within another, preserving aspect ratio.

    Parameters:
        source: wx.Size() of the rectangle being scaled.
        target: wx.Size() of the rectangle that will hold the source

    Return:
        wx.Size() containing the scaled size
        ratio: what to multiply the source to get to the resized size
    """
    ratio = max( [float(source.width) / float(target.width), float(source.height) / float(target.height)] )
    # Avoid divide by zero
    if (ratio == 0.0): ratio = 1.0
    return (wx.Size(source.width / ratio, source.height / ratio), 1/ratio)

#---------------------------------------------------------------------------
# Constants for follow_window
[   FOLLOW_SIDE_TOP,
    FOLLOW_SIDE_BOTTOM,
    FOLLOW_TOP_LEFT,
    FOLLOW_TOP_RIGHT ] = xrange(4)

class FrameFollower():
    """Class to handle following a frame with another."""
    #---------------------------------------------------------------------------
    def __init__(self, *args):
        """Set up a child window that will follow the movements of a parent window.

        Parameters:
            parent: frame that will be moved by the user.
            child: frame that will move with the parent.
            position: integer representing the position to follow at.
        """
        self.rebind(*args)

    #---------------------------------------------------------------------------
    def unbind(self):
        """Stop following the parent window."""
        self.parent.Unbind(wx.EVT_MOVE)
        self.parent.Unbind(wx.EVT_SIZE)
        self.parent.Unbind(wx.EVT_CLOSE)
        self.child.Unbind(wx.EVT_CLOSE)
        self.child.Unbind(wx.EVT_SIZE)

    #---------------------------------------------------------------------------
    def rebind(self,parent, child, position=FOLLOW_SIDE_TOP):
        """Set up a child window that will follow the movements of a parent window.

        Parameters:
            parent: frame that will be moved by the user.
            child: frame that will move with the parent.
            position: integer representing the position to follow at.
        """
        parent.Bind(wx.EVT_MOVE, self.on_move)
        parent.Bind(wx.EVT_SIZE, self.on_move)
        child.Bind(wx.EVT_SIZE, self.on_move)
        parent.Bind(wx.EVT_CLOSE, self.on_close)
        child.Bind(wx.EVT_CLOSE, self.on_close)
        #Save the values
        self.parent = parent
        self.child = child
        self.position = position
        self.margin = 8
        #Trigger a fake parent move
        self.on_move(None)

        
    #---------------------------------------------------------------------------
    def on_move(self, event):
        """Triggered when the parent moves (or resizes)"""
        #Handle each case
        if (self.position == FOLLOW_SIDE_TOP) or (self.position == FOLLOW_SIDE_BOTTOM):
            place_frame_next_to(self.parent, self.child, self.margin, follow_top=(self.position==FOLLOW_SIDE_TOP))
        else:
            raise NotImplementedError("Following top/bottom not implemented.")
        if not event is None:
            event.Skip()


    #---------------------------------------------------------------------------
    def on_close(self, event):
        """Called when the parent window is closed. Clean up the handler."""
        self.unbind()
        event.Skip()

        

#Dictionary of followers
followers = dict()

#---------------------------------------------------------------------------
def follow_window(parent, child, position=FOLLOW_SIDE_TOP):
    """Set up a child window that will follow the movements of a parent window.

    Parameters:
        parent: frame that will be moved by the user.
        child: frame that will move with the parent.
        position: integer representing the position to follow at.
    """
    if not isinstance(parent, wx.Frame) or not isinstance(child, wx.Frame):
        raise ValueError("Parent and child must be frames.")
    #Create the follower. It handles all needed stuff.
    follower = FrameFollower(parent, child, position)
    #Save the object to the dictionary
    followers[(parent, child)] = follower
    return follower

#---------------------------------------------------------------------------
def stop_following_window(parent, child):
    """Stop a child window following a parent."""
    key = (parent, child)
    #Stop following, and remove from dict
    if followers.has_key(key):
        follower = followers[key]
        follower.unbind()
        del followers[key]

    

#---------------------------------------------------------------------------
def place_frame_next_to(parent, child, margin, follow_top=True):
    """Place a new frame next to a parent frame, picking a side
    based on available space.

    Parameters:
        parent: a wx.Frame that is the parent.
        child: the new wx.Frame to place.
        margin: margin to add in pixels
        follow_top: set to True to follow the top edge, false to follow the bottom edge.
    """

    parent_pos = parent.GetScreenRect()
    screen_width = wx.DisplaySize()[0]
    (cw, ch) = child.GetSize()
    
    #Find the right y position
    if follow_top:
        cy = parent_pos.y
    else:
        cy = parent_pos.y + parent_pos.height - ch

    if (parent_pos.x + parent_pos.width + cw) <= screen_width:
        #Place it on the right
        child.Move( wx.Point(parent_pos.x + parent_pos.width + margin, cy) )
    elif (parent_pos.x - cw) >= 0:
        #Place on the left
        child.Move( wx.Point(parent_pos.x - cw - margin, cy) )
    else:
        #Put it on the right, but it'll be over the parent
        child.Move( wx.Point(screen_width - cw, cy) )


#-------------------------------------------------------
def find_parent_frame(window):
    """Find the wx.Frame that is the ultimate parent of this window."""
#    print "find_parent_frame", window.Name
    if window is None:
        return None
    prnt = window.GetParent()
    if isinstance(prnt, wx.Frame):
#        print "find_parent_frame FOUND", prnt.Name
        return prnt
    else:
        #Recurse
        return find_parent_frame(prnt)


if __name__=="__main__":
    model.instrument.inst = model.instrument.Instrument("../instruments/TOPAZ_geom_all_2011.csv")
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    save_configuration()
#    app = wx.PySimpleApp()
#    parent = wx.Frame(None, title='Parent Frame', size=wx.Size(400,600))
#    parent.Show()
#    child = wx.Frame(None, title='Child Frame')
#    child.Show()
##    child2 = wx.Frame(None, title='Child Frame #2')
##    child2.Show()
#    follow_window(parent, child, position=FOLLOW_SIDE_TOP)
##    follow_window(parent, child2, position=FOLLOW_SIDE_TOP)
##    stop_following_window(parent, child)
#    app.MainLoop()
