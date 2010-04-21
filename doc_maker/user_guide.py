"""Module to generate screenshots for the
CrystalPlan user guide.
"""

import wx
import sys
import os
import warnings
from screenshots import *

import time
from time import sleep
from threading import Thread

import model
import dialog_edit_crystal

main_frame = None

#-------------------------------------------------------------------------------
def wait(ms):
    """Wait a given # of milliseconds."""
    print "Waiting for", ms, "ms"
    time.sleep(ms/1000.)

#-------------------------------------------------------------------------------
def ca(function, *args, **kwargs):
    """Alias for wx.CallAfter"""
    wx.CallAfter(function, *args, **kwargs)

#-------------------------------------------------------------------------------
def call_event(control, event_type):
    """Force an event to be called on the GUI."""
    evt = wx.CommandEvent(event_type.typeId, control.GetId())
    evt.SetEventObject(control)
    wx.PostEvent(control, evt)

#-------------------------------------------------------------------------------
def click(widget):
    """Simulate a click on a button. Sends the proper event to it."""
    if isinstance(widget, wx.Button):
        call_event(widget, wx.EVT_BUTTON)
    else:
        raise NotImplementedError("Can't simulate click on " + widget.Name)

#-------------------------------------------------------------------------------
def select_name(widget, entry):
    """Select an entry by name in a list-type widget (Choice, ListBox, etc.)."""

    def send_event():
        """To send the selection event to each type of widget"""
        if isinstance(widget, wx.Choice):
            call_event(widget, wx.EVT_CHOICE)
        else:
            raise NotImplementedError("Can't send_event on " + widget.Name)

    if isinstance(widget, wx.ItemContainer):
        for i in xrange(widget.GetCount()):
            if widget.GetString(i) == entry:
                widget.SetSelection(i)
                send_event()
                return
        raise ValueError("The entry '%s' was not found in %s, and could not be selected." % (entry, widget.Name))
    else:
        raise NotImplementedError("Can't select_name on " + widget.Name)



#========================================================================================================
#========================================================================================================
class UserGuideThread(Thread):
    """Thread to keep updating the displayed q-space coverage as fast as possible
    while keeping the UI responsive."""

    _want_abort = False

    def __init__(self, code, fm, fv):
        """ctor

        Parameters:
            code: list of code statements to execute
            fm: the FrameMain object.
            fv: the FrameQspaceView object
        """
        Thread.__init__(self)
        self.code = code
        self.fm = fm
        self.fv = fv
        self.start()

    def abort(self):
        self._want_abort = True

    def run(self):
        print "-> User guide generation thread starting. Please don't touch anything!"
        fm = self.fm
        fv = self.fv

        for line in self.code:
            print "-> SCRIPT: " + line
            #Don't process comment lines
            if len(line) > 0 and line[0] != "#":
                #Execute that line of code
                exec(line)
                #Wait a default of 10 ms between lines
                time.sleep(0.01)

        #Ok we are done.
        print "-> Script Complete!"
        #Do the latex conversion
        import eqhtml
        #eqhtml.embed_latex_in_html("../docs/user_guide.html", "../docs/user_guide_eq.html")
        #Close the main frame to exit the program
        fm.Destroy()



#==================================================================================================
#==================================================================================================
#==================================================================================================
#==================================================================================================
#==================================================================================================
#==================================================================================================
#==================================================================================================

def make_animated_tab_click(fm):
#    #Make a lot of screenshots to animate yay!
#
    rect = fm.notebook.GetScreenRect()
    rect.Height = 45 #This will depend on platform!
    for i in xrange(fm.notebook.GetPageCount()):
        ca(fm.notebook.SetSelection, i)
        wait(50)
        ca(screenshot_of, rect, 'frame_main-tab'+str(i), margin=[10, 10, 20, 50], gradient_edge=5)

    files = ['../docs/screenshots/frame_main-tab'+str(i)+".png" for i in xrange(fm.notebook.GetPageCount())]
    #Assemble into animated png
    os.system("../doc_maker/apngasm ../docs/screenshots/frame_main-tab_anim.png " + " ".join(files) + " 5 10")
    for fname in files:
        os.remove(fname)


#The following function will be executed line-by-line by a separate thread.
# - All commands should be on single lines
# - No loops or changes of indentation! (Use functions if you need to make a for loop)
# - Finish with "#---END---\n"
def user_guide_script():
    #Shortcuts to the tested objects
    exp = model.experiment.exp
    inst = model.experiment.exp.inst

    ca(screenshot_frame, fm, 'frame_main')
    wait(50)
#    make_animated_tab_click(fm)
    #warnings.warn("Hey! Turn the animated tab maker back on!")
    

#    # --------------------- Q-Space Tab --------------------
#    ca(fm.notebook.SetSelection, 0)
#    wait(100)
#    #Settings for the guide
#    #@type params StartupParameters
#    params = fm.tab_startup.params
#    params.d_min = 1.0
#    params.q_resolution = 0.1
#    params.wl_min = 0.5
#    params.wl_max = 4.0
#    wait(100)
#    ca(screenshot_of, fm.tab_startup.control, 'startup-traits', minheight=True, margin=10, gradient_edge=0)
#    ca(screenshot_of, fm.tab_startup.buttonApply, 'startup-apply', margin=5)
#
#    # ------------------------ Detectors tab ----------------------
#    ca(fm.notebook.SetSelection, 1)
#    wait(50)
#    #@type td PanelDetectors
#    td = fm.tab_detectors
#    ca(screenshot_of, td.buttonLoadDetectors, 'detectors-buttonLoadDetectors', minheight=True, margin=6, gradient_edge=4)
#    ca(td.controller.load_detector_file, "../instruments/TOPAZ_detectors_all.csv")
#    wait(1800)
#    assert len(inst.detectors) == 48, "loaded 48 detectors from TOPAZ. We have %d" % len(inst.detectors)
#    ca(screenshot_of, td.button_view_detectors, 'detectors-button_view_detectors', minheight=True, margin=6, gradient_edge=4)
#
#    #3d shot of detectors
#    ca(click, td.button_view_detectors)
#    wait(2000)
#    ca(screenshot_frame, td.frame3d, 'detectors-3d_view')
#
#
#    # ------------------------ goniometer tab ----------------------
#    ca(fm.notebook.SetSelection, 2)
#    wait(50)
#    #@type tg PanelGoniometer
#    tg = fm.tab_goniometer
#    ca(screenshot_of, tg.boxSizerSelected, 'goniometer-selected', minheight=True, margin=[10, 10, 40, 10], gradient_edge=4)
#    wait(50)
#    ca(screenshot_of, tg.buttonSwitchGoniometer, 'goniometer-buttonSwitchGoniometer', margin=6, gradient_edge=0)
#    wait(50)
#    #Select the TopazInHouseGoniometer and switch to it
#    ca(select_name, tg.choiceGonio, model.goniometer.TopazInHouseGoniometer().name)
#    wait(50)
#    ca(screenshot_of, tg.choiceGonio, 'goniometer-choice', margin=6, gradient_edge=0)
#    wait(50)
#    ca(screenshot_of, [tg.staticTextDesc, tg.staticTextDescLabel], 'goniometer-desc', margin=6, gradient_edge=0)
#    ca(click, tg.buttonSwitchGoniometer)
#    wait(100)
#    assert isinstance(inst.goniometer, model.goniometer.TopazInHouseGoniometer), "we picked a TopazInHouseGoniometer"

    # ------------------------- Sample tab -----------------------
    ca(fm.notebook.SetSelection, 3)
    wait(50)
    #@type ts PanelSample
    ts = fm.tab_sample
    ca(screenshot_of, ts.crystal_control, 'sample-info', margin=10, gradient_edge=0)
    ca(screenshot_of, ts.buttonEditCrystal, 'sample-buttonEditCrystal', margin=5)
    ca(click, ts.buttonEditCrystal)
    wait(500)
    #@type dlg DialogEditCrystal
    dlg = dialog_edit_crystal.last_dialog
    dlg.crystal.name = "Quartz Crystal"
    dlg.crystal.description = "Tutorial sample of quartz."
    wait(50)
    ca(screenshot_frame, dlg, 'dialog_edit_crystal', top_only=80)
    wait(50)
    ca(dlg.notebook.SetSelection, 1)
    wait(50)
    ca(screenshot_of, dlg.notebook, 'dialog_edit_crystal-notebook1', margin=10)
    ca(screenshot_of, dlg.buttonGenerateUB, 'dialog_edit_crystal-buttonGenerateUB', margin=6)
    wait(50)
    ca(dlg.notebook.SetSelection, 0)
    wait(50)
    ca(screenshot_of, dlg.boxSizerIsaw, 'dialog_edit_crystal-notebook0', margin=[16, 16, 46, 16], gradient_edge=6, minheight=True)
    ca(screenshot_of, dlg.buttonReadUB, 'dialog_edit_crystal-buttonReadUB', margin=6)
    ca(screenshot_of, dlg.buttonOK, 'dialog_edit_crystal-buttonOK', margin=6)
    ca(screenshot_of, dlg.buttonCancel, 'dialog_edit_crystal-buttonCancel', margin=6)
    wait(50)
    (phi, chi, omega) = (30, 15, 60)
    dlg.ub_orientation.phi_degrees = phi
    dlg.ub_orientation.chi_degrees = chi
    dlg.ub_orientation.omega_degrees = omega
    wait(50)
    ca(screenshot_of, dlg.control_load_angles, 'dialog_edit_crystal-control_load_angles', margin=6)
    wait(50)
    ca(dlg.crystal.read_ISAW_ubmatrix_file, "../model/data/quartzub.txt", [phi, chi, omega])
    wait(200)
    ca(screenshot_of, dlg.control_top, 'dialog_edit_crystal-control_top', margin=6)
    wait(50)
    ca(click, dlg.buttonOK)
    wait(500)

    ca(screenshot_of, ts.range_control, 'sample-range_control', margin=10, gradient_edge=0)
    wait(50)
    ca(screenshot_of, ts.buttonApplyRange, 'sample-buttonApplyRange', margin=10, gradient_edge=0)
#---END---

    # ------------------------- Trial Positions tab -----------------------
    ca(fm.notebook.SetSelection, 4)
    wait(100)
    ca(screenshot_of, fm.tab_try, 'add_trial_position')

    # ------------------------ Add Orientations tab ----------------------
    ca(fm.notebook.SetSelection, 5)
    #Call the click event
#    ca(call_event, fm.tab_add.buttonCalculate, wx.EVT_BUTTON)
#    wait(1000)
#    assert len(model.experiment.exp.inst.positions)==1, "Length of positions calculated was to be 1, it was %d." % len(model.experiment.exp.inst.positions)
    ca(fm.tab_add.controller.textAngles[0].SetValue, "arange(0,180,10)")
    ca(screenshot_of, [fm.tab_add.boxSizerAngles] , 'add_positions-text', margin=20)
    wait(100)
    ca(screenshot_of, [fm.tab_add.buttonCalculate, fm.tab_add.buttonCancel] , 'add_positions-start_button', margin=20)
    ca(call_event, fm.tab_add.buttonCalculate, wx.EVT_BUTTON)
    wait(500)
    ca(screenshot_of, [fm.tab_add.gaugeProgress, fm.tab_add.staticTextProgress] , 'add_positions-progress_bar', margin=10)
    ca(screenshot_of, [fm.tab_add.buttonCalculate, fm.tab_add.buttonCancel] , 'add_positions-cancel_button', margin=20)
    wait(2000)
    #assert len(model.experiment.exp.inst.positions)==2, "Length of positions calculated was to be 19, it was %d." % len(model.experiment.exp.inst.positions)


#---END---
    
#    #Try position tab
#    fm.notebook.SetSelection(4)
#    #The "add positions" tab
#    fm.notebook.SetSelection(5)
#    wait(100)
#    call_event(fm.tab_add.buttonCalculate, wx.EVT_BUTTON)
#    wait(1000)
#    assert len(model.experiment.exp.inst.positions)==1, "Length of positions calculated."
#    fm.tab_add.controller.textAngles[0].SetValue("arange(0,180,10)")
#    wait(100)
#    call_event(fm.tab_add.buttonCalculate, wx.EVT_BUTTON)
#    wait(1000)
#    assert len(model.experiment.exp.inst.positions)==19, "Length of positions calculated was to be 19, it was %d." % len(model.experiment.exp.inst.positions)
    



#========================================================================================================
def generate_user_guide(fm, fv):
    """Script to generate screenshots for the CrystalPlan user guide.

    Parameters:
        fm: the FrameMain object.
        fv: the FrameQspaceView object
    """
    #Make sure we don't load the .pyc file
    filename = os.path.splitext(__file__)[0] + ".py"
#    filename = "/home/janik/Code/GenUtils/trunk/python/CrystalPlan/doc_maker/user_guide.py"
    print "Reading script from", filename
    #Read the code
    code = []
    f = open(filename, "rU")
    #Find the start
    for line in f:
        if line == "def user_guide_script():\n":
            break
    #Add each line
    for line in f:
        if line=="#---END---\n":
            break
        code.append(line.strip()) #remove indent
    f.close()
    print "... found", len(code), "lines of code."

    #Create the thread and start it
    thread = UserGuideThread(code, fm, fv)



