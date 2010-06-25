"""Module to generate screenshots for the
CrystalPlan user guide.
"""

import wx
import sys
import os
import warnings
import screenshots
from screenshots import *

import time
from time import sleep
from threading import Thread

import model
import dialog_edit_crystal
import display_thread
import gui_utils
import frame_optimizer

main_frame = None

#-------------------------------------------------------------------------------
def wait(ms):
    """Wait a given # of milliseconds."""
#    print "Waiting for", ms, "ms"
    time.sleep(ms/1000.)

#-------------------------------------------------------------------------------
def waitfor( expression, timeout_sec=10, check_interval_sec=0.01, post_wait_sec=0.05):
    """Wait for a given expression to evaluate to true.

    Parameters:
        expression: string that evaluates to true or false
        timeout_sec: timeout in seconds.
        check_interval_sec: frequency to check, in seconds
        post_wait_sec: wait, in seconds, after the expression evaluates True

    Returns:
        True if the expression evaluated to True before timeout, False otherwise.
    """

    t_start = time.time()
    while time.time()-t_start < timeout_sec:
        result = eval(expression)
        if result:
            time.sleep(post_wait_sec)
            return True
        #Delay before checking
        time.sleep(check_interval_sec)
    print "Warning! waitfor('%s') timed out!" % expression
    return False

#-------------------------------------------------------------------------------
_waitfor_function = None
_waitfor_function_args = None
_waitfor_function_kwargs = None
_waitfor_start_value = None

#-------------------------------------------------------------------------------
def waitfor_set( function, *args, **kwargs):
    """Use in combination with waitfor_change. Sets a function that will be monitored for a
    change in result value.

    Parameters:
        function: function to evaluate.
        args and kwargs: arguments and keyword arguments to the function (optional)"""
    global _waitfor_function, _waitfor_function_args, _waitfor_function_kwargs, _waitfor_start_value
    _waitfor_function = function
    _waitfor_function_args = args
    _waitfor_function_kwargs = kwargs
    #Evaluate the function now to save the value.
    _waitfor_start_value = _waitfor_function(*_waitfor_function_args, **_waitfor_function_kwargs)


#-------------------------------------------------------------------------------
def waitfor_change(timeout_sec=10, check_interval_sec=0.01, post_wait_sec=0.05):
    """Wait for the return value of a function to change. Set the function with waitfor_set()
    before calling this.

    Parameters:
        timeout_sec: timeout in seconds.
        check_interval_sec: frequency to check, in seconds
        post_wait_sec: wait, in seconds, after the expression evaluates True
    """
    global _waitfor_function, _waitfor_function_args, _waitfor_function_kwargs, _waitfor_start_value
    t_start = time.time()
    while time.time()-t_start < timeout_sec:
        result = _waitfor_function(*_waitfor_function_args, **_waitfor_function_kwargs)
        if result != _waitfor_start_value:
            time.sleep(post_wait_sec)
            #Reset the start value for next time
            _waitfor_start_value = result
            return True
        #Delay before checking
        time.sleep(check_interval_sec)
    print "Warning! waitfor_change of %s() timed out!" % _waitfor_function.__name__
    return False


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
def check(checkbox, value):
    """Simulate checking a wx.CheckBox. Sends the proper event to it."""
    #@type checkbox wx.CheckBox
    checkbox.SetValue(value)
    call_event(checkbox, wx.EVT_CHECKBOX)

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

#        #Do the latex conversion
#        import eqhtml
#        eqhtml.embed_latex_in_html("../docs/user_guide_source.html", "../docs/user_guide.html")
#
#        #---- Put today's date ----
#        from datetime import date
#        d = date.today()
#        toc_inst = toc.Toc()
#        html = open("../docs/user_guide.html").read()
#        html = html.replace("{{date}}", d.strftime("%B %d, %Y"))
#
##        #---- Make table of contents ----
##        import toc
##        print "converting html of length", len(html)
##        file_output = open("../docs/user_guide.html", 'w')
##        file_output.write(
##            toc_inst.toc_template( '{{toc}}', html, prefix_li=False )
##            )
##        file_output.close()

        #Now run the script
        for line in self.code:
            print "-> SCRIPT: " + line
            #Don't process comment lines
            if len(line) > 0 and line[0] != "#":
                #Execute that line of code
                exec(line)
                #Wait a default of 10 ms between lines
                time.sleep(0.01)
                #Was line = taking a screenshot?
                if line.startswith('ca(screenshot'):
                    #Give it time to save
                    wait(50)

        #Ok we are done.
        print "-> Script Complete!"
        #Close the main frame to exit the program
        #fm.Destroy()



#==================================================================================================
#==================================================================================================
#==================================================================================================
#==================================================================================================
#==================================================================================================
#==================================================================================================
#==================================================================================================

#-----------------------------------------------------------------
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
    os.system("../doc_maker/apngasm ../docs/animations/frame_main-tab_anim.png " + " ".join(files) + " 5 10")
    for fname in files:
        os.remove(fname)


#-----------------------------------------------------------------
def make_animated_phi_rotation(slid, fv, filename):
    """Animate a rotation of phi or chi"""
    files = []
    for (i, angle) in enumerate(np.arange(-180, 181, 5)):
        ca(slid.SetValue, angle)
        wait(20)
        ca(slid.SendScrollEndEvent)
        wait(1800)
        fname = '3d-rotation_anim'+str(i)
        files.append("../docs/screenshots/" + fname + ".png")
        #Grab the 3d view
        ca(screenshot_of, fv.control, fname, margin=[10, 10, 10, 10], gradient_edge=5)
        wait(100)
        
    #Assemble into animated png
    command = "../doc_maker/apngasm ../docs/animations/" + filename + " " + " ".join(files) + " 1 15"
    print command
    os.system(command)
#    for fname in files:
#        os.remove(fname)

#-----------------------------------------------------------------
def pick_a_reflection():
    """Randomly pick a reflection"""
    #@type refl Reflection
    refl = model.experiment.exp.reflections[np.random.random_integers(0, len(model.experiment.exp.reflections))]
    while (refl.times_measured(add_equivalent_ones=True) < 4) or (not refl.is_primary):
        refl = model.experiment.exp.reflections[np.random.random_integers(0, len(model.experiment.exp.reflections))]
    print "I picked", refl


##-----------------------------------------------------------------
#def automatic_optimizer(fm):
#    #--- Automatic coverage optimizer ---
#    #@type fo FrameOptimizer
#    fo = frame_optimizer.get_instance(fm)
#    fo.Raise();
#    ca(screenshot_frame, fo, 'frame_optimizer')



#==========================================================================================
#==========================================================================================
#==========================================================================================
#The following function will be executed line-by-line by a separate thread.
# - All commands should be on single lines
# - No loops or changes of indentation! (Use functions if you need to make a for loop)
# - Finish with "#---END---\n"
def user_guide_script():

    #------- Initial --------------------------------------------
    #Shortcuts to the tested objects
    #@type fm FrameMain
    #@type fv FrameQspaceView
    exp = model.experiment.exp
    inst = model.experiment.exp.inst

    original_size = fm.GetSize()

    ca(screenshot_frame, fm, 'frame_main')
    wait(50)

    #Set the view in the 3D view
    (azimuth, elevation, distance, focalpoint) = fv.controller.scene.mlab.view()
    ca(fv.controller.scene.mlab.view, distance=distance/1.35)
    #Parallel projection off
    ca(fv.controller.scene.__setattr__, 'parallel_projection', False)

#    make_animated_tab_click(fm)
    #warnings.warn("Hey! Turn the animated tab maker back on!")
    
    # --------------------- Q-Space Tab --------------------
    ca(fm.notebook.SetSelection, 0)
    wait(100)
    #Settings for the guide
    #@type params StartupParameters
    params = fm.tab_startup.params
    params.d_min = 1.0
    params.q_resolution = 0.1
    params.wl_min = 0.5
    params.wl_max = 4.0
    wait(100)
    ca(screenshot_of, fm.tab_startup.control, 'startup-traits', minheight=True, margin=10, gradient_edge=0)
    ca(screenshot_of, fm.tab_startup.buttonApply, 'startup-apply', margin=5)
    ca(click, fm.tab_startup.buttonApply)
    wait(1200)

    # ------------------------ Detectors tab ----------------------
    ca(fm.notebook.SetSelection, 1)
    wait(100)
    #@type td PanelDetectors
    td = fm.tab_detectors
    wait(30)
    ca(screenshot_of, td.buttonLoadDetectors, 'detectors-buttonLoadDetectors',  margin=6, gradient_edge=4)
    wait(50)
    ca(screenshot_of, td.button_view_detectors, 'detectors-button_view_detectors', margin=6, gradient_edge=4)
    wait(30)
    ca(td.controller.load_detector_file, "../instruments/TOPAZ_detectors_all.csv")
    waitfor( 'len(model.instrument.inst.detectors) >= 48' )
    assert len(inst.detectors) == 48, "loaded 48 detectors from TOPAZ. We have %d" % len(inst.detectors)

#    #3d shot of detectors
#    ca(click, td.button_view_detectors)
#    wait(2000)
#    ca(screenshot_frame, td.frame3d, 'detectors-3d_view')
#    #Bring back the main window
#    ca(fm.Raise)
#    wait(100)


    # ------------------------ goniometer tab ----------------------
    ca(fm.notebook.SetSelection, 2)
    wait(250)
    #@type tg PanelGoniometer
    tg = fm.tab_goniometer
    ca(screenshot_of, tg.currentControl, 'goniometer-selected', minheight=True, margin=[10, 10, 40, 10], gradient_edge=4)
    ca(screenshot_of, tg.buttonEditAngles, 'goniometer-buttonEditAngles', margin=6, gradient_edge=0)
    ca(screenshot_of, tg.buttonApplyChanges, 'goniometer-buttonApplyChanges', margin=6, gradient_edge=0)
    ca(screenshot_of, tg.buttonSwitchGoniometer, 'goniometer-buttonSwitchGoniometer', margin=6, gradient_edge=0)
    #Select the TopazInHouseGoniometer and switch to it
    ca(select_name, tg.choiceGonio, model.goniometer.TopazInHouseGoniometer().name)
    wait(50)
    ca(screenshot_of, tg.choiceGonio, 'goniometer-choice', margin=6, gradient_edge=0)
    ca(screenshot_of, [tg.staticTextDesc, tg.staticTextDescLabel], 'goniometer-desc', margin=6, gradient_edge=0)
    ca(click, tg.buttonSwitchGoniometer)
    wait(100)
    assert isinstance(inst.goniometer, model.goniometer.TopazInHouseGoniometer), "we picked a TopazInHouseGoniometer"

    # ------------------------- Sample tab -----------------------
    ca(fm.notebook.SetSelection, 3)
    wait(250)
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
    ca(dlg.notebook.SetSelection, 1)
    wait(50)
    ca(screenshot_of, dlg.notebook, 'dialog_edit_crystal-notebook1', margin=10)
    ca(screenshot_of, dlg.buttonGenerateUB, 'dialog_edit_crystal-buttonGenerateUB', margin=6)
    ca(dlg.notebook.SetSelection, 0)
    wait(50)
    ca(screenshot_of, dlg.boxSizerIsaw, 'dialog_edit_crystal-notebook0', margin=[16, 16, 46, 16], gradient_edge=6, minheight=True)
    ca(screenshot_of, dlg.buttonReadUB, 'dialog_edit_crystal-buttonReadUB', margin=6)
    ca(screenshot_of, dlg.buttonOK, 'dialog_edit_crystal-buttonOK', margin=6)
    ca(screenshot_of, dlg.buttonCancel, 'dialog_edit_crystal-buttonCancel', margin=6)
    angles = np.array([30, 15, 60])
    (phi, chi, omega) = angles
    dlg.ub_orientation.phi_degrees = phi
    dlg.ub_orientation.chi_degrees = chi
    dlg.ub_orientation.omega_degrees = omega
    wait(50)
    ca(screenshot_of, dlg.control_load_angles, 'dialog_edit_crystal-control_load_angles', margin=6)
    ca(dlg.crystal.read_ISAW_ubmatrix_file, "../model/data/quartzub.txt", np.deg2rad(angles))
    wait(200)
    ca(screenshot_of, dlg.control_top, 'dialog_edit_crystal-control_top', margin=6)
    ca(click, dlg.buttonOK)
    wait(500)

    ca(screenshot_of, ts.range_control, 'sample-range_control', margin=10, gradient_edge=0)
    ca(screenshot_of, ts.buttonApplyRange, 'sample-buttonApplyRange', margin=10, gradient_edge=0)

    # ------------------------- Trial Positions tab -----------------------
    ca(fm.notebook.SetSelection, 4)
    wait(40)
    #Make the main window narrower
    ca(fm.SetSize, wx.Size(500, original_size[1]))
    wait(150)
    #@type tt PanelTryPosition
    tt = fm.tab_try
    ca(screenshot_of, tt.boxSizerAll, 'try', minheight=True, margin=[10, 10, 40, 10], gradient_edge=5)
    #Check the box
    ca(check, tt.checkAdd, True)
    wait(100)
    ca(screenshot_of, tt.checkAdd, 'try-checkAdd', margin=6, gradient_edge=2)
    wait(40)

    #@type slid ValueSlider
    slid = tt.sliders[0]
    ca(slid.SetValue, -30)
    wait(50)
    ca(slid.SendScrollEndEvent)
    wait(500)
    ca(screenshot_of, slid, 'try-phi-30', margin=6, gradient_edge=2)

    # ------------------------ 3D Viewer! ----------------------
    #The whole frame
    ca(fv.Raise) #Bring it to front first!
    wait(300)

    ca(screenshot_frame, fv, 'frame_qspace')

    #Animate a phi rotation
    wait(100)
    #make_animated_phi_rotation(slid, fv, "3d-phi_rotation_anim.png")
    #make_animated_phi_rotation(tt.sliders[1], fv, "3d-chi_rotation_anim.png")

    # ------------------------ Try Positions, bad goniometer ----------------------
    ca(fm.Raise)

    #Set a chi of +45 deg
    slid = tt.sliders[1]
    ca(slid.SetValue, +45)
    ca(slid.SendScrollEvent) #Trigger the showing the warning
    wait(150)
    ca(screenshot_of, slid, 'try-chi-45', margin=6, gradient_edge=2)
    ca(screenshot_of, [tt.staticTextWarning, tt.staticTextWarningReason], 'try-staticTextWarning', margin=6)
    ca(screenshot_of, tt.buttonSave, 'try-buttonSave', margin=6)
    #un-check the box
    ca(check, tt.checkAdd, False)


    # ------------------------ Add Orientations tab ----------------------
    ca(fm.notebook.SetSelection, 5)

    #@type ta PanelAddPositions
    ta = fm.tab_add

    waitfor_set(ta.textWarnings.GetValue)
    ca(ta.controller.textAngles[0].SetValue, "0, 10, 35.5")
    ca(ta.controller.textAngles[1].SetValue, "arange(0, 50, 12.5)")
    ca(ta.controller.textAngles[2].SetValue, "linspace(0, 360, 6)")
    waitfor_change()
    ca(screenshot_of, [ta.boxSizerAngles] , 'add-lists', margin=12)
    ca(ta.controller.textAngles[0].SetValue, "0")
    ca(ta.controller.textAngles[1].SetValue, "arange(-20, 20, 5)")
    ca(ta.controller.textAngles[2].SetValue, "0")
    waitfor_change()
    ca(screenshot_of, [ta.staticTextWarnings, ta.textWarnings], 'add-textWarnings', margin=12)
    ca(screenshot_of, ta.boxSizerAngles , 'add-lists2', margin=12)
    ca(ta.controller.textAngles[0].SetValue, "0, 30, 60, 90")
    ca(ta.controller.textAngles[1].SetValue, "0")
    ca(ta.controller.textAngles[2].SetValue, "0")
    waitfor_change()
    ca(screenshot_of, ta.boxSizerAngles , 'add-lists3', margin=12)
    wait(50)
    ca(screenshot_of, [ta.buttonCalculate, ta.buttonCancel] , 'add-start_button', margin=20)
    ca(click, ta.buttonCalculate)
    #Delay util the progress bar has something
    waitfor('len(model.experiment.exp.inst.positions) > 1')
    ca(screenshot_of, [ta.gaugeProgress, ta.staticTextProgress] , 'add-progress_bar', margin=10)
    wait(30)
    ca(screenshot_of, [ta.buttonCalculate, ta.buttonCancel] , 'add-cancel_button', margin=20)
    #Wait till done
    waitfor('len(model.experiment.exp.inst.positions) >= 4')
    wait(50)
    #assert len(model.experiment.exp.inst.positions)==2, "Length of positions calculated was to be 19, it was %d." % len(model.experiment.exp.inst.positions)

    # ------------------------ Experiment Plan tab ----------------------
    ca(fm.notebook.SetSelection, 6)

    #Restore the window width
    ca(fm.SetSize, original_size)
    wait(150)

    #@type te PanelExperiment
    te = fm.tab_experiment

    ca(te.gridExp.SelectBlock, 1,0,1,100)
    wait(50)
    ca(screenshot_of, te.gridExp, 'exp-grid', margin=12)
    ca(screenshot_of, te.boxSizerDelete, 'exp-delete_buttons', margin=8)
    ca(screenshot_of, [te.checkUseAll, te.buttonDontUseHighlighted], 'exp-select_buttons', margin=8)
    ca(screenshot_of, te.buttonSaveToCSV, 'exp-buttonSaveToCSV', margin=6)
    ca(screenshot_of, te.buttonOptimizer, 'exp-buttonOptimizer', margin=6)
    ca(screenshot_of, te.staticTextEstimatedTime, 'exp-estimated_time', margin=6)
    wait(50)

    # ------------------------ Back to 3D view----------------------
    #The q-space options panel
    #@type tv PanelQspaceOptions
    tv = fv.tabVolume

    ca(fv.Raise)
    wait(50)

    control_margins = [-180, -180, -30, -10]
    control_gradient_edge = 100
    check_margin = 4

    ca(screenshot_frame, fv, '3d-4orientations')
    ca(screenshot_of, fv.panelStats, '3d-panelStats', margin=10)
    ca(screenshot_of, tv.sliceControl, '3d-sliceControl', margin=10)


    ca(screenshot_of, tv.checkSymmetry, 'volume_options-checkSymmetry', margin=check_margin)
    ca(check, tv.checkSymmetry, True)
    wait(600)
    ca(screenshot_of, fv.control, '3d-symmetry', margin=control_margins, gradient_edge=control_gradient_edge)
    ca(screenshot_of, fv.panelStats, '3d-panelStats-symmetry', margin=10)

    ca(screenshot_of, tv.checkInvert, 'volume_options-checkInvert', margin=check_margin)
    ca(check, tv.checkInvert, True)
    wait(600)
    ca(screenshot_of, fv.control, '3d-inverted', margin=control_margins, gradient_edge=control_gradient_edge)
    ca(check, tv.checkInvert, False)
    ca(check, tv.checkSymmetry, False)
    ca(check, tv.checkShowRedundancy, True)
    wait(1200)

    ca(screenshot_of, tv.checkShowRedundancy, 'volume_options-checkShowRedundancy', margin=check_margin)
    ca(screenshot_of, fv.control, '3d-redundancy', margin=control_margins, gradient_edge=control_gradient_edge)

    ca(screenshot_of, tv.checkShowSlice, 'volume_options-checkShowSlice', margin=check_margin)
    tv.sliceControl.slice_min = 3
    tv.sliceControl.slice_max = 3.5
    wait(30)
    ca(check, tv.checkShowRedundancy, False)
    ca(check, tv.checkShowSlice, True)
    wait(600)
    ca(screenshot_of, tv.sliceControl, '3d-sliceControl-on', margin=10)
    ca(screenshot_of, fv.control, '3d-slice', margin=control_margins, gradient_edge=control_gradient_edge)
    ca(check, tv.checkShowRedundancy, True)
    wait(900)
    ca(screenshot_of, fv.control, '3d-slice-redundancy', margin=control_margins, gradient_edge=control_gradient_edge)


    # ------------------------ Reflections View ---------------------
    #@type tr PanelReflectionsViewOptions
    tr = fv.tabReflections

    #Select it
    ca(fv.notebookView.SetSelection, 1)
    wait(800)
    ca(screenshot_frame, fv, '3d-reflections')
    ca(screenshot_of, fv.panelStats, '3drefs-panelStats-normal', margin=10)

    #Only the measured peaks
    ca(tr.choiceView.SetSelection, 1)
    ca(call_event, tr.choiceView, wx.EVT_CHOICE)
    wait(400)
    ca(screenshot_of, fv.control, '3drefs-measured', margin=control_margins, gradient_edge=control_gradient_edge)
    ca(screenshot_of, tr.boxSizerDisplay, '3drefs-display', margin=10, gradient_edge=4)
    ca(screenshot_of, [tr.staticTextViewOption, tr.choiceView], '3drefs-view_option', margin=check_margin, gradient_edge=2)
    ca(screenshot_of, tr.checkUseSymmetry, '3drefs-checkUseSymmetry', margin=check_margin)

    #Use symmetry
    ca(check, tr.checkUseSymmetry, True)
    ca(tr.choiceView.SetSelection, 0)
    ca(call_event, tr.choiceView, wx.EVT_CHOICE)
    wait(400)
    ca(screenshot_of, fv.control, '3drefs-measured-symmetry', margin=control_margins, gradient_edge=control_gradient_edge)
    ca(screenshot_of, fv.panelStats, '3drefs-panelStats-symmetry', margin=10)

    ca(fv.controller.scene.__setattr__, 'parallel_projection', True)
    wait(400)
    ca(screenshot_of, fv.control, '3drefs-parallel', margin=control_margins, gradient_edge=control_gradient_edge)
    ca(fv.controller.scene.__setattr__, 'parallel_projection', False)
    wait(400)
    
    #--- Find a reflection to display ---
    #Simulate a left-click
    ca(fv.controller.on_button_press, FakeClickObject(50,50), None)
#    refl = pick_a_reflection()
#    #Select it
#    ca(fv.controller.select_reflection, refl)

    wait(250)
    ca(screenshot_of, fv.control, '3drefs-reflection_selected', margin=control_margins, gradient_edge=control_gradient_edge)
    #@type fri FrameReflectionInfo
    fri = fv.controller.open_frame_reflection_info()
    ca(screenshot_frame, fri, 'ref_info')
    #@type pri PanelReflectionInfo
    pri = fri.panel

    #Put in bad values for hkl
    ca(pri.textCtrlH.SetValue, "20")
    wait(40)
    ca(screenshot_of, [pri.staticTextHKLLabel, pri.gridSizerHKL], 'ref_info-bad_hkl', margin=10, gradient_edge=5)
    ca(pri.textCtrlH.SetValue, "3")
    ca(pri.textCtrlK.SetValue, "2")
    ca(pri.textCtrlL.SetValue, "-4")
    wait(300)
    size = fri.GetSize()
    ca(fri.SetSize, wx.Size(size[0], 600))
    ca(screenshot_of, pri, 'ref_info2', margin=5, gradient_edge=0)
    #Check it
    ca(check, pri.checkUseEquivalent, True)
    ca(screenshot_of, pri.checkUseEquivalent, 'ref_info-checkUseEquivalent', margin=4, gradient_edge=2)
    ca(screenshot_of, pri.boxSizerDivergence, 'ref_info-textCtrlDivergence', margin=4, gradient_edge=2)
    wait(150)
    ca(screenshot_of, pri, 'ref_info2-equivalent', margin=5, gradient_edge=0)


    #--------------- Measured Reflections -------------
    #@type tr PanelReflectionsViewOptions
    tr = fv.tabReflections
    #Load some peaks file
    ca(model.experiment.exp.load_peaks_file, "../model/data/TOPAZ_1241.integrate", append=False)
    wait(1000)
    ca(display_thread.handle_change_of_qspace)
    wait(1000)
    ca(screenshot_of, tr.boxSizerColor, 'refl-measurements',  margin=6, gradient_edge=2)

#---END---


    #--------------- The Reflection Placer -------------

    #@type prm PanelReflectionMeasurement
    prm = pri.measure_panels[0][0]
    ca(screenshot_of, prm, 'prm', margin=5, gradient_edge=0)
    ca(screenshot_of, prm.buttonPlace, 'prm-buttonPlace', margin=5, gradient_edge=0)
    ca(click, prm.buttonPlace)
    wait(800)

    #@type frp FrameReflectionPlacer
    global frp
    frp = prm.last_placer_frame
    #Wait for calculation to be done
    wait(3800)
    frp.placer.xy[0] = -5
    wait(100)
    ca(screenshot_frame, frp, 'ref_placer')
    ca(screenshot_of, frp.buttonAddOrientation, 'ref_placer-buttonAddOrientation', margin=4, gradient_edge=2)

#---END---
    

    #--- Automatic coverage optimizer ---
    #@type te PanelExperiment
    te = fm.tab_experiment
    ca(click, te.buttonOptimizer)
    wait(500)
    #@type fo FrameOptimizer
    fo = frame_optimizer.get_instance(fm)
    wait(500)
    fo.Raise()
    ca(screenshot_of, fo.buttonStart, 'optim-buttonStart',  margin=6, gradient_edge=2)
    ca(click, fo.buttonStart)
    wait(500)
    ca(screenshot_of, fo.buttonStop, 'optim-buttonStop',  margin=6, gradient_edge=2)
    ca(screenshot_of, fo.buttonApply, 'optim-buttonApply',  margin=6, gradient_edge=2)
    wait(2500)
    ca(screenshot_frame, fo, 'frame_optimizer')
    #Stop
    ca(click, fo.buttonStop)
    wait(500)
    ca(screenshot_of, fo.buttonKeepGoing, 'optim-buttonKeepGoing',  margin=6, gradient_edge=2)




class FakeClickObject:
    """Class for faking clicks sent to the VTK (mayavi) window"""
    def __init__(self, x, y):
        (self.x, self.y) = (x,y)
    def GetEventPosition(self):
        return (self.x, self.y)

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

    return thread



##==========================================================================================
##==========================================================================================
##==========================================================================================
##The following function will be executed line-by-line by a separate thread.
## - All commands should be on single lines
## - No loops or changes of indentation! (Use functions if you need to make a for loop)
## - Finish with "#---END---\n"
#def test_mask_error():
