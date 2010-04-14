"""Module to generate screenshots for the
CrystalPlan user guide.
"""

import wx
from screenshots import *

import time
from time import sleep
from threading import Thread
import model

main_frame = None

def wait(ms):
    """Wait a given # of milliseconds."""
    print "Waiting for", ms, "ms"
    time.sleep(ms/1000.)

def ca(function, *args, **kwargs):
    """Alias for wx.CallAfter"""
    wx.CallAfter(function, *args, **kwargs)
    
#def ca(function, *args, **kwargs):
#    #The FunctionCall object holds the call to do
#    data = model.messages.FunctionCall(function, *args, **kwargs)
#    #Send a message to do this call.
#    model.messages.send_message(model.messages.MSG_SCRIPT_COMMAND, data)
#
#def ca(function, *args, **kwargs):
#    #Just call it right now! YAY!
#    function(*args, **kwargs)


def doprint(*args):
    for x in args:
        print x,
    print ""

def call_event(control, event_type):
    """Force an event to be called on the GUI."""
    evt = wx.CommandEvent(event_type.typeId, control.GetId())
    evt.SetEventObject(control)
    wx.PostEvent(control, evt)
#    control.GetEventHandler().ProcessEvent(evt)

def do_assert(assertion, message):
    assert assertion, message

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




#The following function will be executed line-by-line by a separate thread.
# - All commands should be on single lines
# - No loops or changes of indentation!
# - Finish with "#---END---\n"
def user_guide_script():
    ca(screenshot_frame, fm, 'frame_main')
    # ---- Q-Space Tab ------
    ca(fm.notebook.SetSelection, 1)


    #
    ca(fm.notebook.SetSelection, 4)
    wait(100)
    ca(screenshot_of, fm.tab_try, 'add_trial_position')
    ca(fm.notebook.SetSelection, 5)
    #Call the click event
    ca(call_event, fm.tab_add.buttonCalculate, wx.EVT_BUTTON)
    wait(1000)
    assert len(model.experiment.exp.inst.positions)==1, "Length of positions calculated was to be 1, it was %d." % len(model.experiment.exp.inst.positions)
#    ca(fm.tab_add.controller.textAngles[0].SetValue, "arange(0,180,10)")
    ca(screenshot_of, [fm.tab_add.boxSizerAngles] , 'add_positions-text', margin=20)
    wait(100)
    ca(screenshot_of, [fm.tab_add.buttonCalculate, fm.tab_add.buttonCancel] , 'add_positions-start_button', margin=20)
    ca(call_event, fm.tab_add.buttonCalculate, wx.EVT_BUTTON)
    wait(400)
    ca(screenshot_of, [fm.tab_add.gaugeProgress, fm.tab_add.staticTextProgress] , 'add_positions-progress_bar', margin=10)
    ca(screenshot_of, [fm.tab_add.buttonCalculate, fm.tab_add.buttonCancel] , 'add_positions-cancel_button', margin=20)
    wait(2000)
    assert len(model.experiment.exp.inst.positions)==19, "Length of positions calculated was to be 19, it was %d." % len(model.experiment.exp.inst.positions)
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



