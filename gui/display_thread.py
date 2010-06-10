"""Module holds a background thread for updating the
user display, when the calculations are particularly slow.
It ensures that the UI remains responsive while the user
updates display options.
"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
from threading import Thread
import time
import copy
import numpy as np
import sys

#--- GUI Imports ---
import gui_utils

#--- Model Imports ---
import model
from model.experiment import ParamsDict

#The latest parameters that are being displayed.
LatestParams = ParamsDict()

#The next parameters that will be displayed, once the current ones are done.
#   This is None when no change is to occur.
NextParams = ParamsDict()



#========================================================================================================
#========================================================================================================
class DisplayThread(Thread):
    """Thread to keep updating the displayed q-space coverage as fast as possible
    while keeping the UI responsive."""
    
    _want_abort = False

    def __init__(self):
        """Constructor, also starts the thread."""
        Thread.__init__(self)
        # This starts the thread running on creation
        self.start()

    def run(self):
        """Continually runs and sees if there was a request to update part of 
        the display"""
        
        while not self._want_abort:
            #Loop until aborted
            there_were_changes = False

            #Check for changes, wrapped with exception handler
            try:
                there_were_changes = check_for_changes()
            except (KeyboardInterrupt, SystemExit):
                #Allow breaking the program
                raise
            except:
                #Unhandled exceptions get thrown to log and message boxes.
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback, thread_information="DisplayThread")
                #Hurray, the thread doesn't die!

            if there_were_changes:
                #The function does all the work. No need to sleep
                pass
            else:
                #No change in the requested display.
                #We just wait a bit
                time.sleep(0.1)
                
                
    def abort(self):
        """Abort the thread. Should only be called upon program exit."""
        self._want_abort = True


#========================================================================================================
#========================================================================================================

            
def check_for_changes():
    """Checks if any of the parameters have changed, requiring a graphical update.
    Performs the update if needed."""
    if any(NextParams.values()):
        #At least one of the entries is not None
#        print "DisplayThread: NextParams was set."
        ChangedParams = ParamsDict()

        for key in NextParams.keys():
            value = NextParams[key]
            if not (value is None):
#                print key, "has been set."
#                if key == "PARAM_POSITIONS":
#                    print "now is", value.positions
#                    if not LatestParams[key] is None: print "was", LatestParams[key].positions
                    
                #Compare current and older value, and check if an update is needed
                if not ( LatestParams[key] == value ):
                    #This part needs to update. Save it for later.
                    ChangedParams[key] = value
                    model.experiment.exp.params[key] = value
                    #Debuggy output
                    # print key, "has changed.",

        #Now we clear the "Next" parameters because we will have calculated them by the time this loop is done.
        for key in NextParams.keys():
            NextParams[key] = None

        #Now we calculate the coverage map by doing the stuff in ChangedParams
        pos = ChangedParams[model.experiment.PARAM_POSITIONS]
        trypos = ChangedParams[model.experiment.PARAM_TRY_POSITION]
        det = ChangedParams[model.experiment.PARAM_DETECTORS]
        symmetry = ChangedParams[model.experiment.PARAM_SYMMETRY]
        invert = ChangedParams[model.experiment.PARAM_INVERT]
        energy_slice = ChangedParams[model.experiment.PARAM_ENERGY_SLICE]
        slice = ChangedParams[model.experiment.PARAM_SLICE]
        display = ChangedParams[model.experiment.PARAM_DISPLAY]
        refls = ChangedParams[model.experiment.PARAM_REFLECTIONS]
        ref_mask = ChangedParams[model.experiment.PARAM_REFLECTION_MASKING]
        ref_display = ChangedParams[model.experiment.PARAM_REFLECTION_DISPLAY]

        #Keep track of what changed
        reflections_changed = False
        reflections_recalculated = False
        qspace_changed = False

        #--- Trial position ---
        changed_trypos = False
        if not trypos is None:
            #Do we need to calculate the trial position?
            poscov = trypos.try_position
            if poscov is None: #this shouldn't happen
                changed_trypos = False
            else:
                changed_trypos = True
                if trypos.use_trial and (poscov.coverage is None):
                    #We are using the trial position, and...
                    #The coverage is not calculated, we need to do it now.
                    print "Recalculating trial position with angles ", np.rad2deg(poscov.angles)
                    poscov.coverage = model.experiment.exp.inst.calculate_coverage(model.experiment.exp.inst.detectors, poscov.angles, sample_U_matrix=poscov.sample_U_matrix)

        if not gui_utils.inelastic_mode():
            #--- Reflections changing? ELASTIC MODE ONLY! ----
            if not (refls is None) or not (det is None):
                #The reflections need recalculating
                model.experiment.exp.recalculate_reflections(pos)
                reflections_changed = True
                reflections_recalculated = True
            elif not (ref_mask is None):
                #Just the mask is changing
                model.experiment.exp.calculate_reflections_mask()
                reflections_changed = True

            if not (ref_display is None):
                #Just reflection display options
                reflections_changed = True


        #--- Changes to qspace (volume) calcs ----
        if not (pos is None) or not (trypos is None) or not (det is None) or not (energy_slice is None):
            #The whole calculation needs to be done
            model.experiment.exp.calculate_coverage()
            qspace_changed = True
            #The reflections weren't recalculated already?
            if not reflections_recalculated and not gui_utils.inelastic_mode():
                model.experiment.exp.recalculate_reflections(pos)
                reflections_changed = True
                reflections_recalculated = True
                
        elif not (symmetry is None):
            #Just the symmetry - no need to recalc reflections
            model.experiment.exp.calculate_coverage()
            qspace_changed = True
            
        elif not (invert is None) or not (slice is None):
            #Invert gets redone (which then does the slice)
            model.experiment.exp.invert_coverage()
            qspace_changed = True

        #The display parameter does not require recalculationg, but it does require updating
        if not display is None:
            qspace_changed = True

        #Update the latest parameters array to what was last displayed
        for key in ChangedParams.keys():
            value = ChangedParams[key]
            if not value is None:
                LatestParams[key] = value

        #If anything was changed, make the UI update as needed.
        if qspace_changed or changed_trypos:
            model.messages.send_message(model.messages.MSG_EXPERIMENT_QSPACE_CHANGED)

        if reflections_changed or changed_trypos:
            model.messages.send_message(model.messages.MSG_EXPERIMENT_REFLECTIONS_CHANGED)

        return True
    else:
        return False


#-----------------------------------------------------------------------------------------------
def handle_change_of_qspace(changed_sample_U_matrix=None):
    """Handle a change of q-space size or resolution. Clear out items that need clearing.
    Force a redraw.
    
    Parameters:
        changed_sample_U_matrix: set to a 3x3 matrix if the sample orientation has changed too.
    """

    #Clear the try_position
    if not get_try_position() is None:
        if not get_try_position().try_position is None:
            #Clear it!
            get_try_position().try_position.coverage = None
            if not changed_sample_U_matrix is None:
                get_try_position().try_position.sample_U_matrix = changed_sample_U_matrix
            
    #TODO: Add a lock?
    #Copy the parameters over
    NextParams.update(LatestParams)
    #Clear the latest to force the thread to re-do everything.
    LatestParams.clear()
    #TODO: Fix the slice display parameter - it can be moved off-scale.
    
    #Re-init the qspace frame last?
    model.messages.send_message(model.messages.MSG_EXPERIMENT_QSPACE_SETTINGS_CHANGED)


#-----------------------------------------------------------------------------------------------
def manual_update():
    """Function that manually updates graphics if needed."""
    global thread_exists
    if thread_exists:
        pass
    else:
        print "---- display_thread.manual_update() ----"
        check_for_changes()

#-----------------------------------------------------------------------------------------------
def is_inverted():
    """Returns True if the q-space coverage is to be shown inverted."""
    inv = LatestParams[model.experiment.PARAM_INVERT]
    if inv is None:
        return False
    else:
        return inv.invert

#-----------------------------------------------------------------------------------------------
def is_detector_selected(num):
    """Returns True if the detector number 'num' is selected in the latest parameters."""
    det = LatestParams[model.experiment.PARAM_DETECTORS]
    if det is None:
        return True #No setting means all are checked by default
    else:
        return det.is_checked(num)

#-----------------------------------------------------------------------------------------------
def is_position_selected(poscov):
    """Returns True if the positionCoverage is selected in the latest parameters."""
    param = LatestParams[model.experiment.PARAM_POSITIONS]
    if param is None:
        return True #No setting means all are checked by default
    else:
        #Get the value, return False if not in dictionary
        return param.positions.get( id(poscov), False)
    
#-----------------------------------------------------------------------------------------------
def show_redundancy(value):
    """Sets whether the redundancy is displayed graphically (using transparent isosurfaces)."""
    display = copy.copy(get_display_params())
    display.show_redundancy = value
    NextParams[model.experiment.PARAM_DISPLAY] = display

#-----------------------------------------------------------------------------------------------
def get_display_params():
    """Returns the display parameters, or a default value if None"""
    display = LatestParams[model.experiment.PARAM_DISPLAY]
    if display is None:
        display = model.experiment.ParamDisplay()
    return display

#-----------------------------------------------------------------------------------------------
def get_reflection_display_params():
    """Returns the reflections display parameters, or a default value if None"""
    display = LatestParams[model.experiment.PARAM_REFLECTION_DISPLAY]
    if display is None:
        display = model.experiment.ParamReflectionDisplay()
    return display

#-----------------------------------------------------------------------------------------------
def get_reflection_masking_params():
    """Returns the reflections masking parameters, or a default value if None"""
    param = LatestParams[model.experiment.PARAM_REFLECTION_MASKING]
    if param is None:
        param = model.experiment.ParamReflectionMasking()
    return param

#-----------------------------------------------------------------------------------------------
def get_positions_dict():
    """Returns the dictionary containing all the selected positions."""
    pos = LatestParams[model.experiment.PARAM_POSITIONS]
    if pos is None:
        return dict()
    else:
        return pos.positions

#-----------------------------------------------------------------------------------------------
def select_additional_position_coverage(poscov_list, update_gui=False, select_items=True):
    """Check (select) or uncheck (unselect) ADDITIONAL PositionCoverage object(s) in the list of selected positions.
    The current list is maintained, these new ones are added/changed.

    Parameters:
        poscov_list: either a single PositionCoverage object or a list of them.
        update_gui: do we send a message to update the checked list in the GUI?
        select_items: True if we want to select them (default), or False to de-select.
    """
    posdict = get_positions_dict().copy()

    #Ensure a list
    if not isinstance(poscov_list, list):
        poscov_list = [poscov_list]
    for poscov in poscov_list:
        posdict[ id(poscov) ] = select_items

    #Take this modified one and put it as the next parameters.
    NextParams[model.experiment.PARAM_POSITIONS] = model.experiment.ParamPositions(posdict)

    #Do we update the gui?
    if update_gui:
        model.messages.send_message(model.messages.MSG_POSITION_SELECTION_CHANGED, posdict)

#-----------------------------------------------------------------------------------------------
def select_position_coverage(poscov_list, update_gui=False):
    """Check (select) PositionCoverage object(s) in the list of selected positions.
    Retrieves the latest selection and makes sure the given position(s) are selected.

    Parameters:
        poscov_list: either a single PositionCoverage object or a list of them.
            or, a dictionary of {id(poscov):bool} that is passed directly.
        update_gui: do we send a message to update the checked list in the GUI?
    """
    if isinstance(poscov_list, dict):
        #Use dictionary directly
        posdict = poscov_list
    else:
        #Ensure a list
        if not isinstance(poscov_list, list):
            poscov_list = [poscov_list]
        #Get the last dictionary, make a copy
        posdict = get_positions_dict().copy()
        for poscov in poscov_list:
            posdict[ id(poscov) ] = True
    #print "select_position_coverage",  posdict
    #Take this modified one and put it as the next parameters.
    NextParams[model.experiment.PARAM_POSITIONS] = model.experiment.ParamPositions(posdict)
    #Do we update the gui?
    if update_gui:
        model.messages.send_message(model.messages.MSG_POSITION_SELECTION_CHANGED, posdict)

#-----------------------------------------------------------------------------------------------
def clear_positions_selected():
    """De-select all entries in the list of selected positions."""
    #An empty dictionary will be = everything deselected.
    select_position_coverage( dict() )

#-----------------------------------------------------------------------------------------------
def get_try_position():
    """Returns the last set trial position parameters, if any."""
    return LatestParams[model.experiment.PARAM_TRY_POSITION]


#========================================================================================================
def get_position_coverage_from_id(poscov_id):
    """Return the PositionCoverage object of the same ID provided."""
    for poscov in model.instrument.inst.positions:
        if id(poscov)==poscov_id:
            return poscov
    trypos = get_try_position()
    if not trypos is None:
        return trypos.try_position
    return None


#Global to indicate that the display_thread is being used.
thread_exists = False