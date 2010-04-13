"""Commands module, CURRENTLY UNUSED BY PROGRAM!
Implementation of the command pattern.
Methods for executing actions - or adding them to the queue as needed.
Thread that executes queued actions, keeps an undo stack.
"""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

from collections import deque
from threading import Thread
import time
import copy

import model



##========================================================================
#class GUIElements:
#    """Enum for the GUI elements that are affected by each command."""
#    (
#    position_list,
#    detector_list,
#    volume_3d_view,
#    points_3d_view,
#    ) = range(4)
#

#========================================================================
class Actions:
    """Enum containing the actions (calculations) or UI elements that need updating
    due to a command."""
    ( 
    full_coverage,
    invert_coverage,
    coverage_display,
    reflections,
    reflections_filter,
    reflections_display,
    list_of_positions,
    list_of_detectors
    ) = range(8)




#========================================================================
#========================================================================
#========================================================================
class Command(object):
    """Generic base class for command objects. Can handle simple commands
    where only the method calls and arguments are needed.

    More complex commands will use a derived class.
    """

    #-------------------------------------------------------------------
    def __init__(self, name, affects=None, method=None, *args, **kwargs):
        """Constructor for a generic command object.

        Parameters:
        -----------
            name: name of the command, to be shown in undo menus, for example.
            affects: list of which Calculations will need to be redone.
            method: method to call when executing the command.
            *args, **kwargs: arguments to be passed to the method.
        """
        self.name = name
        self.affects = affects
        self.method = method
        self.args = args
        self.kwargs = kwargs
        #Clear the undo to start
        self.undo_method = None
        self.undo_args = None
        self.undo_kwargs = None

    #-------------------------------------------------------------------
    def __call__(self):
        """Execute the command."""
        if self.method is None:
            raise NotImplementedError("Command's call method was not overridden or set!")
        else:
            self.method(*self.args, **self.kwargs)

    #-------------------------------------------------------------------
    def set_undo(self, method, *args, **kwargs):
        """Set the method to call when undoing the command.

        Parameters:
        -----------
            name: name of the command, to be shown in undo menus, for example.
            method: method to call when executing the command.
            *args, **kwargs: arguments to be passed to the method.
        """
        self.undo_method = method
        self.undo_args = args
        self.undo_kwargs = kwargs

    #-------------------------------------------------------------------
    def undo(self):
        """Execute the command's undo method."""
        if self.undo_method is None:
            raise NotImplementedError("Command's undo method was not overridden or set!")
        else:
            self.method(*self.undo_args, **self.undo_kwargs)






#========================================================================
#========================================================================
#========================================================================
class CommandController(Thread):
    """A controller object that handles executing command objects.
    """

    #Queue of commands yet to execute
    queue = deque()

    #Stack of commands ready to be undone
    undo_stack = list()

    #Calculations that now have to be redone.
    actions_to_do = set()

    #-------------------------------------------------------------------
    def add_command(self, cmd):
        """Add a command to the queue. """
        self.queue.append(cmd)
        
    #-------------------------------------------------------------------
    def do_next(self):
        """Execute the next command in the list"""
        #Get the command
        cmd = self.queue.popleft()
        #Call it!
        cmd()
        #Set the calculations to be redone
        for what in cmd.affects:
            self.actions_to_do.add(what)
        #Save it in the undo stack
        undo_stack.append(cmd)

    #-------------------------------------------------------------------
    def undo(self):
        """Run the latest undo command."""
        cmd = undo_stack.pop()
        #Call the undo now
        cmd.undo()
        #Set the calculations to be redone
        for what in cmd.affects:
            self.actions_to_do.add(what)

    #-------------------------------------------------------------------
    def process_queue(self):
        """Execute all the commands and do calculations as needed."""

        #Do each command. It should be okay if another command
        # is added at the end of the queue while a previous command is executing.
        while len(self.queue) > 0:
            self.do_next()

        #Do the calculations as needed
        # @type atd set
        atd = self.actions_to_do

        #--- Volume coverage calculations ----
        if Actions.full_coverage in atd:
            #Full recalculation of coverage
            model.experiment.exp.calculate_coverage()
        elif Actions.invert_coverage in atd:
            #Invert gets redone (which then does the slice)
            model.experiment.exp.invert_coverage()
        atd.discard(Actions.full_coverage)
        atd.discard(Actions.invert_coverage)

        # TODO: Notifications
        
        #--- Reflection point calculations ----
        if Actions.reflections in atd:
            model.experiment.exp.recalculate_reflections(None)


    #To abort the thread
    _want_abort = False

    #-------------------------------------------------------------------
    def __init__(self):
        """Constructor, also starts the thread."""
        Thread.__init__(self)
        # This starts the thread running on creation
        self.start()

    #-------------------------------------------------------------------
    def run(self):
        """Continually runs and sees if there was a request to update part of
        the display"""

        while not self._want_abort:
            #Loop until aborted
            if len(self.queue) > 0:
                #The function does all the work.
                self.process_queue()
            else:
                #No change in the requested display.
                #We just wait a bit
                time.sleep(0.07)

    #-------------------------------------------------------------------
    def abort(self):
        """Abort the thread. Should only be called upon program exit."""
        self._want_abort = True




#A global controller
ctlr = CommandController()

#========================================================================
#========================================================================
#========================================================================
def my_test(message, also="12"):
    print message, also

if __name__ == "__main__":
    cmd = Command("Test command", my_test, message="This comes from a test command!", also=123)
    cmd()

    