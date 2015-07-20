"""Message handling routines for the model-view-controller.
Enables model code to send messages back to the GUI, when needed.
"""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
from wx.lib.pubsub import pub
from threading import Thread
import time


#========================================================================================================
# EXPERIMENT.PY
#========================================================================================================
MSG_POSITIONS_USED_CHANGED = "MSG_POSITIONS_USED_CHANGED"
MSG_EXPERIMENT_QSPACE_CHANGED = "MSG_EXPERIMENT_QSPACE_CHANGED"
MSG_EXPERIMENT_QSPACE_SETTINGS_CHANGED = "MSG_EXPERIMENT_QSPACE_SETTINGS_CHANGED"
MSG_EXPERIMENT_REFLECTIONS_CHANGED = "MSG_EXPERIMENT_REFLECTIONS_CHANGED"

#========================================================================================================
# INSTRUMENT.PY
#========================================================================================================
MSG_COVERAGE_CALCULATION_DONE = "MSG_COVERAGE_CALCULATION_DONE"
MSG_POSITION_LIST_CHANGED = "MSG_POSITION_LIST_CHANGED"
MSG_POSITION_LIST_APPENDED = "MSG_POSITION_LIST_APPENDED"
MSG_POSITION_SELECTION_CHANGED = "MSG_POSITION_SELECTION_CHANGED"

#========================================================================================================
# FRAME_MAIN.PY
#========================================================================================================
MSG_UPDATE_MAIN_STATUSBAR = "MSG_UPDATE_MAIN_STATUSBAR"

#========================================================================================================
#   OTHERS
#========================================================================================================
MSG_GONIOMETER_CHANGED = "MSG_GONIOMETER_CHANGED"
MSG_DETECTOR_LIST_CHANGED = "MSG_DETECTOR_LIST_CHANGED"
MSG_SCRIPT_COMMAND = "MSG_SCRIPT_COMMAND"

class FunctionCall:
    """Class holding a function and the arguments to it."""
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs


#----------------------------------------------------------------------
def send_message(message_id, data=None):
    """Thread-safe replacement for pubsub sendMessage."""
    wx.CallAfter(pub.sendMessage, topic=message_id, data=data)

#----------------------------------------------------------------------
def send_message_optional(object, message_id, data=None, delay=0.33):
    """Sends a message but only if a sufficient time has elapsed since the last
    time any message was sent from that object.
    Parameters:
        object: any persistant instance of an object. It will have an attribute added to it.
        message_id: unique ID for the message, usually a string.
        data: any data object.
        delay: in seconds, the closest in time two messages will be sent to the same object.
    """
    if not hasattr(object, '_last_message_send_time') or (time.time()-object._last_message_send_time > delay):
        wx.CallAfter(pub.sendMessage, topic=message_id, data=data)
        object._last_message_send_time = time.time()


#----------------------------------------------------------------------
def subscribe(function, topic):
    """Alias for wx.lib.pubsub.Publisher.subscribe()"""
    pub.subscribe(function, topic)


#----------------------------------------------------------------------
def unsubscribe(function, *args, **kwargs):
    """Alias for wx.lib.pubsub.Publisher.unsubscribe()"""
    pub.unsubscribe(function, *args, **kwargs)



#def subscribe(function, topic, must_indicate_done=False):
#    """Subscribe a function to a message topic.
#    Parameters:
#        must_indicate_done: set to True to say that the function has to indicate
#            that it is done before it can be executed again.
#    """
#    #Get the list of functions or create an empty list
#    funcs = message_thread.functions.get(topic, [])
#    #Make an ID
#    id = wx.NewId()
#    #Some initialization
#    function.message_id = id
#    function.must_indicate_done = must_indicate_done
#    funcs.append( function )
#    message_thread.functions[topic] = funcs
#    #Return the ID to the calling code
#    return id

def _do_process_message():
    pass


def done_processing_message(function, topic):
    pass







#========================================================================================================
#========================================================================================================
class MessageThread(Thread):
    """Thread to call messages but avoid piling up calls to the same function."""

    #Dictionary of lists of functions
    functions = dict()


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
            pass

    def abort(self):
        """Abort the thread. Should only be called upon program exit."""
        self._want_abort = True

#Create and start the thread
#message_thread = MessageThread()
