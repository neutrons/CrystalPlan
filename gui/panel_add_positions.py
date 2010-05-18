#Boa:FramePanel:PanelAddPositions
"""PanelAddPositions: GUI component to add positions to the calculated list, and run their coverage
calculation."""
import gui_utils

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx
from threading import Thread
import sys
import time
import numpy
from numpy import arange, linspace, pi

#--- GUI Imports ---
import display_thread

#--- Model Imports ---
import model

#--- Other Imports ---
try:
    import multiprocessing
    multiprocessing_installed = True
except ImportError, e:
    #Python version is likely < 2.6, therefore multiprocessing is not available
    multiprocessing_installed = False



[wxID_PANELADDPOSITIONS, wxID_PANELADDPOSITIONSBUTTONCALCULATE, 
 wxID_PANELADDPOSITIONSBUTTONCANCEL, 
 wxID_PANELADDPOSITIONSCHECKIGNOREGONIO, 
 wxID_PANELADDPOSITIONSCHECKMULTIPROCESSING, 
 wxID_PANELADDPOSITIONSGAUGEPROGRESS, wxID_PANELADDPOSITIONSSTATICTEXTHELP, 
 wxID_PANELADDPOSITIONSSTATICTEXTPROGRESS, 
 wxID_PANELADDPOSITIONSSTATICTEXTTITLE, 
 wxID_PANELADDPOSITIONSSTATICTEXTWARNINGS, wxID_PANELADDPOSITIONSTEXTWARNINGS, 
] = [wx.NewId() for _init_ctrls in range(11)]



#-------------------------------------------------------------------------------
#MESSAGES 
MSG_POSITION_CALCULATION_PROGRESS = "MSG_POSITION_CALCULATION_PROGRESS"
MSG_POSITION_CALCULATION_DONE = "MSG_POSITION_CALCULATION_DONE"
MSG_POSITION_CALCULATION_ABORTING = "MSG_POSITION_CALCULATION_ABORTING"

#========================================================================================================
#========================================================================================================
class CalculationThread(Thread):
    """Thread to run calculations (for detector coverage) in the background."""
    #Do we want to abort?
    _want_abort = False
    
    #Set to true to use multiple CPUs
    use_multiprocessing = False
    
    #list of angles to calculate
    positions = list()

    def __init__(self, positions_list, use_multiprocessing=False):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self._want_abort = 0
        self.positions = positions_list
        self.use_multiprocessing = use_multiprocessing
        #Create the list of results
        self.poscov_list = list()
        # This starts the thread running on creation
        self.start()

    def run(self):
        """Actually perform the calculation"""
        t_last = time.time()
        for i in range( len(self.positions) ):
            try:
                #These are the angles to calculate
                angles = self.positions[i]

                #GUI update only if it has been long enough
                if (time.time() - t_last) > 0.2:
                    model.messages.send_message(MSG_POSITION_CALCULATION_PROGRESS, i)
                    t_last = time.time()
                    
                #Update position list in GUI, but less often
                model.messages.send_message_optional(self, model.messages.MSG_POSITION_LIST_CHANGED, delay=1.5)

                #This performs the calculation
                newpos = model.instrument.inst.simulate_position(angles,
                    model.experiment.exp.crystal.get_u_matrix(),
                    use_multiprocessing=self.use_multiprocessing)
                self.poscov_list.append(newpos)
            except (KeyboardInterrupt, SystemExit):
                #Allow breaking the program
                raise
            except:
                #Unhandled exceptions get thrown to log and message boxes.
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback, thread_information="PanelAddPositions.CalculationThread")

            #Premature abortion?
            if self._want_abort:
                break
                
        #Ok, we either finished or aborted.
        model.messages.send_message(MSG_POSITION_CALCULATION_PROGRESS, i)
        model.messages.send_message( model.messages.MSG_POSITION_LIST_CHANGED)
        model.messages.send_message( MSG_POSITION_CALCULATION_DONE, self.poscov_list)

    def abort(self):
        """abort worker thread."""
        # Method for use by main thread to signal an abort
        self._want_abort = True
        model.messages.send_message(MSG_POSITION_CALCULATION_ABORTING, None)
                
        

#========================================================================================================
#========================================================================================================
class AddPositionsController():
    """Controller for adding positions"""
    panel = None
    
    #These will be the lists of angles
    valid = None
    invalid = None
    redundant = None
    
    #This will be the background calculation thread
    calculationThread = None
    
    #-------------------------------------------------------------------------------
    def __init__(self, panel):
        """panel: PanelAddPositions we are updating."""
        self.panel = panel
        #Subscribe to messages
        model.messages.subscribe(self.calculation_done, MSG_POSITION_CALCULATION_DONE)
        model.messages.subscribe(self.calculation_progress, MSG_POSITION_CALCULATION_PROGRESS)
        model.messages.subscribe(self.calculation_aborting, MSG_POSITION_CALCULATION_ABORTING)
        model.messages.subscribe( self.on_goniometer_changed, model.messages.MSG_GONIOMETER_CHANGED)

    #-------------------------------------------------------------------------------
    def on_goniometer_changed(self, *args):
        """Called when the goniometer used changes."""
        #Re-do the controls
        self.make_angle_controls()
        #Update the list calcualted
        self.OnTextbox_Text(None)


    #-------------------------------------------------------------------------------
    def make_angle_controls(self):
        """Complete GUI construction by adding the angles text boxes and things."""
        #Remove existing controls
        if hasattr(self, "sizerAngles"):
            for i in range(len(self.sizerAngles)):
                self.textAngles[i].Destroy()
                self.staticTextAngles[i].Destroy()
                self.staticTextUnits[i].Destroy()
#                self.sizerAngles[i].Clear() #Causes crash?
#                self.sizerAngles[i].Destroy()
        self.panel.boxSizerAngles.Clear()

        #These will be the controls
        self.textAngles = list()
        self.staticTextAngles = list()
        self.sizerAngles = list()
        self.staticTextUnits = list()

        i = 0
        for ang in model.instrument.inst.angles:
            #Go through each angleinfo object
            if not ang is None:
                #Do the static text
                static = wx.StaticText(id=wx.NewId(), label=u"List of " + ang.name + ": ",
                    name=u'staticTextAngles'+str(i),
                    parent=self.panel, pos=wx.Point(0, 0), size=wx.Size(151, 14), style=0)
                static_unit = wx.StaticText(id=wx.NewId(), label=u" " + ang.friendly_units,
                    name=u'staticTextAnglesUnits'+str(i),
                    parent=self.panel, pos=wx.Point(0, 0), size=wx.Size(50, 14), style=0)

                #Now the editable textbox
                id = wx.NewId()
                text = wx.TextCtrl(id=id, name=u'textAngles'+str(i), parent=self.panel, pos=wx.Point(0, 0),
                      size=wx.Size(100, 30), style=0, value=u'0')
                text.Bind(wx.EVT_TEXT, self.OnTextbox_Text, id=id)

                #Make a sizer
                sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
                sizer.AddWindow(static, 0, border=0, flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
                sizer.AddWindow(text, 1, border=0, flag=wx.EXPAND)
                sizer.AddWindow(static_unit, 0, border=0, flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)

                #Add the sizer to the other one
                self.panel.boxSizerAngles.AddSizer(sizer, 0, border=0, flag=wx.EXPAND)

                #Add the controls
                self.textAngles.append(text)
                self.staticTextAngles.append(static)
                self.staticTextUnits.append(static_unit)
                self.sizerAngles.append(sizer)

                i += 1
                
        self.panel.boxSizerAngles.Layout()
        self.panel.boxSizerAll.Layout()

    #-------------------------------------------------------------------------------
    def OnTextbox_Text(self, event):
        """Called when any of the 3 angle textboxes are changed."""
        angles_lists = list()
        for text in self.textAngles:
            angles_lists.append(text.GetValue())

        self.evaluate(angles_lists, self.panel.checkIgnoreGonio.GetValue() )
        if not event is None: event.Skip()
        
    #-------------------------------------------------------------------------------
    def make_list(self, input, anginfo):
        """Evaluate a string to make a numpy array. Converts units.
        Parameters:
            input: string with the angles as string, in friendly units.
            anginfo: the AngleInfo object corresponding to it. Will perform unit
                conversion
        """
        #@type anginfo AngleInfo
        try:
            array = eval( 'numpy.array([' + input + '])' )
            #Make sure it is a 1D array by reshaping
            array = array.reshape( array.size )
            #Set to just a zero if empty
            if len(array)==0: array = numpy.array([0])
            #Convert to radians or whatever the internal unit is
            array = anginfo.friendly_to_internal(array)
        except Exception, e:
            return ("Error reading '" + input + "': " + str(e) + "\n", None)
        else:
            return ("", array)      
        
    #-------------------------------------------------------------------------------
    def print_list(self, list, extra_strings=None):
        """Convert the array list back to degrees and display it"""
        import numpy
        s = ""
        for (i, arr) in enumerate(list):
            s += model.instrument.inst.make_angles_string(arr)
            if not extra_strings is None:
                 s += " (%s)" % extra_strings[i]
            s += "\n"
        return s
    
    #-------------------------------------------------------------------------------
    def evaluate(self, angles_lists, ignore_gonio):
        """Evaluate the strings from the text boxes.
            ignore_gonio: if True, allow all angles.
        """
        errors = ""
        angles_values = list()
        for (list_str, anginfo) in zip(angles_lists, model.instrument.inst.angles):
            (err, new_values) = self.make_list(list_str, anginfo)
            errors = errors + err
            angles_values.append(new_values)
            
        if len(errors) > 0:
            #Error in the list textbox; don't evaluate.
            self.panel.textWarnings.SetValue(errors)
            self.panel.buttonCalculate.Enable(False)
        else:
            (self.valid, self.redundant, self.invalid, invalid_reasons) = model.instrument.inst.evaluate_position_list(angles_values, ignore_gonio)
            s = "%s angles will be calculated.\n   Ignored: %s invalid angles; %s redundant angles.\n" % (len(self.valid), len(self.invalid), len(self.redundant))
            if len(self.invalid) > 0:
                s = s + "-- The following angles are invalid - they cannot be achieved with the goniometer! --\n%s\n" % self.print_list(self.invalid, invalid_reasons)
            if len(self.redundant) > 0:
                s = s + "-- These angles have already been calculated: --\n%s\n" % self.print_list(self.redundant)
            if len(self.valid) > 0:
                s = s + "-- These angles will be calculated: --\n%s\n" % self.print_list(self.valid)
            else:
                s = s + "-- NO VALID ANGLES FOUND! --\n"
            self.panel.textWarnings.SetValue(s)
            self.panel.buttonCalculate.Enable( len(self.valid) > 0 )
            
    #-------------------------------------------------------------------------------
    def calculation_done(self, message):
        """Message handler for the calculations of coverage. Runs when calculation is finished (or aborted)."""
        #Reset GUI elements.
        self.panel.buttonCalculate.SetLabel("Begin Calculation")
        self.panel.buttonCancel.SetLabel("Cancel Calculation")
        self.panel.buttonCalculate.Enable(True)
        self.panel.buttonCancel.Enable(False)
        self.panel.gaugeProgress.SetValue(0)
        self.calculationThread = None
        #This is the list of what we did calculate.
        poscov_list = message.data
        #Let's select all these new positions, and show them
        display_thread.select_position_coverage(poscov_list, update_gui=True)

                
    #-------------------------------------------------------------------------------
    def calculation_progress(self, message):
        """Handles displaying progress bar during calculation..."""
        self.panel.gaugeProgress.SetValue(message.data)
        
    #-------------------------------------------------------------------------------
    def calculation_aborting(self, message):
        """GUI updates, saying that it is in the process of aborting."""
        self.panel.buttonCancel.SetLabel("... Aborting ...")
            
    #-------------------------------------------------------------------------------
    def execute(self):
        """Send a command to begin the calculation."""
        self.panel.buttonCalculate.Enable(False)
        self.panel.buttonCalculate.SetLabel("... calculating ...")
        self.panel.buttonCalculate.Refresh()
        self.panel.buttonCancel.Enable(True)
        self.panel.gaugeProgress.SetRange( len(self.valid) )
        #This will start it!
        self.calculationThread = CalculationThread(self.valid, self.panel.checkMultiprocessing.GetValue())
        
    #-------------------------------------------------------------------------------
    def abort(self):
        """Abort the calculation thread, if any."""
        if not (self.calculationThread is None):
            self.calculationThread.abort()
            
            
            



#========================================================================================================
class PanelAddPositions(wx.Panel):
                
    
    def _init_coll_boxSizerMiddle_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkIgnoreGonio, 0, border=0, flag=wx.EXPAND)

    def _init_coll_boxSizerBottom_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.buttonCalculate, 0, border=2,
              flag=wx.ALIGN_CENTER_HORIZONTAL)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.buttonCancel, 0, border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.staticTextTitle, 0, border=0,
              flag=wx.ALIGN_CENTER_HORIZONTAL)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextHelp, 0, border=0,
              flag=wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSizer(self.boxSizerAngles, 0, border=0,
              flag=wx.RIGHT | wx.LEFT | wx.GROW)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSizer(self.boxSizerMiddle, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextWarnings, 0, border=0,
              flag=wx.ALIGN_CENTER_HORIZONTAL)
        parent.AddWindow(self.textWarnings, 1, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextProgress, 0, border=0, flag=0)
        parent.AddWindow(self.gaugeProgress, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSizer(self.boxSizer1, 0, border=0, flag=0)
        parent.AddSizer(self.boxSizerBottom, 0, border=0, flag=wx.ALIGN_CENTER)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)

    def _init_coll_boxSizer1_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.checkMultiprocessing, 0, border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerBottom = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.boxSizer1 = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.boxSizerMiddle = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.boxSizerAngles = wx.BoxSizer(orient=wx.VERTICAL)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerBottom_Items(self.boxSizerBottom)
        self._init_coll_boxSizer1_Items(self.boxSizer1)
        self._init_coll_boxSizerMiddle_Items(self.boxSizerMiddle)

        self.SetSizer(self.boxSizerAll)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_PANELADDPOSITIONS,
              name=u'PanelAddPositions', parent=prnt, pos=wx.Point(1783, 182),
              size=wx.Size(546, 456), style=wx.TAB_TRAVERSAL)
        self.SetClientSize(wx.Size(546, 456))
        self.SetAutoLayout(True)

        self.staticTextHelp = wx.StaticText(id=wxID_PANELADDPOSITIONSSTATICTEXTHELP,
              label=u'A list can be entered as [0, 10, 20] or arange(0, 60, 10), meaning step from 0 to 60 in steps of 10; or linspace(0, 60, 6), meaning go from 0 to 60 in 6 steps.',
              name=u'staticTextHelp', parent=self, pos=wx.Point(0, 25),
              style=0)

        self.staticTextTitle = wx.StaticText(id=wxID_PANELADDPOSITIONSSTATICTEXTTITLE,
              label=u'Enter Angles in Degrees:', name=u'staticTextTitle',
              parent=self, pos=wx.Point(193, 0), size=wx.Size(160, 17),
              style=0)

        self.buttonCalculate = wx.Button(id=wxID_PANELADDPOSITIONSBUTTONCALCULATE,
              label=u'  Begin Calculation  ', name=u'buttonCalculate', parent=self,
              pos=wx.Point(65, 419), style=0)
        self.buttonCalculate.Enable(False)
        self.buttonCalculate.Bind(wx.EVT_BUTTON, self.OnButtonCalculateButton,
              id=wxID_PANELADDPOSITIONSBUTTONCALCULATE)

        self.staticTextWarnings = wx.StaticText(id=wxID_PANELADDPOSITIONSSTATICTEXTWARNINGS,
              label=u'Warnings or Errors:', name=u'staticTextWarnings',
              parent=self, pos=wx.Point(237, 108), size=wx.Size(71, 17),
              style=0)

        self.textWarnings = wx.TextCtrl(id=wxID_PANELADDPOSITIONSTEXTWARNINGS,
              name=u'textWarnings', parent=self, pos=wx.Point(0, 125),
              size=wx.Size(546, 211), style=wx.TE_MULTILINE, value=u'...')
        self.textWarnings.SetEditable(False)
        self.textWarnings.SetMinSize(wx.Size(-1, -1))

        self.buttonCancel = wx.Button(id=wxID_PANELADDPOSITIONSBUTTONCANCEL,
              label=u'  Cancel Calculation  ', name=u'buttonCancel', parent=self,
              pos=wx.Point(281, 419), style=0)
        self.buttonCancel.Enable(False)
        self.buttonCancel.Bind(wx.EVT_BUTTON, self.OnButtonCancelButton,
              id=wxID_PANELADDPOSITIONSBUTTONCANCEL)

        self.gaugeProgress = wx.Gauge(id=wxID_PANELADDPOSITIONSGAUGEPROGRESS,
              name=u'gaugeProgress', parent=self, pos=wx.Point(0, 361),
              range=100, style=wx.GA_HORIZONTAL)

        self.staticTextProgress = wx.StaticText(id=wxID_PANELADDPOSITIONSSTATICTEXTPROGRESS,
              label=u'Progress:', name=u'staticTextProgress', parent=self,
              pos=wx.Point(0, 344), size=wx.Size(60, 17), style=0)

        self.checkMultiprocessing = wx.CheckBox(id=wxID_PANELADDPOSITIONSCHECKMULTIPROCESSING,
              label=u'Use Multiple CPU Processing',
              name=u'checkMultiprocessing', parent=self, pos=wx.Point(0, 397),
              size=wx.Size(222, 22), style=0)
        self.checkMultiprocessing.SetValue(False)
        self.checkMultiprocessing.Bind(wx.EVT_CHECKBOX,
              self.OnCheckMultiprocessingCheckbox,
              id=wxID_PANELADDPOSITIONSCHECKMULTIPROCESSING)

        self.checkIgnoreGonio = wx.CheckBox(id=wxID_PANELADDPOSITIONSCHECKIGNOREGONIO,
              label=u'Ignore goniometer limits (allow all angles)',
              name=u'checkIgnoreGonio', parent=self, pos=wx.Point(0, 78),
              size=wx.Size(320, 22), style=0)
        self.checkIgnoreGonio.SetValue(False)
        self.checkIgnoreGonio.Bind(wx.EVT_CHECKBOX,
              self.OnCheckIgnoreGonioCheckbox,
              id=wxID_PANELADDPOSITIONSCHECKIGNOREGONIO)

        self._init_sizers()

        

    def __init__(self, parent):
        self._init_ctrls(parent)        
        #Additional code
        self.controller = AddPositionsController(self)
        self.controller.make_angle_controls()
        self.controller.OnTextbox_Text(None)

        #Disable the multiprocessing checkbox if using an older python.
        if multiprocessing_installed:
            self.checkMultiprocessing.Enable()
            #TODO: Re-enable this checkbox when multiprocessing can be used again.
            self.checkMultiprocessing.Disable()
        else:
            self.checkMultiprocessing.Disable()


#-------------------------------------------------------------------------------
#------------------ EVENT HANDLERS ---------------------------------------------
#-------------------------------------------------------------------------------


    def OnButtonCalculateButton(self, event):
        self.controller.execute()
        event.Skip()

    def OnButtonCancelButton(self, event):
        self.controller.abort()
        event.Skip()

    def OnCheckMultiprocessingCheckbox(self, event):
        event.Skip()

    def OnCheckIgnoreGonioCheckbox(self, event):
        #To re-evaluate
        self.controller.OnTextbox_Text(None)
        event.Skip()












if __name__ == '__main__':
    #Test routine
    import gui_utils
    model.instrument.inst = model.instrument.Instrument()
    model.goniometer.initialize_goniometers()
    (app, pnl) = gui_utils.test_my_gui(PanelAddPositions)
    app.MainLoop()

