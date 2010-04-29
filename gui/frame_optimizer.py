#Boa:Frame:FrameOptimizer

import wx
import thread
import threading

import display_thread

#--- Model Imports ----
import model
from model.optimization import OptimizationParameters


if __name__=="__main__":
    #Manipulate the PYTHONPATH to put model directly in view of it
    import sys
    sys.path.insert(0, "..")
    

#================================================================================================
class OptimizationThread(threading.Thread):
    """Thread to run the GA optimization."""
    def __init__ (self, controller):
        threading.Thread.__init__(self)
        self.controller = controller
        self.start() #Start on creation

    def run(self):
        #Just run the optimization
        (best, aborted, converged) = model.optimization.run_optimization(self.controller.params, self.controller.step_callback)
        #Call the completion function.
        self.controller.complete( aborted, converged )
            

#================================================================================================
#================================================================================================
#================================================================================================
class OptimizerController():
    """Controller for the coverage optimizer."""

    #--------------------------------------------------------------------
    def __init__(self, frame):
        """Constructor.

        Parameters:
            frame: the FrameOptimizer instance."""
        self.frame = frame
        self.params = OptimizationParameters()
        self.best = None
        self.best_score = 0
        self.average_score = 0
        self.best_chromosome = []
        self.currentGeneration = 0
        self.run_thread = None

    #--------------------------------------------------------------------
    def update(self):
        """Update GUI elements to reflect the current status."""
        frm = self.frame #@type frm FrameOptimizer
        if frm is None:
            return
        #Coverage stats
        frm.staticTextCoverage.SetLabel("Best Coverage: %7.2f %%" % (self.best_score*100))
        frm.gaugeCoverage.SetValue(self.best_score*100)
        frm.staticTextAverage.SetLabel("Average Coverage: %7.2f %%" % (self.average_score*100))
        frm.gaugeAverage.SetValue(self.average_score*100)
        #The generation counter
        maxgen = self.params.max_generations
        frm.gaugeGeneration.SetRange(maxgen)
        frm.gaugeGeneration.SetValue(self.currentGeneration)
        frm.staticTextGeneration.SetLabel("Generation %5d of %5d:" % (self.currentGeneration, maxgen))
        #The individual
        if not self.best is None:
            frm.textStatus.SetValue(str(self.best))
            frm.buttonApply.Enable(True)
        else:
            frm.textStatus.SetValue("No best individual")
            frm.buttonApply.Enable(False)
        #The start/stop buttons enabling
        frm.buttonStart.Enable((self.run_thread is None))
        frm.buttonStop.Enable(not (self.run_thread is None))

    #--------------------------------------------------------------------
    def start(self, event, *args):
        """Start the optimization."""
        self._want_abort = False
        #Start the thread
#        model.optimization.run_optimization(self.frame.params, self.step_callback)
#        self.run_thread = thread.start_new_thread(model.optimization.run_optimization, (self.params, self.step_callback))
        self.run_thread = OptimizationThread(self)
        #Set the buttons right away.
        frm = self.frame #@type frm FrameOptimizer
        frm.buttonStart.Enable((self.run_thread is None))
        frm.buttonStop.Enable(not (self.run_thread is None))

        self.frame.staticTextComplete.SetLabel("Optimization started...")
        if not event is None: event.Skip()

    #--------------------------------------------------------------------
    def close_form(self, event, *args):
        """Call when the form is closing. Aborth the thread if it is running."""
        self._want_abort = True
        #Marker to avoid trying to change GUI
        self.frame = None
        if not event is None: event.Skip()

    #--------------------------------------------------------------------
    def stop(self, event, *args):
        """Stop the optimization."""
        self._want_abort = True
        #Will have to wait for the next generation to stop
        if not event is None: event.Skip()

    #--------------------------------------------------------------------
    def try_again(self, event, *args):
        """Re-start the optimization with more detectors."""
        self._want_abort = True
        #TODO: Wait for it to stop.
        if not event is None: event.Skip()

    #--------------------------------------------------------------------
    def complete(self,  aborted, converged):
        """Called when the optimization completes.
        
        Parameters:
            aborted: True if the optimization was aborted manually
            converged: True if the criterion was reached.
        """
        if aborted:
            label = "ABORTED - Optimization was aborted before completing!"
        elif converged:
            label = "SUCCESS - Optimization met the coverage criterion!"
        else:
            label = "FAILED - Reached the max. # of generations without enough coverage!"

        self.run_thread = None
        wx.CallAfter(self.frame.staticTextComplete.SetLabel, label)
        wx.CallAfter(self.update)

        if self.params.auto_increment and not aborted and not converged:
            #Try again with 1 more orientation
            self.params.number_of_orientations += 1
            wx.CallAfter(self.start, None)


    #--------------------------------------------------------------------
    def apply(self, event, *args):
        """Apply the best results."""
        #TODO: Confirmation message box?
        
        #Get the angles of the best one
        positions = model.optimization.get_angles(self.best)
        
        #This deletes everything in the list in the instrument
        del model.instrument.inst.positions[:]
        #Make sure to clear the parameters too, by giving it an empty dict() object.
        display_thread.clear_positions_selected()
        
        #Now add the new results
        out_positions = []
        for pos_cov_empty in positions:
            #Do the calculation
            poscov = model.instrument.inst.simulate_position(pos_cov_empty.angles, sample_U_matrix=pos_cov_empty.sample_U_matrix, use_multiprocessing=False, quick_calc=False)
            out_positions.append(poscov)

        #Add it to the list of selected items
        display_thread.select_position_coverage(out_positions, update_gui=True)
        if not event is None: event.Skip()
        
    #--------------------------------------------------------------------
    def step_callback(self, ga_engine, *args):
        """Callback during evolution; used to abort it and to display
        stats."""
        #Find the best individual
        self.best = ga_engine.bestIndividual()
        self.best_score = self.best.score
        #More stats
        stats = ga_engine.getStatistics()
        self.average_score = stats["rawAve"]
        #Other stats
        self.currentGeneration = ga_engine.currentGeneration
        
        if self.currentGeneration >= self.params.max_generations:
            print "Optimization complete!"
        #Update gui
        wx.CallAfter(self.update)
        return self._want_abort

    









#================================================================================================
#================================================================================================
#================================================================================================
def create(parent):
    return FrameOptimizer(parent)

[wxID_FRAMEOPTIMIZER, wxID_FRAMEOPTIMIZERBUTTONADDORIENTATION, 
 wxID_FRAMEOPTIMIZERBUTTONAPPLY, wxID_FRAMEOPTIMIZERBUTTONSTART, 
 wxID_FRAMEOPTIMIZERBUTTONSTOP, wxID_FRAMEOPTIMIZERGAUGECOVERAGE, 
 wxID_FRAMEOPTIMIZERGAUGEGENERATION, wxID_FRAMEOPTIMIZERPANELPARAMS, 
 wxID_FRAMEOPTIMIZERPANELSTATUS, wxID_FRAMEOPTIMIZERSPLITTERMAIN, 
 wxID_FRAMEOPTIMIZERSTATICLINE1, wxID_FRAMEOPTIMIZERSTATICTEXT1, 
 wxID_FRAMEOPTIMIZERSTATICTEXTCOVERAGE, 
 wxID_FRAMEOPTIMIZERSTATICTEXTGENERATION, 
 wxID_FRAMEOPTIMIZERSTATICTEXTRESULTS, wxID_FRAMEOPTIMIZERTEXTSTATUS, 
] = [wx.NewId() for _init_ctrls in range(16)]

#================================================================================================
class FrameOptimizer(wx.Frame):
    def _init_coll_boxSizerParams_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticText1, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticLine1, 0, border=0, flag=wx.EXPAND)

    def _init_coll_boxSizerStatus_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextResults, 0, border=4, flag=wx.LEFT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.textStatus, 1, border=10, flag=wx.EXPAND | wx.LEFT | wx.RIGHT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextGeneration, 0, border=4, flag=wx.LEFT)
        parent.AddWindow(self.gaugeGeneration, 0, border=10, flag=wx.EXPAND | wx.LEFT | wx.RIGHT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.EXPAND)
        parent.AddWindow(self.staticTextCoverage, 0, border=4, flag=wx.LEFT)
        parent.AddWindow(self.gaugeCoverage, 0, border=10, flag=wx.EXPAND | wx.LEFT | wx.RIGHT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=wx.EXPAND)
        parent.AddWindow(self.staticTextAverage, 0, border=4, flag=wx.LEFT)
        parent.AddWindow(self.gaugeAverage, 0, border=10, flag=wx.EXPAND | wx.LEFT | wx.RIGHT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextComplete, 0, border=4, flag=wx.LEFT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(wx.StaticLine(parent=self.panelStatus), 0, border=0, flag=wx.EXPAND | wx.LEFT | wx.RIGHT)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddSizer(self.gridSizerStatusButtons, 0, border=0, flag=wx.CENTER)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)

    def _init_coll_boxSizerAll_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.splitterMain, 1, border=8,
              flag=wx.TOP | wx.RIGHT | wx.LEFT | wx.EXPAND)
        parent.AddSpacer(wx.Size(16, 16), border=0,
              flag=wx.BOTTOM | wx.TOP | wx.RIGHT | wx.LEFT)

    def _init_coll_gridSizerStatusButtons_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.buttonStart, 0, border=0,
              flag=wx.ALIGN_CENTER_HORIZONTAL)
        parent.AddWindow(self.buttonAddOrientation, 0, border=0,
              flag=wx.ALIGN_CENTER_HORIZONTAL)
        parent.AddWindow(self.buttonStop, 0, border=0, flag=wx.ALIGN_CENTER)
        parent.AddWindow(self.buttonApply, 0, border=0, flag=wx.ALIGN_CENTER)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizerAll = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerStatus = wx.BoxSizer(orient=wx.VERTICAL)

        self.boxSizerParams = wx.BoxSizer(orient=wx.VERTICAL)

        self.gridSizerStatusButtons = wx.FlexGridSizer(cols=2, hgap=5, rows=2,
              vgap=4)

        self._init_coll_boxSizerAll_Items(self.boxSizerAll)
        self._init_coll_boxSizerStatus_Items(self.boxSizerStatus)
        self._init_coll_boxSizerParams_Items(self.boxSizerParams)
        self._init_coll_gridSizerStatusButtons_Items(self.gridSizerStatusButtons)

        self.SetSizer(self.boxSizerAll)
        self.panelStatus.SetSizer(self.boxSizerStatus)
        self.panelParams.SetSizer(self.boxSizerParams)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_FRAMEOPTIMIZER, name=u'FrameOptimizer',
              parent=prnt, pos=wx.Point(976, 171), size=wx.Size(715, 599),
              style=wx.DEFAULT_FRAME_STYLE, title=u'Coverage Automatic Optimizer')
        self.SetClientSize(wx.Size(900, 599))
        self.Bind(wx.EVT_CLOSE, self.controller.close_form)

        self.splitterMain = wx.SplitterWindow(id=wxID_FRAMEOPTIMIZERSPLITTERMAIN,
              name=u'splitterMain', parent=self, pos=wx.Point(8, 8),
              size=wx.Size(699, 575), style=wx.SP_3D)
        self.splitterMain.SetSashSize(8)
        self.splitterMain.SetSashGravity(0.4)

        self.panelParams = wx.Panel(id=wxID_FRAMEOPTIMIZERPANELPARAMS,
              name=u'panelParams', parent=self.splitterMain, pos=wx.Point(0, 0),
              size=wx.Size(10, 575), style=wx.TAB_TRAVERSAL)
        self.panelParams.SetBackgroundColour(wx.Colour(246, 246, 245))

        self.panelStatus = wx.Panel(id=wxID_FRAMEOPTIMIZERPANELSTATUS,
              name=u'panelStatus', parent=self.splitterMain, pos=wx.Point(18,
              0), size=wx.Size(681, 575), style=wx.TAB_TRAVERSAL)
        self.panelStatus.SetBackgroundColour(wx.Colour(200, 246, 245))
        self.splitterMain.SplitVertically(self.panelParams, self.panelStatus)

        self.staticText1 = wx.StaticText(id=wxID_FRAMEOPTIMIZERSTATICTEXT1,
              label=u'Optimization Parameters:', name='staticText1',
              parent=self.panelParams, pos=wx.Point(0, 8), style=0)
        self.staticText1.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, False, u'Sans'))

        self.staticLine1 = wx.StaticLine(id=wxID_FRAMEOPTIMIZERSTATICLINE1,
              name='staticLine1', parent=self.panelParams, pos=wx.Point(0, 33),
              size=wx.Size(419, 2), style=0)

        self.staticTextResults = wx.StaticText(id=wxID_FRAMEOPTIMIZERSTATICTEXTRESULTS,
              label=u'Current Status:', name=u'staticTextResults',
              parent=self.panelStatus, pos=wx.Point(0, 8), size=wx.Size(98, 17),
              style=0)

        self.textStatus = wx.TextCtrl(id=wxID_FRAMEOPTIMIZERTEXTSTATUS,
              name=u'textStatus', parent=self.panelStatus, pos=wx.Point(0, 33),
              style=wx.TE_MULTILINE + wx.TE_READONLY, value=u' ')

        self.gaugeGeneration = wx.Gauge(id=wxID_FRAMEOPTIMIZERGAUGEGENERATION,
              name=u'gaugeGeneration', parent=self.panelStatus, pos=wx.Point(0,
              424), range=100,  style=wx.GA_HORIZONTAL)

        self.staticTextGeneration = wx.StaticText(id=wxID_FRAMEOPTIMIZERSTATICTEXTGENERATION,
              label=u'Generation Progress:', name=u'staticTextGeneration',
              parent=self.panelStatus, pos=wx.Point(0, 407), style=0)

        self.gaugeCoverage = wx.Gauge(id=wxID_FRAMEOPTIMIZERGAUGECOVERAGE,
              name=u'gaugeCoverage', parent=self.panelStatus, pos=wx.Point(0,
              477), range=100,  style=wx.GA_HORIZONTAL)

        self.staticTextCoverage = wx.StaticText(id=wxID_FRAMEOPTIMIZERSTATICTEXTCOVERAGE,
              label=u'Best Coverage:', name=u'staticTextCoverage',
              parent=self.panelStatus, pos=wx.Point(0, 460), style=0)

        self.gaugeAverage = wx.Gauge(id=wxID_FRAMEOPTIMIZERGAUGECOVERAGE,
              name=u'gaugeAverage', parent=self.panelStatus, pos=wx.Point(0,
              477), range=100,  style=wx.GA_HORIZONTAL)

        self.staticTextAverage = wx.StaticText(id=wxID_FRAMEOPTIMIZERSTATICTEXTCOVERAGE,
              label=u'Average Coverage:', name=u'staticTextAverage',
              parent=self.panelStatus, pos=wx.Point(0, 460), style=0)

        self.staticTextComplete = wx.StaticText(id=wx.NewId(),
              label=u'...', name=u'staticTextComplete',
              parent=self.panelStatus, pos=wx.Point(0, 460), style=0)

        self.buttonAddOrientation = wx.Button(id=wxID_FRAMEOPTIMIZERBUTTONADDORIENTATION,
              label=u'Add an Orientation and Try Again...',
              name=u'buttonAddOrientation', parent=self.panelStatus,
              pos=wx.Point(380, 513), size=wx.Size(264, 29), style=0)
        self.buttonAddOrientation.Bind(wx.EVT_BUTTON, self.controller.try_again)

        self.buttonStart = wx.Button(id=wxID_FRAMEOPTIMIZERBUTTONSTART,
              label=u'Start Optimization', name=u'buttonStart',
              parent=self.panelStatus, pos=wx.Point(93, 513), size=wx.Size(152,
              29), style=0)
        self.buttonStart.SetToolTipString(u'Begin the optimization process in the background')
        self.buttonStart.Bind(wx.EVT_BUTTON, self.controller.start)

        self.buttonStop = wx.Button(id=wxID_FRAMEOPTIMIZERBUTTONSTOP,
              label=u'Stop!', name=u'buttonStop', parent=self.panelStatus,
              pos=wx.Point(126, 546), size=wx.Size(85, 29), style=0)
        self.buttonStop.Bind(wx.EVT_BUTTON, self.controller.stop)

        self.buttonApply = wx.Button(id=wxID_FRAMEOPTIMIZERBUTTONAPPLY,
              label=u'Apply Results...', name=u'buttonApply',
              parent=self.panelStatus, pos=wx.Point(441, 546), size=wx.Size(142,
              29), style=0)
        self.buttonApply.Bind(wx.EVT_BUTTON, self.controller.apply)

        self._init_sizers()

    def __init__(self, parent):
        #The view controller
        self.controller = OptimizerController(self)

        self._init_ctrls(parent)

        #Initial update.
        self.controller.update()

        #--- Parameters ----
        self.params_control = self.controller.params.edit_traits(parent=self.panelParams,kind='subpanel').control
        self.boxSizerParams.Insert(2, self.params_control, 0, border=1, flag=wx.EXPAND)
        self.boxSizerParams.Layout()





if __name__ == "__main__":
    import gui_utils
    (app, frm) = gui_utils.test_my_gui(FrameOptimizer)
    frm.Raise()
    app.MainLoop()
