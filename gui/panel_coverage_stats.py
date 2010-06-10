"""Small panel showing coverage statistics with text
and graphical gauges."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- General Imports ---
import wx

#-------------------------------------------------------------------------------
#-------- Coverage Stats Panel -----------------------------------------------
#-------------------------------------------------------------------------------
class PanelCoverageStats(wx.Panel):
    """Small panel showing coverage statistics with text
    and graphical gauges.
    """

    def _init_coll_boxSizerStats_Items(self, parent):
        parent.AddWindow(self.staticTextStats1, 0, border=0,
              flag=wx.ALIGN_CENTER)
        parent.AddSpacer(wx.Size(8,8))
        parent.AddWindow(self.staticTextReflections,0, border=8, flag=wx.EXPAND | wx.BOTTOM)
        parent.AddWindow(self.staticTextStatsCovered, 0, border=0,
              flag=wx.EXPAND)
        parent.AddWindow(self.gaugeCoverage, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8,8))
        parent.AddWindow(self.staticTextStatsRedundant, 0, border=0, flag=wx.EXPAND)
        parent.AddWindow(self.gaugeRedundancy, 0, border=0, flag=wx.EXPAND)

    def _init_sizers(self):
        self.boxSizerStats = wx.BoxSizer(orient=wx.VERTICAL)
        self._init_coll_boxSizerStats_Items(self.boxSizerStats)
        self.SetSizer(self.boxSizerStats)


    def _init_ctrls(self, prnt):
        self.SetAutoLayout(True)

        self.staticTextStats1 = wx.StaticText(id=wx.ID_ANY,
              label=u'Coverage Statistics:', name=u'staticTextStats1',
              parent=self, pos=wx.Point(25, 0), style=0)
        self.staticTextStats1.SetFont(wx.Font(11, wx.SWISS, wx.NORMAL, wx.BOLD,
              False, u'Sans'))

        self.staticTextReflections = wx.StaticText(label=u'111 reflections.', name=u'staticTextReflections',
              parent=self, pos=wx.Point(0, 26), style=0)

        self.staticTextStatsCovered = wx.StaticText(id=wx.ID_ANY,
              label=u'Coverage %:', name=u'staticTextStatsCovered',
              parent=self, pos=wx.Point(0, 26), style=0)

        self.staticTextStatsRedundant = wx.StaticText(id=wx.ID_ANY,
              label=u'Redundancy:', name=u'staticTextStatsRedundant',
              parent=self, pos=wx.Point(0, 79), style=0)

        self.gaugeCoverage = wx.Gauge(id=wx.ID_ANY,
              name=u'gaugeCoverage', parent=self, pos=wx.Point(0,
              43), range=100, style=wx.GA_HORIZONTAL)
        self.gaugeCoverage.SetValue(0)
        self.gaugeCoverage.SetLabel(u'')
        self.gaugeCoverage.SetForegroundColour(wx.Colour(0, 0, 0))
        self.gaugeCoverage.SetBezelFace(1)
        self.gaugeCoverage.SetShadowWidth(1)

        self.gaugeRedundancy = wx.Gauge(id=wx.ID_ANY,
              name=u'gaugeRedundancy', parent=self, pos=wx.Point(0,
              96), range=100, style=wx.GA_HORIZONTAL)
        self.gaugeRedundancy.SetValue(0)
        self.gaugeRedundancy.SetLabel(u'')
        self.gaugeRedundancy.SetForegroundColour(wx.Colour(0, 0, 0))
        self.gaugeRedundancy.SetBezelFace(1)
        self.gaugeRedundancy.SetShadowWidth(1)

        self._init_sizers()


    def __init__(self, parent, id, pos, size, style, name):
        #Parent constructor
        wx.Panel.__init__(self, parent, id, pos, size, style, name)
        self._init_ctrls(parent)


    #-----------------------------------------------------------------------------------------------
    def show_stats(self, symmetry, coverage_pct, redundant_pct):
        """Update the information displayed on the statistics panel."""
        if symmetry:
            self.staticTextStats1.SetLabel("Coverage with Symmetry:")
        else:
            self.staticTextStats1.SetLabel("Full Sphere Coverage:")

        #Show those
        self.staticTextStatsCovered.SetLabel("Coverage of %5.1f%%:" % coverage_pct)
        self.staticTextStatsRedundant.SetLabel("%5.1f%% measured > once:" % redundant_pct)
        self.gaugeCoverage.SetValue(coverage_pct)
        self.gaugeCoverage.SetToolTipString(u'Gauge showing the % covered.')
        self.gaugeRedundancy.SetValue(redundant_pct)
        self.gaugeRedundancy.SetToolTipString(u'Gauge showing the % measured more than once (redundancy).')

        #Do we see the count of reflections?
        self.staticTextReflections.Hide()
        self.boxSizerStats.Layout()

    #-----------------------------------------------------------------------------------------------
    def show_reflection_stats(self, use_symmetry, stats):
        """Update the information displayed on the statistics panel; for reflection counts.

        Parameters:
            use_symmetry: True if we are considering crystal symmetry.
            stats: a ReflectionStats object with the data.
        """
        #@type stats ReflectionStats
        if use_symmetry:
            self.staticTextStats1.SetLabel("Coverage (w/ symmetry)")
            self.staticTextReflections.SetLabel("%d unique reflections" % stats.total)
        else:
            self.staticTextStats1.SetLabel("Coverage:")
            self.staticTextReflections.SetLabel("%d reflections" % stats.total)

        #Show those
        if stats.total <= 0:
            coverage_pct = 100.
            redundant_pct = 0.
        else:
            coverage_pct = (100.*stats.measured / stats.total)
            redundant_pct = (100.*stats.redundant / stats.total)
            
        self.staticTextStatsCovered.SetLabel("Coverage of %5.1f%%:" % coverage_pct)
        self.staticTextStatsRedundant.SetLabel("%5.1f%% measured > once:" % redundant_pct)
        self.gaugeCoverage.SetValue(coverage_pct)
        self.gaugeCoverage.SetToolTipString(u'%d reflections were measured, out of %d' % (stats.measured, stats.total))
        self.gaugeRedundancy.SetValue(redundant_pct)
        self.gaugeRedundancy.SetToolTipString(u'%d reflections were measured more than once, out of %d' % (stats.redundant, stats.total))
        #Do we see the count of reflections?
        self.staticTextReflections.Show()
        self.boxSizerStats.Layout()



if __name__ == '__main__':
    #Test routine
    from wxPython.wx import *
    import model
    class MyApp(wxApp):
        def OnInit(self):
            frame = wxFrame(NULL, -1, "Window tester")
            frame.Show(true)
            frame.SetClientSize( wx.Size(600, 800) )
            pcs = PanelCoverageStats(parent=frame, id=wx.NewId(), pos=wx.Point(0,0), size=wx.Size(200,60), style=0, name="slider")
            stats =  model.experiment.ReflectionStats()
            stats.total = 1234
            stats.measured = 456
            stats.redundant = 123
            pcs.show_reflection_stats(True, stats)
            return true
    app = MyApp(0)
    app.MainLoop()

