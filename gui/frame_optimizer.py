#Boa:Frame:FrameOptimizer

import wx

def create(parent):
    return FrameOptimizer(parent)

[wxID_FRAMEOPTIMIZER, wxID_FRAMEOPTIMIZERBUTTONADDORIENTATION, 
 wxID_FRAMEOPTIMIZERSTATICLINE1, wxID_FRAMEOPTIMIZERSTATICTEXT1, 
 wxID_FRAMEOPTIMIZERSTATICTEXTRESULTS, wxID_FRAMEOPTIMIZERTEXTSTATUS, 
] = [wx.NewId() for _init_ctrls in range(6)]

class FrameOptimizer(wx.Frame):
    def _init_coll_boxSizer1_Items(self, parent):
        # generated method, don't edit

        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticText1, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticLine1, 0, border=0, flag=wx.EXPAND)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.staticTextResults, 0, border=0, flag=0)
        parent.AddSpacer(wx.Size(8, 8), border=0, flag=0)
        parent.AddWindow(self.textStatus, 1, border=0, flag=wx.EXPAND)
        parent.AddWindow(self.buttonAddOrientation, 0, border=0, flag=0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizer1 = wx.BoxSizer(orient=wx.VERTICAL)

        self._init_coll_boxSizer1_Items(self.boxSizer1)

        self.SetSizer(self.boxSizer1)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_FRAMEOPTIMIZER, name=u'FrameOptimizer',
              parent=prnt, pos=wx.Point(649, 412), size=wx.Size(415, 594),
              style=wx.DEFAULT_FRAME_STYLE, title=u'Optimizer')
        self.SetClientSize(wx.Size(415, 594))

        self.staticText1 = wx.StaticText(id=wxID_FRAMEOPTIMIZERSTATICTEXT1,
              label=u'Optimization Parameters:', name='staticText1',
              parent=self, pos=wx.Point(0, 8), size=wx.Size(167, 17), style=0)

        self.staticLine1 = wx.StaticLine(id=wxID_FRAMEOPTIMIZERSTATICLINE1,
              name='staticLine1', parent=self, pos=wx.Point(0, 33),
              size=wx.Size(415, 2), style=0)

        self.staticTextResults = wx.StaticText(id=wxID_FRAMEOPTIMIZERSTATICTEXTRESULTS,
              label=u'Current Status:', name=u'staticTextResults', parent=self,
              pos=wx.Point(0, 43), size=wx.Size(98, 17), style=0)

        self.textStatus = wx.TextCtrl(id=wxID_FRAMEOPTIMIZERTEXTSTATUS,
              name=u'textStatus', parent=self, pos=wx.Point(0, 68),
              size=wx.Size(415, 497), style=0, value=u' ')

        self.buttonAddOrientation = wx.Button(id=wxID_FRAMEOPTIMIZERBUTTONADDORIENTATION,
              label=u'buttonAddOrientation', name=u'buttonAddOrientation',
              parent=self, pos=wx.Point(0, 565), size=wx.Size(85, 29), style=0)

        self._init_sizers()

    def __init__(self, parent):
        self._init_ctrls(parent)
