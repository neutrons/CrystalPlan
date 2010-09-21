""" Make an Ewald sphere figure """
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

from pylab import *

clf()
# Fill up the axes
axes([0,0,1,1])
theta = arange(0, 2*pi, 0.01)

i1 = 100
i2 = 150
det = range(i1,i2)

cx = cos(theta)
cy = sin(theta)

lam_min = 1.0
L1 = 1/lam_min
lam_max = 2.5
L2 = 1/lam_max

cx1 = cx/lam_min + L1
cy1 = cy/lam_min
cx2 = cx/lam_max + L2
cy2 = cy/lam_max

plot( cx1, cy1, '-g')
plot( cx2, cy2, '-b')
axis('equal')

plot( cx1[det], cy1[det], '-r', linewidth=6)
plot( cx2[det], cy2[det], '-r', linewidth=6)

plot( 0, 0, 'k.', markersize=10)
annotate(' (000)', (0,0) , va='center')

#Volume wedge
num = i1
plot( [cx2[num], cx1[num]], [cy2[num], cy1[num]], '-r', linewidth=2)
num = i2
plot( [cx2[num], cx1[num]], [cy2[num], cy1[num]], '-r', linewidth=2)

annotate('$q_{min}$', xy=(cx2[num], cy2[num]), xytext=(0.3, 0.5),
            arrowprops=dict(facecolor='black', arrowstyle='->') )

annotate('$q_{max}$', xy=(cx1[num], cy1[num]), xytext=(0.8, 1.1),
            arrowprops=dict(facecolor='black', arrowstyle='->') )

r = 0.125
plot( cx[det]*r + L2, cy[det]*r, ':k')
r = 0.25
plot( cx[det]*r + L1, cy[det]*r, ':k')
annotate('angular coverage \nof detector', xy=(1.1,0.3), xytext=(1.2, 0.15),
            arrowprops=dict(facecolor='black', arrowstyle='->') )


#Detector wedge
annotate('S2', (L1,0) , va='top', ha='center')
annotate('S1', (L2,0) , va='top', ha='center')
plot( [L2,cx2[i1]], [0,cy2[i1]], ':k')
plot( [L2,cx2[i2]], [0,cy2[i2]], ':k')
plot( [L1,cx1[i1]], [0,cy1[i1]], ':k')
plot( [L1,cx1[i2]], [0,cy1[i2]], ':k')

annotate('$1/\lambda_{max}$ sphere', (L2,-L2*0.9), ha='center', va='bottom' )

annotate('$1/\lambda_{min}$ sphere  ', (2*L1,0), ha='right' )

x = (mean( cx1[det]) + mean(cx2[det] ))/2
y = (mean( cy1[det]) + mean(cy2[det] ))/2
print x, y
annotate('volume\nmeasured', (x,y), ha='center', va='center' )

ylim( (-0.4, 1.2) )
xticks( [])
yticks( [])

savefig("ewald_fig.png", dpi=300)
savefig("ewald_fig.pdf")
savefig("ewald_fig.ps")

