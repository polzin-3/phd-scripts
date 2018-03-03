import numpy as np
import matplotlib.pyplot as plt

def C_XYZ(x,y,z,q):
    return (2./(1+q))*1./(x**2+y**2+z**2)**.5 + (2*q/(1+q))*1./((x-1)**2+y**2+z**2)**.5 + (x-q/(1+q))**2 + y**2

def Circle(x,y):
    return x**2 + y**2 - (1.28858/1.33)**2

q=0.045/1.4 # mass ratio

# Plot potential along line of centres
# See two clear potential wells at origin (PSR) and x=1 (companion)
x=np.linspace(-1.5,1.5,1000)
plt.plot(x,-C_XYZ(x,0,0,q))
# L1 point can be seen as the maxima between 0<x<1
# Value of C_XYZ at L1 is required to find full equipotential surface
C_L1 = 3.32465

fig1 = plt.figure(2, figsize=(12,5.5), frameon=False)
#-------------------------- Side view --------------------------#
# Plot Roche Lobe
xx = np.linspace(-0.75, 1.25, 1000)
zz = np.linspace(-0.75, 0.75, 1000)
X,Z = np.meshgrid(xx, zz)
F = C_XYZ(X,0,Z,q) - C_L1
ax1 = fig1.add_subplot(121, xlim=[-.75,1.25], ylim=[-1,1], aspect='equal')
ax1.set_axis_off()
ax1.set_title('Side view')
plt.contour(X,Z,F,0)                      # Roche lobe
plt.plot(([0,1]), ([0,0]), 'x')           # Star centre markers
plt.plot(([0,1]), ([0,np.tan(40.*np.pi/180)]), '--')   # LoS
plt.vlines(0, 0, 0.9, color='k')          # Vertical for angle part 1
plt.plot(([0]), ([0.9]), '^', color='k')  # Vertical for angle part 2
plt.vlines(0, -.6, -.75, color='k')       # Vertical for separation
plt.vlines(1, -.15, -.75, color='k')      # Vertical for separation
plt.hlines(-.7, 0, 1, color='k')          # Horizontal for separation
plt.plot(([0.015]), ([-.7]), '<', color='k')  # Separation arrow
plt.plot(([.985]), ([-.7]), '>', color='k')   # Separation arrow
plt.plot(([-0.75,1.25]), ([0,0]), '--', color='0.5')  # Orbital plane
x=np.linspace(0, 0.15, 100)               # Angle curve 1
plt.plot(x, ((0.15/np.cos(40*np.pi/180.))**2 - x**2)**.5, color='g')  # Angle curve 2
# Annotations
plt.annotate('PSR', xy=(0,0), xytext=(-.05,-.08))
plt.annotate('Companion', xy=(1,0), xytext=(.76,0.16))
plt.annotate(r'$a$', xy=(.5,-.7), xytext=(.5,-.76))
plt.annotate(r'$i$', xy=(.1,.2), xytext=(.1,.2), color='g')
plt.annotate('LoS', xy=(1,.8), xytext=(1.02,.85), color='g')
plt.annotate('Orbital', xy=(-.6,0), xytext=(-.6,.02), color='.5')
plt.annotate('plane', xy=(-.6,-.06), xytext=(-.6,-.08), color='.5')
plt.annotate('Roche', xy=(-.7,-.5), xytext=(-.64,-.53), color='b')
plt.annotate('lobe', xy=(-.7,-.55), xytext=(-.64,-.61), color='b')

#-------------------------- Top view --------------------------#
xx = np.linspace(-2, 1.17, 1000)
yy = np.linspace(-.73, .73, 1000)
yy_2 = np.linspace(-2, 2, 1000)
X,Y = np.meshgrid(xx, yy)
X,Y_2 = np.meshgrid(xx, yy_2)
F = C_XYZ(X, Y, 0, q) - C_L1
F2 = Circle(X+1-1.28858/1.33, Y_2)
ax = fig1.add_subplot(122, xlim=[-1.05,1.05], ylim=[-1.45,.65], aspect='equal')
ax.set_axis_off()
ax.set_title('Top view')
#plt.pie(([.092,.02]),startangle=360*.718,radius=1.4/1.33,colors=['red','pink'],center=(0,1.28858/1.33-1))
#plt.pie(([.13,.03]),startangle=360*.7,radius=1.288/1.33,colors=['green','lightgreen'],center=(0,1.28858/1.33-1))
#wedges,text=plt.pie(([1.]),radius=1.2/1.33,colors=['white'],center=(0,1.28858/1.33-1))
#for w in wedges:
#    w.set_linewidth(0)
plt.contour(Y_2, X, F2, 0, colors='.5')   # Plot orbit
plt.plot(([0]), ([0]), 'x')
plt.plot(([0]), ([-1]), '<', color='.5')                         #
plt.plot(([-1.28858/1.33]), ([1.28858/1.33-1]), '^', color='.5') # Plot arrows
plt.plot(([1.28858/1.33]), ([1.28858/1.33-1]), 'v', color='.5')  #
plt.plot(([0]), ([1.28858/1.33-1]), 'o', color='k')
plt.contour(Y, -X, F, 0)                  # Plot Roche lobe
plt.plot(([0,-1.3*.3/.95]), ([0,-1.3]), '--', color='g')
plt.plot(([0,1.3*.468/.88]), ([0,-1.3]), '--', color='g')
plt.plot(([0,1.3*.618/.779]), ([0,-1.3]), '--', color='g')
plt.hlines(-1.28, -1.28*.3/.95, 1.28*.618/.779, color='g')
plt.plot(([-1.25*.3/.95,.7]), ([-1.28,-1.28]), '<', color='g')
plt.plot(([1.25*.618/.779,.66]), ([-1.28,-1.28]), '>', color='g')
plt.plot(([0,-1.2*.19/.98]), ([0,-1.2]), '--', color='r')
plt.plot(([0,1.2*.356/.93]), ([0,-1.2]), '--', color='r')
plt.plot(([0,1.24*.468/.88]), ([0,-1.2]), '--', color='r')
plt.hlines(-1.18, -1.18*.19/.98, 1.22*.468/.88, color='r')
plt.plot(([-1.15*.19/.98,.47]), ([-1.18,-1.18]), '<', color='r')
plt.plot(([1.17*.468/.88,.43]), ([-1.18,-1.18]), '>', color='r')
plt.annotate('PSR', xy=(0,0), xytext=(-0.06,.03))
plt.annotate('Barycentre', xy=(0,0), xytext=(0.03,1.28858/1.33-1.02))
plt.annotate('Companion', xy=(-.6,-.95), xytext=(-.89,-.95), color='.5')
plt.annotate('orbit', xy=(-.6,-1), xytext=(-.89,-1.03), color='.5')
plt.annotate('Roche', xy=(-.6,-.55), xytext=(-.75,-.55), color='b')
plt.annotate('lobe', xy=(-.7,-.6), xytext=(-.75,-.63), color='b')
plt.annotate('345MHz ecl', xy=(-0,-1.24), xytext=(-.12,-1.26), color='r')
plt.annotate(r'$\Delta$DM', xy=(.23,-1.23), xytext=(.45,-1.26), color='r')
plt.annotate('149MHz ecl', xy=(-0,-1.34), xytext=(-.12,-1.36), color='g')
plt.annotate(r'$\Delta$DM', xy=(.36,-1.33), xytext=(.75,-1.36), color='g')
plt.show()
