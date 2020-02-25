import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import c

#-------------------------------- (Theta,phi) vs Vph -----------------------------------#

nm = 1e-9

dx = 30 * nm
dy = 30 * nm
dz = 30 * nm

dx2 = 60 * nm
dy2 = 60 * nm
dz2 = 60 * nm

Nx = 100
Ny = 100
Nz = 100
wl = 600 * nm
k0 = 2 * np.pi / wl

S2  = 1./2
S4  = 1./4
S8  = 1./8
S16 = 1./16
S32 = 1./32

dt2  = min(dx,dy,dz) * S2  / c
dt4  = min(dx,dy,dz) * S4  / c
dt8  = min(dx,dy,dz) * S8  / c
dt16 = min(dx,dy,dz) * S16 / c
dt32 = min(dx,dy,dz) * S32 / c

N1 = 500
N2 = 500

N1c = int(N1/2)
N2c = int(N2/2)

theta = np.linspace(0, np.pi/2, N1)
phi   = np.linspace(0, np.pi/2, N2)

dtheta = theta[1] - theta[0]
dphi   = phi  [1] - phi  [0]

Theta, Phi = np.meshgrid(theta, phi)

#Vph_FDTD2  = np.ones((N1,N2), dtype=np.float64)
#Vph_FDTD4  = np.ones((N1,N2), dtype=np.float64)
#Vph_FDTD8  = np.ones((N1,N2), dtype=np.float64)
#Vph_FDTD16 = np.ones((N1,N2), dtype=np.float64)
Vph_FDTD32 = np.ones((N2,N1), dtype=np.float64)

#Vph_PSTD4 = np.ones((N1,N2), dtype=np.float64)
#Vph_PSTD8 = np.ones((N1,N2), dtype=np.float64)
#Vph_PSTD16 = np.ones((N1,N2), dtype=np.float64)
Vph_PSTD32 = np.ones((N2,N1), dtype=np.float64)

#Vph_HPF_4  = np.ones((N1,N2), dtype=np.float64)
#Vph_HPF_8  = np.ones((N1,N2), dtype=np.float64)
#Vph_HPF_16 = np.ones((N1,N2), dtype=np.float64)
Vph_HPF_32 = np.ones((N2,N1), dtype=np.float64)

for i in range(N2):
	for j in range(N1):

		kx = k0 * np.sin(Theta[i,j]) * np.cos(Phi[i,j])
		ky = k0 * np.sin(Theta[i,j]) * np.sin(Phi[i,j])
		kz = k0 * np.cos(Theta[i,j])

		A = (np.sin(kx * dx / 2)/dx)**2
		B = (np.sin(ky * dy / 2)/dy)**2
		C = (np.sin(kz * dz / 2)/dz)**2
		
		#Vph_FDTD2[i,j] = (2./c/k0/dt2) * np.arcsin(c * dt2 * np.sqrt(A+B+C))
		#Vph_FDTD4[i,j] = (2./c/k0/dt4) * np.arcsin(c * dt4 * np.sqrt(A+B+C))
		#Vph_FDTD8[i,j] = (2./c/k0/dt8) * np.arcsin(c * dt8 * np.sqrt(A+B+C))
		Vph_FDTD32[i,j] = (2./c/k0/dt32) * np.arcsin(c * dt32 * np.sqrt(A+B+C))

		A = (kx/2)**2
		B = (ky/2)**2
		C = (kz/2)**2

		#Vph_PSTD4[i,j] = (2./c/k0/dt4) * np.arcsin(c * dt4 * np.sqrt(A+B+C))
		#Vph_PSTD8[i,j] = (2./c/k0/dt8) * np.arcsin(c * dt8 * np.sqrt(A+B+C))
		Vph_PSTD32[i,j] = (2./c/k0/dt32) * np.arcsin(c * dt32 * np.sqrt(A+B+C))
		
		A = (np.sin(kx * dx / 2)/dx)**2
		B = (ky/2)**2
		C = (kz/2)**2

		#Vph_HPF_4[i,j] = (2./c/k0/dt4) * np.arcsin(c * dt4 * np.sqrt(A+B+C))
		#Vph_HPF_8[i,j] = (2./c/k0/dt8) * np.arcsin(c * dt8 * np.sqrt(A+B+C))
		Vph_HPF_32[i,j] = (2./c/k0/dt32) * np.arcsin(c * dt32 * np.sqrt(A+B+C))

#------------------------------------- 3D plot ---------------------------------------------#
Angle_vs_ND = plt.figure(figsize=(12,12))		

ax = Angle_vs_ND.add_subplot(1,1,1, projection='3d')
#ax.plot_surface(Theta, Phi, Vph_FDTD2, color='b', label="FDTD2, S={}" .format(S2))
#ax.plot_surface(Theta, Phi, Vph_FDTD4, color='m', label="FDTD4, S={}" .format(S4))
#ax.plot_surface(Theta, Phi, Vph_PSTD4, color='c', label="PSTD4")
ax.plot_surface(Theta, Phi, Vph_FDTD32, color='b', label="FDTD32")
ax.plot_surface(Theta, Phi, Vph_PSTD32, color='c', label="PSTD32")
ax.plot_surface(Theta, Phi, Vph_HPF_32, color='r', label="HPF32")
ax.plot_surface(Theta, Phi, 1., color='g', label="Ideal")
ax.set_zlim(0.996,1.001)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\phi$")

blue_proxy = plt.Rectangle((0,0), 1, 1, fc='b')
purp_proxy = plt.Rectangle((0,0), 1, 1, fc='m')
cyan_proxy = plt.Rectangle((0,0), 1, 1, fc='c')
yell_proxy = plt.Rectangle((0,0), 1, 1, fc='y')
gree_proxy = plt.Rectangle((0,0), 1, 1, fc='y')
red_proxy  = plt.Rectangle((0,0), 1, 1, fc='r')

ax.legend([blue_proxy, cyan_proxy, red_proxy, gree_proxy], ["FDTD32", "PSTD32", "HPF32", "Ideal"])

Angle_vs_ND.savefig("./Angle_vs_ND.png", format='png', dpi=300)
plt.close('all')

#------------------------------------- 2D plot ---------------------------------------------#

theta_vs_ND  = plt.figure(figsize=(16,10))
theta_vs_ND.suptitle(r"$\lambda_{min}$= 600 nm, $\Delta$t=$\Delta$x/32c" + "\n"								\
						+ r"FDTD: $\Delta$x=$\Delta$y=$\Delta$z=30 nm" + "\n"								\
						+ r"HPF: $\Delta$x=30 nm and $\Delta$y,$\Delta$z < $\frac{\lambda_{min}}{2}$"+"\n"	\
						+ r"PSTD: $\Delta$x,$\Delta$y,$\Delta$z < $\frac{\lambda_{min}}{2}$")

theta_vs_ND.subplots_adjust(top=0.84, wspace=0.4, hspace=0.3)

ax_phi0   = theta_vs_ND.add_subplot(2,3,1)
ax_phi45  = theta_vs_ND.add_subplot(2,3,2)
ax_phi90  = theta_vs_ND.add_subplot(2,3,3)
ax_phi135 = theta_vs_ND.add_subplot(2,3,4)
ax_phi180 = theta_vs_ND.add_subplot(2,3,5)

ax_phi0  .set_title (r"$\phi=0$")
ax_phi45 .set_title (r"$\phi=\frac{\pi}{8}$")
ax_phi90 .set_title (r"$\phi=\frac{\pi}{4}$")
ax_phi135.set_title (r"$\phi=\frac{3\pi}{8}$")
ax_phi180.set_title (r"$\phi=\frac{\pi}{2}$")

ax_phi0  .set_xlabel(r"$\theta$")
#ax_phi45 .set_xlabel(r"$\theta$")
#ax_phi90 .set_xlabel(r"$\theta$")
ax_phi135.set_xlabel(r"$\theta$")
#ax_phi180.set_xlabel(r"$\theta$")

ax_phi0  .set_ylabel(r"$\frac{v_{ph}}{c}$")
#ax_phi45 .set_ylabel(r"$\frac{v_{ph}}{c}$")
#ax_phi90 .set_ylabel(r"$\frac{v_{ph}}{c}$")
ax_phi135.set_ylabel(r"$\frac{v_{ph}}{c}$")
#ax_phi180.set_ylabel(r"$\frac{v_{ph}}{c}$")

ax_phi0  .plot(theta, Vph_FDTD32[0,:], label="FDTD")
ax_phi0  .plot(theta, Vph_PSTD32[0,:], label="PSTD")
ax_phi0  .plot(theta, Vph_HPF_32[0,:], label="HPF")

ax_phi45 .plot(theta, Vph_FDTD32[int(N2/4),:], label="FDTD")
ax_phi45 .plot(theta, Vph_PSTD32[int(N2/4),:], label="PSTD")
ax_phi45 .plot(theta, Vph_HPF_32[int(N2/4),:], label="HPF")

ax_phi90 .plot(theta, Vph_FDTD32[int(N2/2),:], label="FDTD")
ax_phi90 .plot(theta, Vph_PSTD32[int(N2/2),:], label="PSTD")
ax_phi90 .plot(theta, Vph_HPF_32[int(N2/2),:], label="HPF")

ax_phi135.plot(theta, Vph_FDTD32[int(3*N2/4),:], label="FDTD")
ax_phi135.plot(theta, Vph_PSTD32[int(3*N2/4),:], label="PSTD")
ax_phi135.plot(theta, Vph_HPF_32[int(3*N2/4),:], label="HPF")

ax_phi180.plot(theta, Vph_FDTD32[-1,:], label="FDTD")
ax_phi180.plot(theta, Vph_PSTD32[-1,:], label="PSTD")
ax_phi180.plot(theta, Vph_HPF_32[-1,:], label="HPF")

ax_phi0  .set_ylim(0.9955,1.0005)
ax_phi45 .set_ylim(0.9955,1.0005)
ax_phi90 .set_ylim(0.9955,1.0005)
ax_phi135.set_ylim(0.9955,1.0005)
ax_phi180.set_ylim(0.9955,1.0005)

ax_phi0  .legend(loc='best')
ax_phi45 .legend(loc='best')
ax_phi90 .legend(loc='best')
ax_phi135.legend(loc='best')
ax_phi180.legend(loc='best')

ax_phi0  .grid(True)
ax_phi45 .grid(True)
ax_phi90 .grid(True)
ax_phi135.grid(True)
ax_phi180.grid(True)

theta_vs_ND.savefig("./theta_vs_ND.eps", format='eps', dpi=300)
plt.close(theta_vs_ND)

#-------------- Nx vs Vph -----------------#

nm = 1e-9

start = 2
end   = 21

Nx = np.arange(start,end, dtype=int)
Ny = np.arange(start,end, dtype=int)
Nz = np.arange(start,end, dtype=int)

Nx_PSTD = np.ones(len(Nx)) * 2

S = 1./32 # courant number.

Vph_REAL_Norm   = 1.
#Vph_HPF__Norm_x_axis   = (Nx/S/np.pi) * np.arcsin(S * np.sin(np.pi/Nx))
#Vph_HPF__Norm_yz_axis  = (Ny/S/np.pi) * np.arcsin(S * np.pi/Ny)
#Vph_PSTD_Norm_xyz_axis = (Nx/S/np.pi) * np.arcsin(S * np.pi/Nx)
#Vph_HPF__Norm_xy_diagonal = (Nx/S/np.pi) * np.arcsin(S * np.sqrt(np.sin(np.pi/np.sqrt(2)/Nx)**2 + (np.pi/Nx)**2/2))
#Vph_PSTD_Norm_xy_diagonal = (Nx/S/np.pi) * np.arcsin(S * np.pi/Nx)

Vph_FDTD_Norm_xyz_axis = (Nx/S/np.pi) * np.arcsin(S * np.sin(np.pi/Nx))
Vph_FDTD_Norm_trigonal = (Nx/S/np.pi) * np.arcsin(S * np.sqrt((2*np.sin(np.pi/Nx/2)**2) + np.sin(np.pi/np.sqrt(2)/Nx)**2))
Vph_HPF__Norm_trigonal = (Nx/S/np.pi) * np.arcsin(S * np.sqrt(np.sin(np.pi/Nx/2)**2 + ((3*(np.pi**2))/(4*Nx**2))))
#Vph_PSTD_Norm_trigonal = (Nx/S/np.pi) * np.arcsin(S * np.pi/Nx)
Vph_PSTD_Norm_xyz_axis = (Nx_PSTD/S/np.pi) * np.arcsin(S * np.pi/Nx_PSTD)

Vph_FDTD_Norm_xyz_axis_Err = abs((Nx/S/np.pi) * np.arcsin(S * np.sin(np.pi/Nx)) - 1)
Vph_FDTD_Norm_trigonal_Err = abs((Nx/S/np.pi) * np.arcsin(S * np.sqrt((2*np.sin(np.pi/Nx/2)**2) + np.sin(np.pi/np.sqrt(2)/Nx)**2)) - 1)
Vph_HPF__Norm_trigonal_Err = abs((Nx/S/np.pi) * np.arcsin(S * np.sqrt(np.sin(np.pi/Nx/2)**2 + ((3*(np.pi**2))/(4*Nx**2)))) - 1)
#Vph_PSTD_Norm_trigonal_Err = abs((Nx/S/np.pi) * np.arcsin(S * np.pi/Nx) - 1)
Vph_PSTD_Norm_xyz_axis_Err = abs((Nx_PSTD/S/np.pi) * np.arcsin(S * np.pi/Nx_PSTD) - 1)

NG_vs_ND = plt.figure(figsize=(16,7))

NG_vs_ND.subplots_adjust(wspace=0.4)
ax1 = NG_vs_ND.add_subplot(1,2,1)

ax1.plot(Nx, Vph_FDTD_Norm_trigonal, label=r"FDTD: $\phi,\theta=45^\circ$")
ax1.plot(Nx, Vph_HPF__Norm_trigonal, label=r"HPF : $\phi,\theta=45^\circ$")
ax1.plot(Nx, Vph_FDTD_Norm_xyz_axis, label="FDTD: all-axis / HPF: x-axis")
ax1.plot(Nx, Vph_PSTD_Norm_xyz_axis, label="PSTD: all axis / HPF: y,z-axis", color='k')
ax1.set_title("Normalized Dispersion relation")
ax1.set_xlabel("Number of grids per wavelength")
ax1.set_xlim(start, end)
ax1.set_ylabel(r"$v_{ph}$")
ax1.grid(True)
ax1.legend(loc='lower right')

ax2 = NG_vs_ND.add_subplot(1,2,2)
ax2.plot(Nx[12:17], Vph_FDTD_Norm_trigonal_Err[12:17], label=r"FDTD: $\phi,\theta=45^\circ$")
ax2.plot(Nx[12:17], Vph_HPF__Norm_trigonal_Err[12:17], label=r"HPF : $\phi,\theta=45^\circ$")
ax2.plot(Nx[12:17], Vph_FDTD_Norm_xyz_axis_Err[12:17], label="FDTD: all axes / HPF: x-axis")
ax2.plot(Nx[12:17], Vph_PSTD_Norm_xyz_axis_Err[12:17], label="PSTD: all axes / HPF: y,z-axis", color='k')
ax2.set_title("Error of Normalized Dispersion relation")
ax2.set_xlabel("Number of grids per wavelength")
#ax2.set_xlim(start, end)
ax2.set_ylabel("Normalized Error")
ax2.grid(False)
ax2.legend(loc='upper right')

NG_vs_ND.savefig("./NumGRID_vs_Vph.eps", format='eps', dpi=300)

plt.close('all')


#------------------------ Checking Difference with respect to 'c' ------------------------------#
#Error_Norm_FDTD2 = 0
#Error_Norm_FDTD4 = 0
#Error_Norm_FDTD8 = 0
#Error_Norm_PSTD4 = 0
#Error_Norm_PSTD8 = 0
#Error_Norm_HPF_4 = 0
#Error_Norm_HPF_8 = 0
Error_Norm_FDTD32 = 0
Error_Norm_PSTD32 = 0
Error_Norm_HPF_32 = 0

#----------------------------#
#max_FDTD2 = np.amax(Vph_FDTD2)
#max_FDTD4 = np.amax(Vph_FDTD4)
#max_FDTD8 = np.amax(Vph_FDTD2)
#max_PSTD4 = np.amax(Vph_PSTD4)
#max_PSTD8 = np.amax(Vph_PSTD8)
#max_HPF_4 = np.amax(Vph_HPF_4)
#max_HPF_8 = np.amax(Vph_HPF_8)

max_FDTD32 = np.amax(Vph_FDTD32)
max_PSTD32 = np.amax(Vph_PSTD32)
max_HPF_32 = np.amax(Vph_HPF_32)

#----------------------------#

#min_FDTD2 = np.amin(Vph_FDTD2)
#min_FDTD4 = np.amin(Vph_FDTD4)
#min_FDTD8 = np.amin(Vph_FDTD2)
#min_PSTD4 = np.amin(Vph_PSTD4)
#min_PSTD8 = np.amin(Vph_PSTD8)
#min_HPF_4 = np.amin(Vph_HPF_4)
#min_HPF_8 = np.amin(Vph_HPF_8)
min_FDTD32 = np.amin(Vph_FDTD32)
min_PSTD32 = np.amin(Vph_PSTD32)
min_HPF_32 = np.amin(Vph_HPF_32)

#----------------------------#
for i, angle1 in enumerate(phi):
	for j, angle2 in enumerate(theta):
		
		#-------- This Error shows how much the value of Vph is far from light speed c ----------#
		#Error_Norm_FDTD2 += abs(Vph_FDTD2[i,j]-1) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_FDTD4 += abs(Vph_FDTD4[i,j]-1) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_FDTD8 += abs(Vph_FDTD8[i,j]-1) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_PSTD4 += abs(Vph_PSTD4[i,j]-1) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_PSTD8 += abs(Vph_PSTD8[i,j]-1) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_HPF_4 += abs(Vph_HPF_4[i,j]-1) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_HPF_8 += abs(Vph_HPF_8[i,j]-1) * np.sin(theta[i]) * dtheta * dphi
		Error_Norm_FDTD32 += abs(Vph_FDTD32[i,j]-1) * np.sin(theta[j]) * dtheta * dphi
		Error_Norm_PSTD32 += abs(Vph_PSTD32[i,j]-1) * np.sin(theta[j]) * dtheta * dphi
		Error_Norm_HPF_32 += abs(Vph_HPF_32[i,j]-1) * np.sin(theta[j]) * dtheta * dphi

		#-------- This Error shows how much the shape of sphere is far from perfect shpere ----------#
		#Error_Norm_FDTD2 += abs(Vph_FDTD2[i,j]-min_FDTD2) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_FDTD4 += abs(Vph_FDTD4[i,j]-min_FDTD4) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_FDTD8 += abs(Vph_FDTD8[i,j]-min_FDTD8) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_PSTD4 += abs(Vph_PSTD4[i,j]-min_PSTD4) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_PSTD8 += abs(Vph_PSTD8[i,j]-min_PSTD8) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_HPF_4 += abs(Vph_HPF_4[i,j]-min_HPF_4) * np.sin(theta[i]) * dtheta * dphi
		#Error_Norm_HPF_8 += abs(Vph_HPF_8[i,j]-min_HPF_8) * np.sin(theta[i]) * dtheta * dphi

#Error_Norm_FDTD2 = abs(Error_Norm_FDTD2) * 100 * 2 / np.pi
#Error_Norm_FDTD4 = abs(Error_Norm_FDTD4) * 100 * 2 / np.pi
#Error_Norm_FDTD4 = abs(Error_Norm_FDTD8) * 100 * 2 / np.pi
#Error_Norm_PSTD4 = abs(Error_Norm_PSTD4) * 100 * 2 / np.pi
#Error_Norm_PSTD8 = abs(Error_Norm_PSTD8) * 100 * 2 / np.pi
#Error_Norm_HPF_4 = abs(Error_Norm_HPF_4) * 100 * 2 / np.pi
#Error_Norm_HPF_8 = abs(Error_Norm_HPF_8) * 100 * 2 / np.pi
Error_Norm_FDTD32 = abs(Error_Norm_FDTD32) * 100 * 2 / np.pi
Error_Norm_PSTD32 = abs(Error_Norm_PSTD32) * 100 * 2 / np.pi
Error_Norm_HPF_32 = abs(Error_Norm_HPF_32) * 100 * 2 / np.pi

#print(Error_Norm_FDTD2)
#print(Error_Norm_FDTD4)
#print(Error_Norm_PSTD4)
print(Error_Norm_FDTD32)
print(Error_Norm_PSTD32)
print(Error_Norm_HPF_32)

plt.close()
