"""
created on Mon Apr. 22, 2019

@author: He Sun, Princeton University

helper functions used in wavefront sensing control.

1. Compute_DH(model, dh_shape='wedge', range_r=[2.5, 9], range_angle=30)
	# define the high-contrast planet searching regions (dark holes)
	# compute the indices of pixels in the dark holes

2. Compute_Jacobian(model, dh_ind1, dh_ind2, dV=1.0)
	# compute the Jacobian matrix of the optical system

"""
import numpy as np

def Compute_DH(model, dh_shape='wedge', range_r=[2.5, 9], range_angle=30):
	# define the high-contrast planet searching regions (dark holes)
	# compute the indices of pixels in the dark holes
	wl_c = np.mean(model.wavelength) # center wavelength
	f_lambda_over_D = wl_c * model.focalLength / model.SPwidth
	xs = np.arange(-model.camera_Nxi/2+0.5, model.camera_Nxi/2+0.5) * model.camera_pitch * model.camera_binXi / f_lambda_over_D
	ys = np.arange(-model.camera_Neta/2+0.5, model.camera_Neta/2+0.5) * model.camera_pitch * model.camera_binEta / f_lambda_over_D
	[XS, YS] = np.meshgrid(xs, ys)
	RS = np.sqrt(XS**2 + YS**2)
	TANS = YS / (XS+1e-12)
	if dh_shape.lower() == 'wedge':
		dark_hole = (range_r[0]<=RS) * (RS<=range_r[1]) * (np.abs(TANS) < np.tan(np.deg2rad(range_angle)))
	elif dh_shape.lower() == 'circ':
		dark_hole = (range_r[0]<=RS) * (RS<=range_r[1])
	dh_ind1, dh_ind2 = np.nonzero(dark_hole)
	return dh_ind1, dh_ind2


def Compute_Jacobian(model, dh_ind1, dh_ind2, dV=1.0, print_flag=False):
	# compute the Jacobian matrix of the optical system
	N = int(np.sum(model.DMstop))
	G1 = np.empty((len(dh_ind1), N, len(model.wavelength)), dtype=complex)
	G2 = np.empty((len(dh_ind1), N, len(model.wavelength)), dtype=complex)
	u1 = np.zeros((34, 34))
	u2 = np.zeros((34, 34))
	# Ef0 = model.Propagate(u1, u2)
	# Ef_vector0 = Ef0[dh_ind1, dh_ind2, :]
	# Jacobian of DM1
	for k in range(N):
		if print_flag:
			print(k)
		u1 = np.zeros((34, 34))
		u2 = np.zeros((34, 34))
		u1[model.DMind1[k], model.DMind2[k]] = dV
		Ef_plus = model.Propagate(u1, u2)
		Ef_plus_vector = Ef_plus[dh_ind1, dh_ind2, :]
		Ef_negative = model.Propagate(-u1, u2)
		Ef_negative_vector = Ef_negative[dh_ind1, dh_ind2, :]
		G1[:, k, :] = (Ef_plus_vector - Ef_negative_vector) / (2.*dV)
	# Jacobian of DM2
	for k in range(N):
		if print_flag:
			print(k)
		u1 = np.zeros((34, 34))
		u2 = np.zeros((34, 34))
		u2[model.DMind1[k], model.DMind2[k]] = dV
		Ef_plus = model.Propagate(u1, u2)
		Ef_plus_vector = Ef_plus[dh_ind1, dh_ind2, :]
		Ef_negative = model.Propagate(u1, -u2)
		Ef_negative_vector = Ef_negative[dh_ind1, dh_ind2, :]
		G2[:, k, :] = (Ef_plus_vector - Ef_negative_vector) / (2.*dV)

	return G1, G2