"""
created on Mon Apr. 23, 2019

@author: He Sun, Princeton University

wavefront estimators, including least-square batch process estimator (BPE), 
Kalman filter, extended Kalman filter.

"""
import numpy as np
from . import filters

class Batch_process:
	def __init__(self, params_values):
		self.params_values = params_values
		self.est_type = 'batch process estimator'

	def Estimate(self, If_p, u1p, u2p, exp_time):
		n_pair = u1p.shape[1]
		G1 = self.params_values['G1']
		G2 = self.params_values['G2']
		G = np.concatenate((G1, G2), axis=1)
		dI = np.empty((If_p.shape[0], If_p.shape[1], n_pair), dtype=float)
		for k in range(n_pair):
			dI[:, :, k] = If_p[:, :, 2*k] - If_p[:, :, 2*k+1]
		E_est = np.empty((If_p.shape[0], If_p.shape[1]), dtype=complex)
		P_est = np.empty((If_p.shape[0], 2, 2), dtype=float)
		for k in range(G1.shape[2]):
			G1m = G1[:, :, k]
			G2m = G2[:, :, k]
			dE = np.matmul(G1m, u1p) + np.matmul(G2m, u2p)
			contrast_p = np.mean(If_p[:, k, :], 0)
			d_contrast_p_half = np.mean(np.abs(dE)**2, 0)
			d_contrast_p = np.empty(contrast_p.shape)
			d_contrast_p[0::2] = d_contrast_p_half
			d_contrast_p[1::2] = d_contrast_p_half
			cov_p = self.params_values['R0'] / exp_time**2 + \
					self.params_values['R1'] / exp_time * contrast_p + \
					4*(self.params_values['Q0'] + self.params_values['Q1']*d_contrast_p) * contrast_p
			R = np.diag(cov_p[0::2] + cov_p[1::2])
			for i in range(G.shape[0]):
				H = np.empty((dE.shape[1], 2), dtype=float)
				for l in range(dE.shape[1]):
					H[l, :] = 4 * np.array([dE[i, l].real, dE[i, l].imag])
				y = dI[i, k , :]
				x_hat, P_est_now = filters.lse(y, H, R)
				E_est[i, k] = x_hat[0] + 1j * x_hat[1]
				P_est[i, :, :] = P_est_now
		return E_est, P_est


class Kalman_filter:
	def __init__(self, params_values):
		self.params_values = params_values
		self.est_type = 'Kalman filter'

	def Estimate(self, If_p, u1p, u2p, Enp_old, P_old, u1c, u2c, exp_time):
		n_pair = u1p.shape[1]
		G1 = self.params_values['G1']
		G2 = self.params_values['G2']
		G = np.concatenate((G1, G2), axis=1)
		uc = np.concatenate([u1c, u2c])
		dI = np.empty((If_p.shape[0], If_p.shape[1], n_pair), dtype=float)
		for k in range(n_pair):
			dI[:, :, k] = If_p[:, :, 2*k] - If_p[:, :, 2*k+1]
		E_est = np.empty((If_p.shape[0], If_p.shape[1]), dtype=complex)
		P_est = np.empty((If_p.shape[0], 2, 2), dtype=float)
		for k in range(G1.shape[2]):
			G1m = G1[:, :, k]
			G2m = G2[:, :, k]
			Gm = G[:, :, k]
			dEc = np.matmul(G1m, u1c) + np.matmul(G2m, u2c)
			dE = np.matmul(G1m, u1p) + np.matmul(G2m, u2p)
			# Q = (self.params_values['Q1'] * np.mean(np.abs(dE)**2) + \
			# 	self.params_values['Q0']) * np.eye(2) # this is a serious error
			Q = (self.params_values['Q1'] * np.mean(np.abs(dEc)**2) + \
				self.params_values['Q0']) * np.eye(2)
			contrast_p = np.mean(If_p[:, k, :], 0)
			d_contrast_p_half = np.mean(np.abs(dE)**2, 0)
			d_contrast_p = np.empty(contrast_p.shape)
			d_contrast_p[0::2] = d_contrast_p_half
			d_contrast_p[1::2] = d_contrast_p_half
			cov_p = self.params_values['R0'] / exp_time**2 + \
					self.params_values['R1'] / exp_time * contrast_p + \
					4*(self.params_values['Q0'] + self.params_values['Q1']*d_contrast_p) * contrast_p
			R = np.diag(cov_p[0::2] + cov_p[1::2])
			for i in range(G.shape[0]):
				H = np.empty((dE.shape[1], 2), dtype=float)
				for l in range(dE.shape[1]):
					H[l, :] = 4 * np.array([dE[i, l].real, dE[i, l].imag])
				G_now = np.concatenate([Gm[i, :].real.reshape((1, -1)), 
										Gm[i, :].imag.reshape((1, -1))], 0)
				y = dI[i, k , :]
				x_old = np.array([Enp_old[i].real, Enp_old[i].imag])
				P_est_old = P_old[i, 0:2, 0:2]
				x_hat, P_est_now, _, _ = filters.Kalman_filter(y, uc, x_old, P_est_old, 
														np.eye(2), G_now, H, Q, R)
				E_est[i, k] = x_hat[0] + 1j * x_hat[1]
				P_est[i, :, :] = P_est_now
		return E_est, P_est


class Extended_Kalman_filter:
	def __init__(self, params_values):
		self.params_values = params_values
		self.est_type = 'Extended Kalman filter'

	def Estimate(self, If_p, u1p, u2p, Enp_old, Iinco_old, P_old, u1c, u2c, exp_time):
		n_images = u1p.shape[1]
		G1 = self.params_values['G1']
		G2 = self.params_values['G2']
		G = np.concatenate((G1, G2), axis=1)
		uc = np.concatenate([u1c, u2c])
		E_est = np.empty((If_p.shape[0], If_p.shape[1]), dtype=complex)
		Iinco_est = np.empty((If_p.shape[0], If_p.shape[1]), dtype=float)
		P_est = np.empty((If_p.shape[0], 3, 3), dtype=float)
		for k in range(G1.shape[2]):
			G1m = G1[:, :, k]
			G2m = G2[:, :, k]
			Gm = G[:, :, k]
			dEc = np.matmul(G1m, u1c) + np.matmul(G2m, u2c)
			dE = np.matmul(G1m, u1p) + np.matmul(G2m, u2p)
			# Qco = (self.params_values['Q1'] * np.mean(np.abs(dE)**2) + \
			# 	self.params_values['Q0']) * np.eye(2) # this is a serious error
			Qco = (self.params_values['Q1'] * np.mean(np.abs(dEc)**2) + \
				self.params_values['Q0']) * np.eye(2)
			Q = np.zeros((3, 3))
			Q[0:2, 0:2] = Qco
			Q[2, 2] = 1e-22
			contrast_p = np.mean(If_p[:, k, :], 0)
			d_contrast_p = np.mean(np.abs(dE)**2, 0)
			cov_p = self.params_values['R0'] / exp_time**2 + \
					self.params_values['R1'] / exp_time * contrast_p + \
					4*(self.params_values['Q0'] + self.params_values['Q1']*d_contrast_p) * contrast_p
			R = np.diag(cov_p)
			for i in range(G.shape[0]):
				# Q[2, 2] = np.max([0.1 * Iinco_old[i], 1e-20])
				# H = np.empty((dE.shape[1], 2), dtype=float)
				# for l in range(dE.shape[1]):
				# 	H[l, :] = 4 * np.array([dE[i, l].real, dE[i, l].imag])
				G_now = np.concatenate([Gm[i, :].real.reshape((1, -1)), 
										Gm[i, :].real.reshape((1, -1))], 0)
				y = If_p[i, k , :] - 2*(self.params_values['Q1']*d_contrast_p+self.params_values['Q0'])
				x_old = np.array([Enp_old[i].real, Enp_old[i].imag, Iinco_old[i]])
				d_xc = np.array([dEc[i].real, dEc[i].imag, 0.]).reshape((-1, 1))
				d_xp = np.concatenate([dE[i].real.reshape((n_images, 1)), 
									dE[i].imag.reshape((n_images, 1)), 
									np.zeros((n_images, 1))], axis=1)
				if P_old.shape[-1] == 3:
					P_est_old = P_old[i, :, :]
				else:
					P_est_old = np.zeros((3, 3))
					P_est_old[0:2, 0:2] = P_old[i, :, :]
					P_est_old[2, 2] = 1e-20
				x_hat, P_est_now, _, _ = filters.IEKF(y, d_xc, d_xp, x_old, P_est_old, Q, R, 10)
				E_est[i, k] = x_hat[0] + 1j * x_hat[1]
				Iinco_est[i, k] = x_hat[2]
				P_est[i, :, :] = P_est_now
		return E_est, Iinco_est, P_est


class Extended_Kalman_filter_DH_maintain:
	def __init__(self, params_values):
		self.params_values = params_values
		self.est_type = 'Extended Kalman filter dark hole maintain'

	def Estimate(self, If, Enp_old, Iinco_old, P_old, u1c, u2c, exp_time):
		G1 = self.params_values['G1']
		G2 = self.params_values['G2']
		G = np.concatenate((G1, G2), axis=1)
		uc = np.concatenate([u1c, u2c])
		E_est = np.empty((If.shape[0], If.shape[1]), dtype=complex)
		Iinco_est = np.empty((If.shape[0], If.shape[1]), dtype=float)
		P_est = np.empty((If.shape[0], 3, 3), dtype=float)
		for k in range(G1.shape[2]):
			G1m = G1[:, :, k]
			G2m = G2[:, :, k]
			Gm = G[:, :, k]
			dEc = np.matmul(G1m, u1c) + np.matmul(G2m, u2c)
			Qco = (self.params_values['Q1'] * np.mean(np.abs(dEc)**2) + \
				self.params_values['Q0']) * np.eye(2)
			Q = np.zeros((3, 3))
			Q[0:2, 0:2] = Qco
			Q[2, 2] = 1e-22
			contrast = np.mean(If[:, k], 0)
			# R = (self.params_values['R0'] / exp_time**2 + \
			# 	self.params_values['R1'] / exp_time * contrast) * np.ones([1, 1])
			for i in range(G.shape[0]):
				R = (self.params_values['R0'] / exp_time**2 + \
				self.params_values['R1'] / exp_time * np.max([np.abs(If[i, 0]), 1e-10])) * np.ones([1, 1])
				# Q[2, 2] = np.max([0.1 * Iinco_old[i], 1e-20])
				G_now = np.concatenate([Gm[i, :].real.reshape((1, -1)), 
										Gm[i, :].real.reshape((1, -1))], 0)
				y = If[i, k]
				x_old = np.array([Enp_old[i].real, Enp_old[i].imag, Iinco_old[i]])
				d_xc = np.array([dEc[i].real, dEc[i].imag, 0.]).reshape((-1, 1))

				if P_old.shape[-1] == 3:
					P_est_old = P_old[i, :, :]
				else:
					P_est_old = np.zeros((3, 3))
					P_est_old[0:2, 0:2] = P_old[i, :, :]
					P_est_old[2, 2] = 1e-20
				x_hat, P_est_now, _, _ = filters.IEKF_DH(y, d_xc, x_old, P_est_old, Q, R, 5)
				E_est[i, k] = x_hat[0] + 1j * x_hat[1]
				Iinco_est[i, k] = x_hat[2]
				P_est[i, :, :] = P_est_now
		return E_est, Iinco_est, P_est
