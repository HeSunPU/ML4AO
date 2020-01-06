import scipy.io as sio
import scipy as sp
import numpy as np
from FPWCpy import estimation as est
from FPWCpy import sensing
from FPWCpy import detector
from FPWCpy import helper_function as hp
from FPWCpy.optical_layout import SPLC_model as splc
from FPWCpy.optical_layout import VORTEX_model as vortex
import matplotlib.pyplot as plt
import warnings

import EMsystemID as em

warnings.filterwarnings("ignore")
plt.ion()


def EFC(x, G, weight, alpha):
	M = np.zeros((G.shape[1], G.shape[1]))
	Gx = np.zeros((G.shape[1], 1))
	for k in range(len(weight)):
		Gx += weight[k] * np.real(np.matmul(np.conj(G[:, :, k].T), x[:, k].reshape((-1, 1))))
		M += weight[k] * np.real(np.matmul(np.conj(G[:, :, k].T), G[:, :, k]))
	command = -np.real(np.matmul(np.linalg.inv(M + alpha*np.eye(G.shape[1])), Gx.reshape((-1, 1))))
	return command.reshape(G.shape[1])

def params_set(G1, G2, Q0, Q1, R0, R1):
	params_values = {}
	params_values['G1'] = G1
	params_values['G2'] = G2
	params_values['Q0'] = Q0
	params_values['Q1'] = Q1
	params_values['R0'] = R0
	params_values['R1'] = R1
	return params_values



if __name__ == "__main__":
	#############################################################################
	# define the optical model and the corresponding state space model
	#############################################################################
	wavelength = 635e-9 * np.ones(1)
	DM1gain = 5.06e-9 * np.ones((34, 34))
	DM2gain = 6.27e-9 * np.ones((34, 34))
	coronagraph_type = 'splc'#'vortex'#

	model = splc.Optical_Model(wavelength, DM1gain, DM2gain, wfe=True)
	model_perfect = splc.Optical_Model(wavelength, DM1gain, DM2gain, wfe=False)

	# define the dark hole region
	dh_ind1, dh_ind2 = hp.Compute_DH(model, dh_shape='wedge', range_r=[2.5, 9], range_angle=30)

	# compute or load the Jacobian matrices
	# G1, G2 = hp.Compute_Jacobian(model_perfect, dh_ind1, dh_ind2, print_flag=True) # Jacobian computation
	G1 = np.load('splc_Jacobian1.npy')
	G2 = np.load('splc_Jacobian2.npy')

	G = np.concatenate((G1, G2), axis=1)

	#############################################################################
	# define the control parameters
	#############################################################################
	Nitr = 20 # number of control iterations
	n_trials = 10
	n_act = G1.shape[1] # number of active actuators on the DM
	n_pix = G1.shape[0] # number of pixels in the dark hole
	weight = np.ones(len(wavelength))
	alpha = 3e-7
	img_number = 4
	exp_time = 1

	# define the wavefront estimator
	params_values = params_set(G1, G2, 1e-10, 0.05, 3.6e-17, 1e-4)
	BPE_estimator = est.Batch_process(params_values)

	# define the sensing policy (computing DM probe command)
	sensor = sensing.Empirical_probe(model, params_values, img_number, 
									pair_wise=True, probe_area=[1, 17, -17, 17], method='alter')

	# define the camera noise model
	camera = detector.CCD(flux=2e9, readout_std=12, readout=True, photon=True, exp_time=exp_time)

	# define the system identifier for adaptive control
	n_batch = 20
	# vl = sysid.linear_vl(params_values, img_number//2)
	em_identifier = em.linear_em(params_values, img_number//2, model_type='normal')

	# decide whether to save the wfsc data
	save_data_flag = True
	if save_data_flag:
		data_train = {}
		data_train['u1'] = np.ones((n_act, Nitr))
		data_train['u2'] = np.ones((n_act, Nitr))
		data_train['u1p'] = np.ones((n_act, img_number, Nitr))
		data_train['u2p'] = np.ones((n_act, img_number, Nitr))
		data_train['I'] = np.ones((n_pix, img_number+1, Nitr))
		data_train['time'] = np.zeros(Nitr)


	#############################################################################
	# start WFSC trials, update model params using system ID after each trial
	#############################################################################
	# define model errors (shift images by some pixels)
	shift_pix = 0.5

	# wavefront trials
	contrast_set = []
	for trial in range(n_trials):
		u1 = np.zeros((34, 34))
		u2 = np.zeros((34, 34))
		contrast = np.zeros((Nitr, ))
		E_true = []
		for k in range(Nitr):
			#####################################################################
			# collect the unprobed image
			#####################################################################
			Ef = model.Propagate(u1, u2)
			Ef = sp.ndimage.shift(Ef.real, shift=[shift_pix, shift_pix, 0]) + 1j * sp.ndimage.shift(Ef.imag, shift=[shift_pix, shift_pix, 0])
			Ef_vector = Ef[dh_ind1, dh_ind2, :]
			If_vector = np.abs(Ef_vector)**2
			contrast[k] = np.mean(If_vector)
			If = camera.Add_noise(If_vector)
			print('The contrast at step #{} is {}'.format(k, contrast[k]))
			E_true.append(Ef_vector)
			
			#####################################################################
			# collect probe images
			#####################################################################
			print('The exposure time at step #{} is {}'.format(k, exp_time))

			R_coef = [params_values['R0']/exp_time**2, params_values['R1']/exp_time+4*params_values['Q0'], 4*params_values['Q1']]
			u_p = sensor.Probe_command(contrast[k], k, index=1, R_coef=R_coef)

			If_p = np.empty((len(dh_ind1), len(wavelength), img_number), dtype=float)
			Ef_p_set = np.empty((len(dh_ind1), len(wavelength), img_number), dtype=complex)
			for i in range(u_p.shape[2]):
				# images with positive probes
				Ef_p = model.Propagate(u1+u_p[:, :, i], u2)
				Ef_p = sp.ndimage.shift(Ef_p.real, shift=[shift_pix, shift_pix, 0]) + 1j * sp.ndimage.shift(Ef_p.imag, shift=[shift_pix, shift_pix, 0])
				Ef_p_vector = Ef_p[dh_ind1, dh_ind2, :]
				If_p_vector = np.abs(Ef_p_vector)**2
				If_p[:, :, 2*i] = camera.Add_noise(If_p_vector)
				Ef_p_set[:, :, 2*i] = Ef_p_vector
				print('The contrast of the No.{} postive image is {}'.format(i, np.mean(If_p_vector)))

				# images with negative probes
				Ef_p = model.Propagate(u1-u_p[:, :, i], u2)
				Ef_p = sp.ndimage.shift(Ef_p.real, shift=[shift_pix, shift_pix, 0]) + 1j * sp.ndimage.shift(Ef_p.imag, shift=[shift_pix, shift_pix, 0])
				Ef_p_vector = Ef_p[dh_ind1, dh_ind2, :]
				If_p_vector = np.abs(Ef_p_vector)**2
				If_p[:, :, 2*i+1] = camera.Add_noise(If_p_vector)
				Ef_p_set[:, :, 2*i+1] = Ef_p_vector
				print('The contrast of the No.{} negative image is {}'.format(i, np.mean(If_p_vector)))
				print('The contrast of the No.{} difference image is {}'.format(i, np.mean(np.abs(If_p[:, :, 2*i] - If_p[:, :, 2*i+1]))))

			#####################################################################
			# estimate the electric field using pair-wise batch process estimator
			#####################################################################
			u_p_vector = u_p[model.DMind1, model.DMind2, :]
			Ef_est, P_est = BPE_estimator.Estimate(If_p, u_p_vector, np.zeros(u_p_vector.shape), exp_time)
			
			#####################################################################
			# compute control command
			#####################################################################
			command = EFC(Ef_est, G, weight, alpha)
			u1[model.DMind1, model.DMind2] += command[:int(len(command)/2):]
			u2[model.DMind1, model.DMind2] += command[int(len(command)/2)::]

			if save_data_flag:
				data_train['u1'][:, k] = command[0:n_act]
				data_train['u2'][:, k]= command[n_act::]
				for k_image in range(img_number):
					data_train['u1p'][:, k_image, k] = ((-1)**(k_image%2)) * u_p[model.DMind1, model.DMind2, k_image//2]
					data_train['u2p'][:, k_image, k] = np.zeros((n_act, ))
				data_train['I'][:, :, k] = np.concatenate([np.squeeze(If).reshape((-1, 1)), np.squeeze(If_p)], 1)
				data_train['time'][k] = camera.exp_time
				

			if (k+1) % n_batch == 0:# and trial > 0:
				data_train_now = {}
				data_train_now['u1'] = data_train['u1'][:, k+1-n_batch:k+1]
				data_train_now['u2'] = data_train['u2'][:, k+1-n_batch:k+1]
				data_train_now['u1p'] = data_train['u1p'][:, :, k+1-n_batch:k+1]
				data_train_now['u2p'] = data_train['u2p'][:, :, k+1-n_batch:k+1]
				data_train_now['I'] = data_train['I'][:, :, k+1-n_batch:k+1]
				# mse_list = vl.train_params(data_train_now, lr=1e-8, lr2=1e-3, epoch=10, 
				# 				params_trainable='all', print_flag=True)
				# _, _, _, _ = em_identifier.initialize_noise_params(data_train, lr=1e-8, 
				# 					lr2=1e-2, mstep_itr=100, print_flag=True)
				mse_list = em_identifier.train_params(data_train, lr=1e-7, 
									lr2=1e-2, epoch=10, mstep_itr=2, print_flag=True, params_trainable='Jacobian')

				G1 = params_values['G1']
				G2 = params_values['G2']
				G = np.concatenate((G1, G2), axis=1)

		contrast_set.append(contrast)

	for k in range(len(contrast_set)):
		plt.figure(1), plt.semilogy(contrast_set[k])