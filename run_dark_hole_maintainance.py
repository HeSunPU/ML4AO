import scipy.io as sio
import scipy as sp
import numpy as np
from FPWCpy import systemID as sysid
from FPWCpy import estimation as est
from FPWCpy import sensing
from FPWCpy import detector
from FPWCpy import helper_function as hp
from FPWCpy.optical_layout import SPLC_model as splc
from FPWCpy.optical_layout import VORTEX_model as vortex
from FPWCpy import wfe
from FPWCpy import smpc
import matplotlib.pyplot as plt
import warnings

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
	params_values['Q1'] = Q1 # 0.05 # 0.5 # 1e-8 for u^2, 6e-8 for u^3, 0.5 for dE^2
	params_values['R0'] = R0#/exp_time**2 #1e-14
	params_values['R1'] = R1#/exp_time #1e-9
	return params_values

if __name__ == "__main__":
	# define the optical model
	wavelength = 635e-9 * np.ones(1)
	DM1gain = 5.06e-9 * np.ones((34, 34))
	DM2gain = 6.27e-9 * np.ones((34, 34))
	coronagraph_type = 'splc'#'vortex'#

	if coronagraph_type.lower() == 'splc':
		model = splc.Optical_Model(wavelength, DM1gain, DM2gain, wfe=True)
		model_perfect = splc.Optical_Model(wavelength, DM1gain, DM2gain, wfe=False)
		# define the dark hole region
		dh_ind1, dh_ind2 = hp.Compute_DH(model, dh_shape='wedge', range_r=[2.5, 9], range_angle=30)
	elif coronagraph_type.lower() == 'vortex':
		model = vortex.Optical_Model(wavelength, DM1gain, DM2gain, wfe=True)
		model_perfect = vortex.Optical_Model(wavelength, DM1gain, DM2gain, wfe=False)
		dh_ind1, dh_ind2 = hp.Compute_DH(model, dh_shape='circ', range_r=[3, 9], range_angle=30)



	# compute or load the Jacobian matrices
	# G1, G2 = hp.Compute_Jacobian(model_perfect, dh_ind1, dh_ind2, print_flag=True)

	# G1 = np.load('vortex_compact_Jacobian1.npy')
	# G2 = np.load('vortex_compact_Jacobian2.npy')

	G1 = np.load('splc_Jacobian1.npy')
	G2 = np.load('splc_Jacobian2.npy')

	G = np.concatenate((G1, G2), axis=1)


	# define the control parameters
	Nitr = 20 # number of control iterations
	n_trials = 1
	n_act = G1.shape[1] # number of active actuators on the DM
	n_pix = G1.shape[0] # number of pixels in the dark hole
	weight = np.ones(len(wavelength))
	alpha = 3e-7
	img_number = 4
	exp_time = 1

	# define the wavefront estimator
	params_values = params_set(G1, G2, 1e-10, 0.05, 3.6e-17, 5e-10)
	# params_values['R2'] = 4 * params_values['Q1']#4 * 0.045 #5e-3 # 5e-3 # 1e-2
	BPE_estimator = est.Batch_process(params_values)

	# define the sensing policy
	sensor0 = sensing.Empirical_probe(model, params_values, img_number, 
									pair_wise=True, probe_area=[1, 17, -17, 17], method='rot')

	# define the camera noise model
	camera = detector.CCD(flux=2e9, readout_std=12, readout=True, photon=True, exp_time=exp_time)

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


	# start wavefront control
	u1 = np.zeros((34, 34))
	u2 = np.zeros((34, 34))
	u10 = u1
	u20 = u2
	contrast = np.zeros((Nitr, ))
	E_true = []
	u_p_nominal_set = []
	u_p_optimal_set = []
	for k in range(Nitr):
		# collect the unprobed image
		Ef = model.Propagate(u1, u2)
		Ef_vector = Ef[dh_ind1, dh_ind2, :]
		If_vector = np.abs(Ef_vector)**2
		contrast[k] = np.mean(If_vector)
		If = camera.Add_noise(If_vector)
		print('The contrast at step #{} is {}'.format(k, contrast[k]))
		E_true.append(Ef_vector)
		
		
		# collect probe images
		print('The exposure time at step #{} is {}'.format(k, exp_time))

		R_coef = [params_values['R0']/exp_time**2, params_values['R1']/exp_time+4*params_values['Q0'], 4*params_values['Q1']]
		# R_coef = []
		u_p = sensor0.Probe_command(contrast[k], k, index=1, R_coef=R_coef)
		
		If_p = np.empty((len(dh_ind1), len(wavelength), img_number), dtype=float)
		Ef_p_set = np.empty((len(dh_ind1), len(wavelength), img_number), dtype=complex)
		for i in range(u_p.shape[2]):
			# images with positive probes
			Ef_p = model.Propagate(u1+u_p[:, :, i], u2)
			Ef_p_vector = Ef_p[dh_ind1, dh_ind2, :]
			If_p_vector = np.abs(Ef_p_vector)**2
			# If_p[:, :, 2*i] = If_p_vector
			If_p[:, :, 2*i] = camera.Add_noise(If_p_vector)
			Ef_p_set[:, :, 2*i] = Ef_p_vector
			print('The contrast of the No.{} postive image is {}'.format(i, np.mean(If_p_vector)))
			# images with negative probes
			Ef_p = model.Propagate(u1-u_p[:, :, i], u2)
			Ef_p_vector = Ef_p[dh_ind1, dh_ind2, :]
			If_p_vector = np.abs(Ef_p_vector)**2
			# If_p[:, :, 2*i+1] = If_p_vector
			If_p[:, :, 2*i+1] = camera.Add_noise(If_p_vector)
			Ef_p_set[:, :, 2*i+1] = Ef_p_vector
			print('The contrast of the No.{} negative image is {}'.format(i, np.mean(If_p_vector)))
			# print('The contrast of the No.{} difference image is {}'.format(i, np.mean(np.abs(If_p[:, :, 2*i] - If_p[:, :, 2*i+1]))))
		
		u_p_vector = u_p[model.DMind1, model.DMind2, :]
		Ef_est, P_est = BPE_estimator.Estimate(If_p, u_p_vector, np.zeros(u_p_vector.shape), exp_time)
		
		# compute control command
		command = EFC(Ef_est, G, weight, alpha)
		# command += 0.3*np.std(command)*np.random.normal(size=command.shape)
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

	# save the control commands
	u10 = np.array(u1)
	u20 = np.array(u2)
	P_est0 = np.array(P_est)
	Ef_est0 = np.array(Ef_est)
	command0 = np.array(command)


# generate a drift process
n_step = 500
drift_process = 1e-10*np.random.normal(size=[n_step, 15])


# open loop
u1 = np.array(u10)
u2 = np.array(u20)
phase_error = wfe.PhaseErr(wfe_shape=model.Ein.shape[0], coef=np.zeros(15), drift_dynamics=1e-10*np.ones(15))
# n_step = 500

Ef_set = np.zeros([Ef.shape[0], Ef.shape[1], n_step])
contrast_drift = np.zeros(n_step)

for k in range(n_step):
	Ef = model.Propagate(u1, u2, model.Ein * np.exp(1j * 2 * np.pi / model.wavelength[0] * phase_error.phase_map))
	Ef_vector = Ef[dh_ind1, dh_ind2, :]
	If_vector = np.abs(Ef_vector)**2
	contrast_drift[k] = np.mean(If_vector)
	print('The contrast at step #{} is {}'.format(k, contrast_drift[k]))
	Ef_set[:, :, k] = np.squeeze(Ef)
	phase_error.drift(drift_process[k, :])



# closed loop dark hole maintainance
u1 = np.array(u10)
u2 = np.array(u20)
P_est = np.zeros((1292, 3, 3))
P_est[:, 0:2, 0:2] = np.array(P_est0)
P_est[:, 2, 2] = 1e-20
Ef_est = np.array(Ef_est0)
command = np.array(command0)
dither_coef = 3e-3
alpha = 4e-6
params_values['Q0'] = 1e-10
DH_controller = smpc.SMPC(params_values, 5)
EKF_DH_estimator = est.Extended_Kalman_filter_DH_maintain(params_values)
phase_error = wfe.PhaseErr(wfe_shape=model.Ein.shape[0], coef=np.zeros(15), drift_dynamics=1e-10*np.ones(15))
# n_step = 500

command_set = np.zeros([2*n_act, n_step])
command_EFC_set = np.zeros([2*n_act, n_step])

Ef_set = np.zeros([Ef.shape[0], Ef.shape[1], n_step])
Ef_est_set = np.zeros(Ef_set.shape)
contrast_drift = np.zeros(n_step)
Iinco_est = np.zeros(Ef_est.shape)
for k in range(n_step):
	Ef = model.Propagate(u1, u2, model.Ein * np.exp(1j * 2 * np.pi / model.wavelength[0] * phase_error.phase_map))
	Ef_vector = Ef[dh_ind1, dh_ind2, :]
	If_vector = np.abs(Ef_vector)**2
	If_vector = camera.Add_noise(If_vector)
	contrast_drift[k] = np.mean(If_vector)
	print('The contrast at step #{} is {}'.format(k, contrast_drift[k]))
	Ef_set[:, :, k] = np.squeeze(Ef)
	# estimate the electric field
	u1c = command[:int(len(command)/2):]
	u2c = command[int(len(command)/2)::]
	Ef_est, Iinco_est, P_est = EKF_DH_estimator.Estimate(If_vector, Ef_est, Iinco_est, P_est, u1c, u2c, exp_time)
	Ef_est_set[dh_ind1, dh_ind2, k] = np.squeeze(Ef_est)
	# compute control command
	# command = EFC(Ef_est, G, weight, alpha) + dither_coef*np.random.normal(size=command.shape)
	# command = dither_coef*np.random.normal(size=command.shape)#dither_coef*np.random.normal(size=command.shape)
	# u1c, u2c = DH_controller.Control(command[:int(len(command)/2):], command[int(len(command)/2)::], 
	# 				Ef_est, Iinco_est, P_est, camera.exp_time, beta=1e-2, rate=5e-4, Nitr=1000, print_flag=True)
	if k % 2 == 0:
		u1c, u2c, u1c_next, u2c_next = DH_controller.Control(dither_coef*np.random.normal(size=u1c.shape), dither_coef*np.random.normal(size=u2c.shape), 
						dither_coef*np.random.normal(size=u1c.shape), dither_coef*np.random.normal(size=u2c.shape), 
						Ef_est, Iinco_est, P_est, camera.exp_time, beta=1e0, rate=5e-4, Nitr=1000, print_flag=True)

		command = np.concatenate([u1c, u2c])
		command_next = np.concatenate([u1c_next, u2c_next])
	else:
		command = np.array(command_next)

	command_set[:, k] = command
	command_EFC_set = EFC(Ef_est, G, weight, alpha)
	u1[model.DMind1, model.DMind2] += command[:int(len(command)/2):]
	u2[model.DMind1, model.DMind2] += command[int(len(command)/2)::]
	phase_error.drift(drift_process[k, :])