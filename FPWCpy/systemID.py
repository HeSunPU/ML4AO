"""
created on Sun Apr. 14, 2019

@author: He Sun, Princeton University

Identify the state space model of the optical system.
The optical system is modeled as a hidden Markov model (HMM).

y_0     y_1             y_n
 ^       ^               ^
H|      H|              H|
x_0 --> x_1 --> ... --> x_n
     F   ^   F       F   ^
        G|              G|
        u_1             u_n

x_k = F x_{k-1} + G u_k + w_k, w_k ~ N(0, Q),
y_k = H x_k + n_k, n_k ~ N(0, R),

where
u_k: the control inputs,
y_k: the observations,
x_k: the hidden states,
F: the state transition matrix,
G: the control Jacobian matrix,
H: the observation matrix,
w_k: the process noises,
n_k: the observation noises.

Here we want to learn the model parameters, F, G, H, Q, R, using u_1:n, y_1:n.
In our specific case, we assume
(1) y_k = Ip_k+ - Ip_k-, Ip_k+/- are the pair-wise probe images
(2) F = I, the identity matrix
(3) H_k = 4 (G up_k).T, up_k is the pair-wise probe commands
(4) Q_k = Q0 + Q1 * mean(|G u_k|^2)
(5) R_k = R0 + R1 * (Ip_k+ + Ip_k-) + R2 * (Ip_k+^2 + Ip_k-^2)

References: 
"Modern wavefront control for space-based exoplanet coronagraph imaging", He Sun et.al, 2019
"Identification and adaptive control of a high-contrast focal plane wavefront correction system", He Sun et.al, 2018
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.ion()

class SSM(object):
	# linear state space model for the optical system
	# the incoherent light is not considered here, however, it should be easy to include them
	def __init__(self, Jacobian1, Jacobian2, Q0, Q1, R0, R1, n_observ=4):
		# parameters Q0, Q1, R0, R1 define the noise covariance of the state space model
		# process noises covariance: Q = Q0 + Q1 * sum(uc^2)
		# observation noises covariance: R = R0 + R1 * probe_contrast + R2 * probe_contrast^2
		# Jacobian matrices
		self.G1_real = tf.Variable(Jacobian1.real, dtype=tf.float64)
		self.G1_imag = tf.Variable(Jacobian1.imag, dtype=tf.float64)
		self.G2_real = tf.Variable(Jacobian2.real, dtype=tf.float64)
		self.G2_imag = tf.Variable(Jacobian2.imag, dtype=tf.float64)
		
		# noise parameters, all of them should be positive
		self.q0 = tf.Variable(np.log(Q0), dtype=tf.float64)
		self.q1 = tf.Variable(np.log(Q1), dtype=tf.float64)
		self.r0 = tf.Variable(np.log(R0), dtype=tf.float64)
		self.r1 = tf.Variable(np.log(R1), dtype=tf.float64)
		# self.r2 = tf.Variable(np.log(R2), dtype=tf.float64)

		self.Q0 = tf.exp(self.q0)
		self.Q1 = tf.exp(self.q1)
		self.R0 = tf.exp(self.r0)
		self.R1 = tf.exp(self.r1)
		# self.R2 = tf.exp(self.r2)
		self.num_pix = Jacobian1.shape[0]
		self.num_act = Jacobian1.shape[1]
		self.n_observ = int(n_observ)
	def transition(self, Enp, u1, u2):
		# state transition model
		Enp_next = Enp + tf.cast(tf.tensordot(u1, self.G1_real, axes=[[-1], [1]]) + tf.tensordot(u2, self.G2_real, axes=[[-1], [1]]), tf.complex128) + \
					+ 1j * tf.cast(tf.tensordot(u1, self.G1_imag, axes=[[-1], [1]]) + tf.tensordot(u2, self.G2_imag, axes=[[-1], [1]]), tf.complex128)
		return Enp_next
	def transition_covariance(self, Enp, u1, u2):
		# covariance of process/transition noises
		# u1_square = tf.reduce_sum(tf.abs(u1)**2, axis=1)
		# u2_square = tf.reduce_sum(tf.abs(u2)**2, axis=1)
		# Qco = tf.tensordot(tf.expand_dims(u1_square, 1), tf.expand_dims(self.Q1*tf.ones(self.num_pix, dtype=tf.float64), 0), axes=[[1], [0]]) + \
		# 	tf.tensordot(tf.expand_dims(u2_square, 1), tf.expand_dims(self.Q1*tf.ones(self.num_pix, dtype=tf.float64), 0), axes=[[1], [0]]) + self.Q0 + 1e-14
		# u1_cubic = tf.reduce_sum(tf.abs(u1)**3, axis=1)
		# u2_cubic = tf.reduce_sum(tf.abs(u2)**3, axis=1)
		# Qco = tf.tensordot(tf.expand_dims(u1_cubic, 1), tf.expand_dims(self.Q1*tf.ones(self.num_pix, dtype=tf.float64), 0), axes=[[1], [0]]) + \
		# 	tf.tensordot(tf.expand_dims(u2_cubic, 1), tf.expand_dims(self.Q1*tf.ones(self.num_pix, dtype=tf.float64), 0), axes=[[1], [0]]) + self.Q0 + 1e-14
		dE = tf.cast(tf.tensordot(u1, self.G1_real, axes=[[-1], [1]]) + tf.tensordot(u2, self.G2_real, axes=[[-1], [1]]), tf.complex128) + \
					+ 1j * tf.cast(tf.tensordot(u1, self.G1_imag, axes=[[-1], [1]]) + tf.tensordot(u2, self.G2_imag, axes=[[-1], [1]]), tf.complex128)
		dE_square = tf.reduce_mean(tf.abs(dE)**2, axis=1)
		Qco = tf.tensordot(tf.expand_dims(dE_square, 1), tf.expand_dims(self.Q1*tf.ones(self.num_pix, dtype=tf.float64), 0), axes=[[1], [0]]) + self.Q0 + 1e-14
		return Qco
	def observation(self, Enp, u_p1, u_p2):
		# observation model
		n_probe = self.n_observ
		E_p_list = []
		I_p_correction_list = []
		for k in range(n_probe):
			E_p = self.transition(Enp, u_p1[:, k, :], u_p2[:, k, :])
			E_p_list.append(tf.expand_dims(E_p, 1))
			I_p_correction = 2*self.transition_covariance(Enp, u_p1[:, k, :], u_p2[:, k, :])
			I_p_correction_list.append(tf.expand_dims(I_p_correction, 1))

		E_p_obs = tf.concat(E_p_list, axis=1)
		I_p_cor = tf.concat(I_p_correction_list, axis=1)
		I_p_obs = tf.abs(E_p_obs)**2 + I_p_cor
		return E_p_obs, I_p_obs
	def observation_covariance(self, I_p_obs, u_p1, u_p2):#(self, Enp, u_p1, u_p2):
		# covariance of observation noises
		# _, I_p_obs = self.observation(Enp, u_p1, u_p2)
		contrast_p = tf.reduce_mean(I_p_obs, axis=2)
		dEp = tf.cast(tf.tensordot(u_p1, self.G1_real, axes=[[-1], [1]]) + tf.tensordot(u_p2, self.G2_real, axes=[[-1], [1]]), tf.complex128) + \
					+ 1j * tf.cast(tf.tensordot(u_p1, self.G1_imag, axes=[[-1], [1]]) + tf.tensordot(u_p2, self.G2_imag, axes=[[-1], [1]]), tf.complex128)
		d_contrast_p = tf.reduce_mean(tf.abs(dEp)**2, axis=2)
		R = tf.tensordot(tf.expand_dims(self.R0 + self.R1*contrast_p + 4*(self.Q0+self.Q1*d_contrast_p)*contrast_p, axis=-1), tf.ones((1, self.num_pix), dtype=tf.float64), axes=[[-1], [0]]) + 1e-24	
		return R
	def get_params(self):
		# get the parameters of the identified system
		return [self.G1_real, self.G1_imag, self.G2_real, self.G2_imag,
				self.q0, self.q1, self.r0, self.r1]


def LSEnet(model, Ip, u1p, u2p):
	# computation graph that defines least squared estimation of the electric field
	delta_Ep_pred = tf.cast(tf.tensordot(u1p, model.G1_real, axes=[[-1], [1]]) + tf.tensordot(u2p, model.G2_real, axes=[[-1], [1]]), tf.complex128) + \
			+ 1j * tf.cast(tf.tensordot(u1p, model.G1_imag, axes=[[-1], [1]]) + tf.tensordot(u2p, model.G2_imag, axes=[[-1], [1]]), tf.complex128)
	delta_Ep_expand = tf.expand_dims(delta_Ep_pred, 2)
	delta_Ep_expand_diff = delta_Ep_expand[:, 1::2, :, :] - delta_Ep_expand[:, 2::2, :, :]
	y = tf.transpose(Ip[:, 1::2, :]-Ip[:, 2::2, :], [0, 2, 1])
	H = tf.concat([2*tf.real(delta_Ep_expand_diff), 2*tf.imag(delta_Ep_expand_diff)], axis=2)
	H = tf.transpose(H, [0, 3, 1, 2])
	Ht_H = tf.matmul(tf.transpose(H, [0, 1, 3, 2]), H)
	Ht_H_inv_Ht = tf.matmul(tf.matrix_inverse(Ht_H+tf.eye(2, dtype=tf.float64)*1e-12), tf.transpose(H, [0, 1, 3, 2]))
	x_new = tf.squeeze(tf.matmul(Ht_H_inv_Ht, tf.expand_dims(y, -1)), -1)
	
	n_observ = model.n_observ
	contrast_p = tf.reduce_mean(Ip, axis=2)

	d_contrast_p = tf.reduce_mean(tf.abs(delta_Ep_pred)**2, axis=2)

	Rp = tf.tensordot(tf.expand_dims(model.R0 + model.R1*contrast_p + 4*(model.Q0+model.Q1*d_contrast_p)*contrast_p, axis=-1), 
						tf.ones((1, model.num_pix), dtype=tf.float64), axes=[[-1], [0]]) + 1e-24	
	Rp = tf.transpose(Rp, [0, 2, 1])
	R_diff = Rp[:, :, 1::2]+Rp[:, :, 2::2]
	R = tf.matrix_set_diag(tf.concat([tf.expand_dims(tf.zeros_like(R_diff), -1)]*(n_observ//2), -1), R_diff)
	P_new = tf.matmul(tf.matmul(Ht_H_inv_Ht, R), tf.transpose(Ht_H_inv_Ht, [0, 1, 3, 2]))
	Enp_pred_new = tf.cast(x_new[:, :, 0], dtype=tf.complex128) + 1j * tf.cast(x_new[:, :, 1], dtype=tf.complex128)
	return Enp_pred_new, P_new, H

def KFnet(model, Ip, Enp_old, P_old, u1c, u2c, u1p, u2p):
	# computation graph that defines Kalman filtering estimation of the electric field
	delta_Ep_pred = tf.cast(tf.tensordot(u1p, model.G1_real, axes=[[-1], [1]]) + tf.tensordot(u2p, model.G2_real, axes=[[-1], [1]]), tf.complex128) + \
			+ 1j * tf.cast(tf.tensordot(u1p, model.G1_imag, axes=[[-1], [1]]) + tf.tensordot(u2p, model.G2_imag, axes=[[-1], [1]]), tf.complex128)
	delta_Ep_expand = tf.expand_dims(delta_Ep_pred, 2)
	delta_Ep_expand_diff = delta_Ep_expand[:, 1::2, :, :] - delta_Ep_expand[:, 2::2, :, :]
	y = tf.transpose(Ip[:, 1::2, :]-Ip[:, 2::2, :], [0, 2, 1])
	H = tf.concat([2*tf.real(delta_Ep_expand_diff), 2*tf.imag(delta_Ep_expand_diff)], axis=2)
	H = tf.transpose(H, [0, 3, 1, 2])

	n_observ = model.n_observ
	contrast_p = tf.reduce_mean(Ip, axis=2)
	d_contrast_p = tf.reduce_mean(tf.abs(delta_Ep_pred)**2, axis=2)

	Rp = tf.tensordot(tf.expand_dims(model.R0 + model.R1*contrast_p + 4*(model.Q0+model.Q1*d_contrast_p)*contrast_p, axis=-1), 
						tf.ones((1, model.num_pix), dtype=tf.float64), axes=[[-1], [0]]) + 1e-24	
	Rp = tf.transpose(Rp, [0, 2, 1])
	R_diff = Rp[:, :, 1::2]+Rp[:, :, 2::2]
	R = tf.matrix_set_diag(tf.concat([tf.expand_dims(tf.zeros_like(R_diff), -1)]*(n_observ//2), -1), R_diff)

	Qco = model.transition_covariance(Enp_old, u1c, u2c)
	Q = tf.concat([tf.expand_dims(Qco, 2), tf.expand_dims(Qco, 2)], axis=2)
	Q = tf.matrix_set_diag(tf.concat([tf.expand_dims(tf.zeros_like(Q), -1)]*2, -1), Q)

	# state and covariance prediction
	Enp_pred = model.transition(Enp_old, u1c, u2c)
	Enp_expand = tf.expand_dims(Enp_pred, 2)
	x = tf.concat([tf.real(Enp_expand), tf.imag(Enp_expand)], axis=2)
	P = P_old + Q

	# state and covariance update
	y_model = tf.squeeze(tf.matmul(H, tf.expand_dims(x, -1)), -1)
	S = R + tf.matmul(tf.matmul(H, P), tf.transpose(H, [0, 1, 3, 2]))
	K = tf.matmul(tf.matmul(P, tf.transpose(H, [0, 1, 3, 2])), tf.matrix_inverse(S))
	x_new = x + tf.squeeze(tf.matmul(K, tf.expand_dims(y-y_model, -1)), -1)
	P_new = P - tf.matmul(tf.matmul(K, H), P)
	
	Enp_pred_new = tf.cast(x_new[:, :, 0], dtype=tf.complex128) + 1j * tf.cast(x_new[:, :, 1], dtype=tf.complex128)
	return Enp_pred_new, P_new


class linear_vl:
	def __init__(self, params_values, n_pair):
		self.params_values = params_values

		G1 = np.squeeze(params_values['G1'])
		G2 = np.squeeze(params_values['G2'])
		Q0 = params_values['Q0']
		Q1 = params_values['Q1']
		R0 = params_values['R0']
		R1 = params_values['R1']
		# R2 = params_values['R2']

		n_act = G1.shape[1] # number of active actuators on the DM
		n_pix = G1.shape[0] # number of pixels in the dark hole
		n_image = 2 * n_pair + 1 # number of probe images in each control step

		# define the placeholders for the computation graph
		u1c = tf.placeholder(tf.float64, shape=(None, n_act))
		u2c = tf.placeholder(tf.float64, shape=(None, n_act))
		u1p = tf.placeholder(tf.float64, shape=(None, n_image, n_act))
		u2p = tf.placeholder(tf.float64, shape=(None, n_image, n_act))
		Enp_old = tf.placeholder(tf.complex128, shape=(None, n_pix))
		Ip = tf.placeholder(tf.float64, shape=(None, n_image, n_pix))
		P_old = tf.placeholder(tf.float64, shape=(None, n_pix, 2, 2))
		learning_rate = tf.placeholder(tf.float64, shape=())
		learning_rate2 = tf.placeholder(tf.float64, shape=())

		# define the optical model as a state space model (SSM) or a neural network
		# model = SSM(G1, G2, Q0, Q1, R0, R1, R2, n_image)
		model = SSM(G1, G2, Q0, Q1, R0, R1, n_image)

		# define the relations of the control/probe inputs, camera images and hidden electric fields
		Enp_pred = model.transition(Enp_old, u1c, u2c)
		Qco = model.transition_covariance(Enp_old, u1c, u2c)

		Enp_est, P_est, H = LSEnet(model, Ip, u1p, u2p)
		Enp_est2, P_est2, _ = LSEnet(model, Ip, u1p, u2p)
		# Enp_est, P_est = KFnet(model, Ip, Enp_old, P_old, u1c, u2c, u1p, u2p)


		_, Ip_pred = model.observation(Enp_est, u1p, u2p)
		Ip_pred_err = tf.tile(tf.expand_dims(tf.trace(P_est), 1), [1, n_image, 1])
		obs_cov = 0.#4 * tf.tensordot(tf.expand_dims(tf.reduce_mean(Ip, -1), -1), tf.ones((1, n_pix), dtype=tf.float64), [-1, 0]) * tf.tile(tf.expand_dims(tf.trace(P_est), 1), [1, n_image, 1])
		Rp = model.observation_covariance(Ip, u1p, u2p)

		# Ip_pred_diff = Ip_pred[:, 1::2, :] - Ip_pred[:, 2::2, :]
		# Ip_diff = Ip[:, 1::2, :] - Ip[:, 2::2, :]
		# Rp_diff = Rp[:, 1::2, :] + Rp[:, 2::2, :]
		# HPHt = tf.matmul(tf.matmul(H, P_est), tf.transpose(H, [0, 1, 3, 2]))
		# obs_bias = tf.transpose(tf.linalg.diag_part(HPHt), [0, 2, 1])

		# evidence lower bound (elbo): cost function for system identification
		# we need to maximize the elbo for system ID
		elbo = - tf.reduce_sum((tf.abs(Ip-Ip_pred-Ip_pred_err)**2 + obs_cov) / Rp) - tf.reduce_sum(tf.log(2*np.pi*Rp)) - \
				(tf.reduce_sum(tf.abs(Enp_pred-Enp_est)**2 / Qco) + tf.reduce_sum(2 * tf.log(Qco)) - \
				 tf.reduce_sum(tf.linalg.logdet(P_est)) + tf.reduce_sum(tf.trace(P_est) / Qco))

		# elbo = - (tf.reduce_sum(tf.abs(Enp_pred-Enp_est)**2 / Qco) + tf.reduce_sum(2 * tf.log(Qco)) - \
		# 		 tf.reduce_sum(tf.linalg.logdet(P_est)) + tf.reduce_sum(tf.trace(P_est) / Qco))


		# elbo = - tf.reduce_sum((tf.abs(Ip_diff-Ip_pred_diff)**2 + obs_bias) / Rp_diff) - tf.reduce_sum(tf.log(2*np.pi*Rp)) - \
		# 		(tf.reduce_sum(tf.abs(Enp_pred-Enp_est)**2 / Qco) + tf.reduce_sum(2 * tf.log(Qco)) - \
		# 		 tf.reduce_sum(tf.linalg.logdet(P_est)) + tf.reduce_sum(tf.trace(P_est) / Qco))

		
		params_list = model.get_params() # parameters to be identified

		self.model = model
		self.Enp_est2 = Enp_est2
		self.P_est2 = P_est2
		self.Enp_est = Enp_est
		self.P_est = P_est
		self.Ip = Ip
		self.u1p = u1p
		self.u2p = u2p
		self.u1c = u1c
		self.u2c = u2c
		self.Enp_old = Enp_old
		self.P_old = P_old
		self.learning_rate = learning_rate
		self.learning_rate2 = learning_rate2
		self.elbo = elbo

		# mean squared error (MSE): a metric for checking the system ID results
		self.MSE = tf.reduce_sum(tf.abs(Ip - Ip_pred)**2)

		# start identifying/learning the model parameters
		self.train_Jacobian = tf.train.AdamOptimizer(learning_rate=learning_rate, 
												beta1=0.99, beta2=0.9999, epsilon=1e-08).minimize(-elbo, var_list=params_list[0:4])
		# self.train_noise_coef = tf.train.AdamOptimizer(learning_rate=learning_rate2, 
		# 										beta1=0.99, beta2=0.9999, epsilon=1e-08).minimize(-elbo, var_list=params_list[4::])
		self.train_noise_coef = tf.train.AdamOptimizer(learning_rate=learning_rate2, 
												beta1=0.99, beta2=0.9999, epsilon=1e-08).minimize(-elbo, var_list=[params_list[4], params_list[5]])
		self.train_group = tf.group(self.train_Jacobian, self.train_noise_coef)
		self.init = tf.global_variables_initializer()

	def train_params(self, data_train, lr=1e-7, lr2=1e-2, epoch=10, 
					print_flag=False, params_trainable='all'):
		u1_train = data_train['u1']
		u2_train = data_train['u2']
		u1p_train = data_train['u1p']
		u2p_train = data_train['u2p']
		image_train = data_train['I']

		n_step = u1_train.shape[1] # number of control steps
		n_act = u1_train.shape[0] # number of active actuators on the DM
		n_pix = image_train.shape[0] # number of pixels in the dark hole
		n_pair = u1p_train.shape[1]//2 # number of probing pairs in each control step
		n_image = 2 * n_pair + 1 # number of probe images in each control step

		u1p_train = np.concatenate([np.zeros((n_act, 1, n_step)), u1p_train], axis=1)
		u2p_train = np.concatenate([np.zeros((n_act, 1, n_step)), u2p_train], axis=1)

		if params_trainable.lower() == 'jacobian':
			train_op = self.train_Jacobian
		elif params_trainable.lower() == 'noise_coef':
			train_op = self.train_noise_coef
		elif params_trainable.lower() == 'all':
			train_op = self.train_group
		else:
			raise ValueError('Trainable params are not well defined!')

		mse_list = []
		with tf.Session() as sess:
			sess.run(self.init)
			# assign values to the variables in the model
			self.model.G1_real.load(np.squeeze(self.params_values['G1']).real)
			self.model.G1_imag.load(np.squeeze(self.params_values['G1']).imag)
			self.model.G2_real.load(np.squeeze(self.params_values['G2']).real)
			self.model.G2_imag.load(np.squeeze(self.params_values['G2']).imag)
			self.model.q0.load(np.log(self.params_values['Q0']))
			self.model.q1.load(np.log(self.params_values['Q1']))
			self.model.r0.load(np.log(self.params_values['R0']))
			self.model.r1.load(np.log(self.params_values['R1']))
			# self.model.r2.load(np.log(self.params_values['R2']))

			# Enp_est_values, P_est_values = sess.run([self.Enp_est2, self.P_est2], feed_dict={self.Ip: np.transpose(image_train, [2, 1, 0]),
			# 												self.u1p: np.transpose(u1p_train, [2, 1, 0]),
			# 												self.u2p: np.transpose(u2p_train, [2, 1, 0])})
			# mse = sess.run(self.MSE, feed_dict={self.Ip: np.transpose(image_train[:, :, 1:n_step], [2, 1, 0]),
			# 								self.u1p: np.transpose(u1p_train[:, :, 1:n_step], [2, 1, 0]),
			# 								self.u2p: np.transpose(u2p_train[:, :, 1:n_step], [2, 1, 0]),
			# 								self.u1c: np.transpose(u1_train[:,0:n_step-1]), 
			# 								self.u2c: np.transpose(u2_train[:,0:n_step-1]),
			# 								self.Enp_old: Enp_est_values[0:n_step-1, :],
			# 								self.P_old: P_est_values[0:n_step-1, :, :, :]})

			Enp_est_values, P_est_values = sess.run([self.Enp_est, self.P_est], feed_dict={self.Ip: np.transpose(image_train, [2, 1, 0]),
															self.u1p: np.transpose(u1p_train, [2, 1, 0]),
															self.u2p: np.transpose(u2p_train, [2, 1, 0])})
			mse = sess.run(self.MSE, feed_dict={self.Ip: np.transpose(image_train[:, :, 1:n_step], [2, 1, 0]),
											self.u1p: np.transpose(u1p_train[:, :, 1:n_step], [2, 1, 0]),
											self.u2p: np.transpose(u2p_train[:, :, 1:n_step], [2, 1, 0])})

			mse_list.append(mse)
			print('initial MSE: {}'.format(mse))

			for k in range(epoch):
				# Enp_est_values0, P_est_values0 = sess.run([self.Enp_est2, self.P_est2], feed_dict={self.Ip: np.transpose(np.expand_dims(image_train[:, :, 0], -1), [2, 1, 0]),
				# 											self.u1p: np.transpose(np.expand_dims(u1p_train[:, :, 0], -1), [2, 1, 0]),
				# 											self.u2p: np.transpose(np.expand_dims(u2p_train[:, :, 0], -1), [2, 1, 0])})

				# Enp_est_values[0, :] = np.squeeze(Enp_est_values0)
				# P_est_values[0, :, :, :] = np.array([1e-6*np.eye(2)] * n_pix)# np.squeeze(P_est_values0)
				# for i in range(1, n_step):
				# 	Enp_est_values_now, P_est_values_now = sess.run([self.Enp_est, self.P_est], feed_dict={self.Ip: np.transpose(np.expand_dims(image_train[:, :, i], -1), [2, 1, 0]),
				# 								self.u1p: np.transpose(np.expand_dims(u1p_train[:, :, i], -1), [2, 1, 0]),
				# 								self.u2p: np.transpose(np.expand_dims(u2p_train[:, :, i], -1), [2, 1, 0]),
				# 								self.u1c: np.transpose(np.expand_dims(u1_train[:,i-1], -1)), 
				# 								self.u2c: np.transpose(np.expand_dims(u2_train[:,i-1], -1)),
				# 								self.Enp_old: np.expand_dims(Enp_est_values[i-1, :], 0),
				# 								self.P_old: np.expand_dims(P_est_values[i-1, :, :, :], 0)})
				# 	Enp_est_values[i, :] = np.squeeze(Enp_est_values_now)
				# 	P_est_values[i, :, :, :] = np.squeeze(P_est_values_now)


				# sess.run(train_op, feed_dict={self.Enp_old: Enp_est_values[0:n_step-1, :], self.P_old: P_est_values[0:n_step-1, :, :, :],
				# 								self.Ip: np.transpose(image_train[:, :, 1:n_step], [2, 1, 0]),
				# 								self.u1c: np.transpose(u1_train[:,0:n_step-1]), self.u2c: np.transpose(u2_train[:,0:n_step-1]),
				# 								self.u1p: np.transpose(u1p_train[:, :, 1:n_step], [2, 1, 0]), self.u2p: np.transpose(u2p_train[:, :, 1:n_step], [2, 1, 0]),
				# 								self.learning_rate: lr, self.learning_rate2: lr2})


				# Enp_est_values, P_est_values = sess.run([self.Enp_est2, self.P_est2], feed_dict={self.Ip: np.transpose(image_train, [2, 1, 0]),
				# 											self.u1p: np.transpose(u1p_train, [2, 1, 0]),
				# 											self.u2p: np.transpose(u2p_train, [2, 1, 0])})
				# mse = sess.run(self.MSE, feed_dict={self.Ip: np.transpose(image_train[:, :, 1:n_step], [2, 1, 0]),
				# 							self.u1p: np.transpose(u1p_train[:, :, 1:n_step], [2, 1, 0]),
				# 							self.u2p: np.transpose(u2p_train[:, :, 1:n_step], [2, 1, 0]),
				# 							self.u1c: np.transpose(u1_train[:,0:n_step-1]), 
				# 							self.u2c: np.transpose(u2_train[:,0:n_step-1]),
				# 							self.Enp_old: Enp_est_values[0:n_step-1, :],
				# 							self.P_old: P_est_values[0:n_step-1, :, :, :]})

				Enp_est_values, P_est_values = sess.run([self.Enp_est, self.P_est], feed_dict={self.Ip: np.transpose(image_train, [2, 1, 0]),
															self.u1p: np.transpose(u1p_train, [2, 1, 0]),
															self.u2p: np.transpose(u2p_train, [2, 1, 0])})
				sess.run(train_op, feed_dict={self.Enp_old: Enp_est_values[0:n_step-1, :], self.P_old: P_est_values[0:n_step-1, :, :, :],
												self.Ip: np.transpose(image_train[:, :, 1:n_step], [2, 1, 0]),
												self.u1c: np.transpose(u1_train[:,0:n_step-1]), self.u2c: np.transpose(u2_train[:,0:n_step-1]),
												self.u1p: np.transpose(u1p_train[:, :, 1:n_step], [2, 1, 0]), self.u2p: np.transpose(u2p_train[:, :, 1:n_step], [2, 1, 0]),
												self.learning_rate: lr, self.learning_rate2: lr2})
				mse = sess.run(self.MSE, feed_dict={self.Ip: np.transpose(image_train[:, :, 1:n_step], [2, 1, 0]),
											self.u1p: np.transpose(u1p_train[:, :, 1:n_step], [2, 1, 0]),
											self.u2p: np.transpose(u2p_train[:, :, 1:n_step], [2, 1, 0])})

				if print_flag:
					print('epoch {} MSE: {}'.format(k, mse))
					# print('Q1: {}, R2: {}'.format(sess.run(self.model.Q1), sess.run(self.model.R2)))
					print('Q0: {}, Q1: {}'.format(sess.run(self.model.Q0), sess.run(self.model.Q1)))

			# update the model parameters
			self.params_values['G1'] = sess.run(self.model.G1_real) + 1j * sess.run(self.model.G1_imag)
			self.params_values['G2'] = sess.run(self.model.G2_real) + 1j * sess.run(self.model.G2_imag)
			self.params_values['G1'] = np.expand_dims(self.params_values['G1'], -1)
			self.params_values['G2'] = np.expand_dims(self.params_values['G2'], -1)
			self.params_values['Q0'] = np.exp(sess.run(self.model.q0))
			self.params_values['Q1'] = np.exp(sess.run(self.model.q1))
			self.params_values['R0'] = np.exp(sess.run(self.model.r0))
			self.params_values['R1'] = np.exp(sess.run(self.model.r1))
			# self.params_values['R2'] = np.exp(sess.run(self.model.r2))

		return mse_list


