"""
created on Mon Jul. 25, 2019

@author: He Sun, Princeton University

Stochastic model predictive control for focal plane wavefront control

Still under testing!!!!

"""

import numpy as np
import scipy as sp
import tensorflow as tf
def IEKF(params_values, n_pix, y, x_old, dx, Q, P_old, time, iterations):
	# G1 = np.squeeze(params_values['G1'])
	# G2 = np.squeeze(params_values['G2'])
	# dEc = tf.matmul(G1, tf.cast(tf.expand_dims(u1c, -1), tf.complex128)) +\
	# 		 tf.matmul(G2, tf.cast(tf.expand_dims(u2c, -1), tf.complex128))
	# Qco = params_values['Q1'] * tf.reduce_mean(tf.abs(dEc)**2) + params_values['Q0'] 
	# Q = tf.tile(tf.expand_dims(tf.diag([Qco, Qco, 1e-22]), 0), [n_pix, 1, 1])

	# # contrast = np.mean(self.If, 0)
	# # R = tf.tile([self.params_values['R0'] / exp_time**2 + self.params_values['R1'] / exp_time * contrast], self.n_pix)
	# # R = tf.expand_dims(tf.expand_dims(R, -1), -1)

	# Enp_old_expand = tf.expand_dims(Enp_old, -1)
	# Iinco_old_expand = tf.expand_dims(Iinco_old, -1)
	# x_old = tf.concat([tf.real(Enp_old_expand), tf.imag(Enp_old_expand), Iinco_old_expand], axis=1)
	# dx = tf.concat([tf.real(dEc), tf.imag(dEc), tf.zeros_like(dEc, dtype=tf.float64)], axis=1)

	x_new0 = x_old + dx
	P_new0 = P_old + Q

	R = params_values['R0'] / time**2 + params_values['R1'] / time * tf.maximum(tf.abs(y), 1e-10*np.ones(n_pix))
	R = tf.expand_dims(tf.expand_dims(R, -1), -1)

	H = tf.concat([tf.expand_dims(x_new0[:, 0], -1), 
					tf.expand_dims(x_new0[:, 1], -1),
					tf.ones([n_pix, 1], dtype=tf.float64)], 1)
	H = tf.expand_dims(H, 1)
	y_new0 = x_new0[:, 0]**2 + x_new0[:, 1]**2 + x_new0[:, 2]

	S = R + tf.matmul(tf.matmul(H, P_new0), tf.transpose(H, [0, 2, 1]))
	K = tf.matmul(tf.matmul(P_new0, tf.transpose(H, [0, 2, 1])), tf.matrix_inverse(S))
	x_new1 = x_new0 + tf.squeeze(tf.matmul(K, tf.expand_dims(tf.expand_dims(y-y_new0, -1), -1)), -1)
	P_new1 = P_new0 - tf.matmul(tf.matmul(K, H), P_new0)
	

	for k in range(iterations):
		H = tf.concat([tf.expand_dims(x_new1[:, 0], -1), 
			tf.expand_dims(x_new1[:, 1], -1),
			tf.ones([n_pix, 1], dtype=tf.float64)], 1)
		H = tf.expand_dims(H, 1)
		y_new1 = x_new1[:, 0]**2 + x_new1[:, 1]**2 + x_new1[:, 2]
		S = R + tf.matmul(tf.matmul(H, P_new0), tf.transpose(H, [0, 2, 1]))
		K = tf.matmul(tf.matmul(P_new0, tf.transpose(H, [0, 2, 1])), tf.matrix_inverse(S))
		x_new1 = x_new0 + tf.squeeze(tf.matmul(K, tf.expand_dims(tf.expand_dims(y-y_new1, -1), -1)-tf.matmul(H, tf.expand_dims(x_new0-x_new1, -1))), -1)
		P_new1 = P_new0 - tf.matmul(tf.matmul(K, H), P_new0)

	return x_new1, P_new1

class SMPC:
	def __init__(self, params_values, iterations=5):
		self.params_values = params_values

		G1 = np.squeeze(params_values['G1'])
		G2 = np.squeeze(params_values['G2'])
		Q0 = params_values['Q0']
		Q1 = params_values['Q1']
		R0 = params_values['R0']
		R1 = params_values['R1']
		R2 = 4 * params_values['Q1']

		self.n_act = G1.shape[1] # number of active actuators on the DM
		self.n_pix = G1.shape[0] # number of pixels in the dark hole

		self.u1c = tf.Variable(np.zeros(self.n_act), trainable=True, dtype=tf.float64)
		self.u2c = tf.Variable(np.zeros(self.n_act), trainable=True, dtype=tf.float64)
		self.u1c_next = tf.Variable(np.zeros(self.n_act), trainable=True, dtype=tf.float64)
		self.u2c_next = tf.Variable(np.zeros(self.n_act), trainable=True, dtype=tf.float64)

		self.noise = tf.placeholder(tf.float64, shape=(self.n_pix, 3, 1))
		self.noise2 = tf.placeholder(tf.float64, shape=(self.n_pix, ))
		self.noise3 = tf.placeholder(tf.float64, shape=(self.n_pix, 3, 1))
		self.noise4 = tf.placeholder(tf.float64, shape=(self.n_pix, ))
		self.Enp_old = tf.placeholder(tf.complex128, shape=(self.n_pix,))
		self.Iinco_old = tf.placeholder(tf.float64, shape=(self.n_pix,))
		self.P_old = tf.placeholder(tf.float64, shape=(self.n_pix, 3, 3))
		self.t = tf.placeholder(tf.float64, shape=())
		self.lr = tf.placeholder(tf.float64, shape=())
		self.beta = tf.placeholder(tf.float64, shape=())
		self.gamma = tf.placeholder(tf.float64, shape=())

		# generate the random images
		dEc = tf.matmul(G1, tf.cast(tf.expand_dims(self.u1c, -1), tf.complex128)) +\
			 tf.matmul(G2, tf.cast(tf.expand_dims(self.u2c, -1), tf.complex128))
		Qco = self.params_values['Q1'] * tf.reduce_mean(tf.abs(dEc)**2) + self.params_values['Q0'] 
		Q = tf.tile(tf.expand_dims(tf.diag([Qco, Qco, 1e-22]), 0), [self.n_pix, 1, 1])
		Enp_old_expand = tf.expand_dims(self.Enp_old, -1)
		Iinco_old_expand = tf.expand_dims(self.Iinco_old, -1)
		x_old = tf.concat([tf.real(Enp_old_expand), tf.imag(Enp_old_expand), Iinco_old_expand], axis=1)
		dx = tf.concat([tf.real(dEc), tf.imag(dEc), tf.zeros_like(dEc, dtype=tf.float64)], axis=1)

		dEc_next = tf.matmul(G1, tf.cast(tf.expand_dims(self.u1c_next, -1), tf.complex128)) +\
			 tf.matmul(G2, tf.cast(tf.expand_dims(self.u2c_next, -1), tf.complex128))
		Qco_next = self.params_values['Q1'] * tf.reduce_mean(tf.abs(dEc_next)**2) + self.params_values['Q0'] 
		Q_next = tf.tile(tf.expand_dims(tf.diag([Qco_next, Qco_next, 1e-22]), 0), [self.n_pix, 1, 1])
		dx_next = tf.concat([tf.real(dEc), tf.imag(dEc), tf.zeros_like(dEc, dtype=tf.float64)], axis=1)


		x_new0 = x_old + dx
		P_new0 = self.P_old + Q

		x_new0_noise = x_new0 + tf.squeeze(tf.matmul(tf.cholesky(P_new0), self.noise), -1)
		y = x_new0_noise[:, 0]**2 + x_new0_noise[:, 1]**2 + x_new0_noise[:, 2]
		R = self.params_values['R0'] / self.t**2 + self.params_values['R1'] / self.t * tf.maximum(tf.abs(y), 1e-10*np.ones(self.n_pix))
		y = y + self.noise2 * tf.sqrt(R)

		x_new0_next = x_new0_noise + dx_next
		x_new0_next_noise = x_new0_next + tf.squeeze(tf.matmul(tf.cholesky(Q_next), self.noise3), -1)
		y_next = x_new0_next_noise[:, 0]**2 + x_new0_next_noise[:, 1]**2 + x_new0_next_noise[:, 2]
		R_next = self.params_values['R0'] / self.t**2 + self.params_values['R1'] / self.t * tf.maximum(tf.abs(y_next), 1e-10*np.ones(self.n_pix))
		y_next = y_next + self.noise4 * tf.sqrt(R_next)



		x_new1, P_new1 = IEKF(params_values, self.n_pix, y, x_old, dx, Q, self.P_old, self.t, iterations)
		x_new2, P_new2 = IEKF(params_values, self.n_pix, y_next, x_new1, dx_next, Q_next, P_new1, self.t, iterations)


		# dEc = tf.matmul(G1, tf.cast(tf.expand_dims(self.u1c, -1), tf.complex128)) +\
		# 	 tf.matmul(G2, tf.cast(tf.expand_dims(self.u2c, -1), tf.complex128))
		# Qco = self.params_values['Q1'] * tf.reduce_mean(tf.abs(dEc)**2) + self.params_values['Q0'] 
		# Q = tf.tile(tf.expand_dims(tf.diag([Qco, Qco, 1e-22]), 0), [self.n_pix, 1, 1])

		# # contrast = np.mean(self.If, 0)
		# # R = tf.tile([self.params_values['R0'] / exp_time**2 + self.params_values['R1'] / exp_time * contrast], self.n_pix)
		# # R = tf.expand_dims(tf.expand_dims(R, -1), -1)

		# Enp_old_expand = tf.expand_dims(self.Enp_old, -1)
		# Iinco_old_expand = tf.expand_dims(self.Iinco_old, -1)
		# x_old = tf.concat([tf.real(Enp_old_expand), tf.imag(Enp_old_expand), Iinco_old_expand], axis=1)
		# dx = tf.concat([tf.real(dEc), tf.imag(dEc), tf.zeros_like(dEc, dtype=tf.float64)], axis=1)

		# x_new0 = x_old + dx
		# P_new0 = self.P_old + Q

		# x_new0_noise = x_new0 + tf.squeeze(tf.matmul(tf.cholesky(P_new0), self.noise), -1)
		# y = x_new0_noise[:, 0]**2 + x_new0_noise[:, 1]**2 + x_new0_noise[:, 2]

		# R = self.params_values['R0'] / self.t**2 + self.params_values['R1'] / self.t * tf.maximum(tf.abs(y), 1e-10*np.ones(self.n_pix))

		# y = y + self.noise2 * tf.sqrt(R)

		# R = self.params_values['R0'] / self.t**2 + self.params_values['R1'] / self.t * tf.maximum(tf.abs(y), 1e-10*np.ones(self.n_pix))
		# R = tf.expand_dims(tf.expand_dims(R, -1), -1)

		# H = tf.concat([tf.expand_dims(x_new0[:, 0], -1), 
		# 				tf.expand_dims(x_new0[:, 1], -1),
		# 				tf.ones([self.n_pix, 1], dtype=tf.float64)], 1)
		# H = tf.expand_dims(H, 1)
		# y_new0 = x_new0[:, 0]**2 + x_new0[:, 1]**2 + x_new0[:, 2]

		# S = R + tf.matmul(tf.matmul(H, P_new0), tf.transpose(H, [0, 2, 1]))
		# K = tf.matmul(tf.matmul(P_new0, tf.transpose(H, [0, 2, 1])), tf.matrix_inverse(S))
		# x_new1 = x_new0 + tf.squeeze(tf.matmul(K, tf.expand_dims(tf.expand_dims(y-y_new0, -1), -1)), -1)
		# P_new1 = P_new0 - tf.matmul(tf.matmul(K, H), P_new0)
		

		# for k in range(iterations):
		# 	H = tf.concat([tf.expand_dims(x_new1[:, 0], -1), 
		# 		tf.expand_dims(x_new1[:, 1], -1),
		# 		tf.ones([self.n_pix, 1], dtype=tf.float64)], 1)
		# 	H = tf.expand_dims(H, 1)
		# 	y_new1 = x_new1[:, 0]**2 + x_new1[:, 1]**2 + x_new1[:, 2]
		# 	S = R + tf.matmul(tf.matmul(H, P_new0), tf.transpose(H, [0, 2, 1]))
		# 	K = tf.matmul(tf.matmul(P_new0, tf.transpose(H, [0, 2, 1])), tf.matrix_inverse(S))
		# 	x_new1 = x_new0 + tf.squeeze(tf.matmul(K, tf.expand_dims(tf.expand_dims(y-y_new1, -1), -1)-tf.matmul(H, tf.expand_dims(x_new0-x_new1, -1))), -1)
		# 	P_new1 = P_new0 - tf.matmul(tf.matmul(K, H), P_new0)



		# self.cost = tf.reduce_mean(x_new0[:, 0]**2 + x_new0[:, 1]**2) + \
		# 			tf.reduce_mean(tf.trace(Q)) + self.beta*tf.reduce_mean(tf.trace(P_new1))
		self.cost = tf.reduce_mean(x_new1[:, 0]**2 + x_new1[:, 1]**2) + tf.reduce_mean(tf.trace(P_new1)) + \
					self.beta * tf.reduce_mean(x_new2[:, 0]**2 + x_new2[:, 1]**2) + self.beta * tf.reduce_mean(tf.trace(P_new2))

		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.99, beta2=0.999).minimize(self.cost, var_list=[self.u1c, self.u2c,
																															self.u1c_next, self.u2c_next])
		self.init = tf.global_variables_initializer()

	def Control(self, u1c_values, u2c_values, u1c_next_values, u2c_next_values, Ef_est, Iinco_est, P_est, time, beta=1, rate=1e-3, Nitr=100, print_flag=False):
		u1c_values = u1c_values.reshape((self.n_act, ))
		u2c_values = u2c_values.reshape((self.n_act, ))
		Ef_est = Ef_est.reshape((self.n_pix, ))
		Iinco_est = Iinco_est.reshape((self.n_pix, ))
		with tf.Session() as sess:
			sess.run(self.init)
			self.u1c.load(u1c_values, sess)
			self.u2c.load(u2c_values, sess)
			self.u1c_next.load(u1c_next_values, sess)
			self.u2c_next.load(u2c_next_values, sess)
			if print_flag:
				cost_values = sess.run(self.cost, feed_dict={
									self.t: time,
									self.Enp_old: Ef_est,
									self.Iinco_old: Iinco_est,
									self.P_old: P_est,
									self.noise: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise2: np.random.normal(size=(self.n_pix, )), 
									self.noise3: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise4: np.random.normal(size=(self.n_pix, )), 
									self.beta: 0,#beta,
									self.lr: rate})
				cost_values2 = sess.run(self.cost, feed_dict={
									self.t: time,
									self.Enp_old: Ef_est,
									self.Iinco_old: Iinco_est,
									self.P_old: P_est,
									self.noise: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise2: np.random.normal(size=(self.n_pix, )), 
									self.noise3: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise4: np.random.normal(size=(self.n_pix, )), 
									self.beta: 1,#beta,
									self.lr: rate})

				print('SGD #: {}, cost1: {}, cost2: {}'.format(0, cost_values, cost_values2-cost_values))
			for i in range(int(Nitr)):
				sess.run(self.optimizer, feed_dict={
									self.t: time,
									self.Enp_old: Ef_est,
									self.Iinco_old: Iinco_est,
									self.P_old: P_est,
									self.noise: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise2: np.random.normal(size=(self.n_pix, )), 
									self.noise3: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise4: np.random.normal(size=(self.n_pix, )), 
									self.beta: beta,
									self.lr: rate})
				if print_flag and ((i+1)%200==0 or (i+1)==Nitr):
					cost_values = sess.run(self.cost, feed_dict={
									self.t: time,
									self.Enp_old: Ef_est,
									self.Iinco_old: Iinco_est,
									self.P_old: P_est,
									self.noise: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise2: np.random.normal(size=(self.n_pix, )), 
									self.noise3: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise4: np.random.normal(size=(self.n_pix, )), 
									self.beta: 0,#beta,
									self.lr: rate})
					cost_values2 = sess.run(self.cost, feed_dict={
									self.t: time,
									self.Enp_old: Ef_est,
									self.Iinco_old: Iinco_est,
									self.P_old: P_est,
									self.noise: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise2: np.random.normal(size=(self.n_pix, )), 
									self.noise3: np.random.normal(size=(self.n_pix, 3, 1)),
									self.noise4: np.random.normal(size=(self.n_pix, )), 
									self.beta: 1,#beta,
									self.lr: rate})

					print('SGD #: {}, cost1: {}, cost2: {}'.format(i, cost_values, cost_values2-cost_values))

			u1c_values = sess.run(self.u1c)
			u2c_values = sess.run(self.u2c)
			u1c_next_values = sess.run(self.u1c_next)
			u2c_next_values = sess.run(self.u2c_next)
		return u1c_values, u2c_values, u1c_next_values, u2c_next_values