"""
created on Mon Apr. 22, 2019

@author: He Sun, Princeton University

Sensing policies (DM probing policies) for the focal plane wavefront control.

"""

import numpy as np
import scipy as sp
import tensorflow as tf

class Optimal_probe:
	def __init__(self, params_values, img_number, pair_wise=True):
		self.params_values = params_values

		G1 = np.squeeze(params_values['G1'])
		G2 = np.squeeze(params_values['G2'])
		Q0 = params_values['Q0']
		Q1 = params_values['Q1']
		R0 = params_values['R0']
		R1 = params_values['R1']
		R2 = 4 * params_values['Q1']

		self.img_number = img_number
		self.n_act = G1.shape[1] # number of active actuators on the DM
		self.n_pix = G1.shape[0] # number of pixels in the dark hole
		self.n_pair = self.img_number // 2

		self.u1p = tf.Variable(np.zeros((self.n_pair, self.n_act)), trainable=True, dtype=tf.float64)
		self.t = tf.placeholder(tf.float64, shape=())
		self.contrast = tf.placeholder(tf.float64, shape=())
		self.lr = tf.placeholder(tf.float64, shape=())
		self.beta = tf.placeholder(tf.float64, shape=())
		self.gamma = tf.placeholder(tf.float64, shape=())

		u1p = tf.cast(self.u1p, tf.complex128)
		probe = tf.tensordot(u1p, G1, axes=[1, 1])

		probe_expand = tf.expand_dims(probe, 2)
		H = tf.concat([4*tf.real(probe_expand), 4*tf.imag(probe_expand)], axis=2)
		H = tf.transpose(H, [1, 0, 2])
		HtH = tf.matmul(tf.transpose(H, perm=[0, 2, 1]), H) + tf.diag(np.array([1e-18, 1e-18], dtype=np.float64))
		Hinv = tf.matmul(tf.linalg.inv(HtH), tf.transpose(H, perm=[0, 2, 1]))
		probe_contrast = tf.reduce_mean(tf.abs(probe)**2, 1)

		# probe_contrast_opt = tf.sqrt(R0/(self.t)**2/R2 + R1/self.t/R2*self.contrast + self.contrast**2)
		# probe_contrast_opt = tf.minimum(tf.sqrt(self.contrast * 1e-5), 5e-4)
		probe_contrast_opt = tf.sqrt(R0/(self.t)**2/R2 + (R1/self.t+4*Q0)/R2*self.contrast)
		probe_contrast_opt_sqrt = tf.sqrt(probe_contrast_opt)

		probe_dot = tf.real(tf.conj(probe[0::2, :]) * probe[1::2, :]) / (tf.abs(probe[0::2, :]) * tf.abs(probe[1::2, :]))
		
		H_det = tf.imag(tf.conj(probe[0::2, :]) * probe[1::2, :]) / (tf.abs(probe[0::2, :]) * tf.abs(probe[1::2, :]))


		# R = 2 * (R0/(self.t)**2 * tf.eye(self.n_pair, batch_shape=[self.n_pix], dtype=tf.float64) + \
		# 	R1*tf.linalg.diag(self.contrast+probe_contrast)/self.t + \
		# 	R2 * tf.linalg.diag(self.contrast**2+probe_contrast**2+6*self.contrast*probe_contrast))

		# R = 2 * (R0/(self.t)**2 * tf.eye(self.n_pair, batch_shape=[self.n_pix], dtype=tf.float64) + \
		# 	R1*(self.contrast+probe_contrast_opt)*tf.eye(self.n_pair, dtype=tf.float64)/self.t + \
		# 	R2 *(self.contrast**2+probe_contrast_opt**2+6*self.contrast*probe_contrast_opt) * tf.eye(self.n_pair, dtype=tf.float64))

		R = 0.125 * (R0/(self.t)**2 + (R1/self.t+4*Q0) * (self.contrast + tf.abs(probe)**2) + R2 * tf.abs(probe)**2 * (self.contrast + tf.abs(probe)**2)) / tf.abs(probe)**2
		# R = tf.matrix_diag(tf.transpose(R, [1, 0]))
		R = R[0::2, :]*R[1::2, :]
		# self.R = R
		# self.H_det = H_det
		

		# P_est = tf.matmul(tf.matmul(Hinv, R), tf.transpose(Hinv, perm=[0, 2, 1]))
		# self.cost = tf.reduce_mean(tf.linalg.logdet(P_est)) + self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)
		self.cost = tf.reduce_mean(tf.log(R / H_det**2 / self.contrast)) + \
					self.beta * tf.reduce_mean(tf.abs(probe_dot)) + \
					self.gamma * tf.reduce_mean(tf.abs(self.u1p)**2)
		# self.cost = tf.reduce_mean(tf.linalg.logdet(P_est)) + self.beta*tf.log(tf.reduce_mean(tf.abs(self.u1p)))
		# self.cost = tf.reduce_mean(tf.linalg.logdet(P_est)) + \
		# 			self.beta * tf.reduce_mean((tf.abs(probe)/probe_contrast_opt_sqrt - 1)**2)
		# self.cost = tf.reduce_mean(tf.abs(probe_dot)) + tf.reduce_mean((tf.abs(probe)**2/probe_contrast_opt - 1)**2) + \
		# 			self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)
		# self.cost = tf.reduce_mean(tf.abs(probe_dot)) + 0.1 * tf.reduce_mean((probe_contrast_opt/tf.abs(probe)**2 - 1)**2) + \
		# 			self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)
		# self.cost = tf.reduce_mean(tf.abs(probe_dot)) + 1e9*tf.reduce_mean((tf.abs(probe)**2-probe_contrast_opt)**2) + \
		# 			self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)
		# self.cost = tf.reduce_mean(tf.abs(probe_dot)) + 0.1*tf.reduce_mean((tf.log(tf.abs(probe)**2) - tf.log(probe_contrast_opt))**2) + \
		# 			self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)
		# self.cost = tf.reduce_mean(tf.abs(probe_dot)) + 0.1*tf.reduce_mean((tf.log(tf.abs(probe)**2) - tf.log(probe_contrast_opt))**2) + \
		# 			self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)
		# self.cost = tf.reduce_mean(tf.abs(probe_dot)) + 0.1*tf.reduce_mean(0.5*(tf.abs(probe)**2/probe_contrast_opt + probe_contrast_opt/tf.abs(probe)**2)) + \
		# 			self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)
		# self.cost = tf.reduce_mean(tf.abs(probe_dot)) + \
		# 			0.1*tf.reduce_mean(0.125*(tf.abs(probe)**2/self.contrast*R2 + (R1+R0/self.contrast)/tf.abs(probe)**2)) + \
		# 			self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)
		# self.cost = tf.reduce_mean(tf.abs(probe_dot)) + 0.01 * tf.reduce_mean(((probe_contrast_opt-tf.abs(probe)**2)/(probe_contrast_opt+tf.abs(probe)**2))**2) + \
		# 			self.beta * tf.reduce_mean(tf.abs(self.u1p)**2)


		# self.cost1 = tf.reduce_mean((tf.abs(angle_diff) - 0.5*np.pi)**2)#tf.reduce_mean(tf.abs(probe_dot))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.99, beta2=0.999).minimize(self.cost, var_list=[self.u1p])
		# self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.cost, var_list=[self.u1p])
		self.init = tf.global_variables_initializer()

	def Probe_command(self, u1p_values, time, contrast, rate=1e-3, beta=1, gamma=0.0, Nitr=100, print_flag=False):
		u1p_values = u1p_values.reshape((self.n_pair, self.n_act))
		with tf.Session() as sess:
			sess.run(self.init)
			self.u1p.load(u1p_values, sess)
			if print_flag:
				cost_values = sess.run(self.cost, feed_dict={
									self.t: time,
									self.contrast: contrast,
									self.lr: rate,
									self.beta: 0.,
									self.gamma: 0.})
				cost_values2 = sess.run(self.cost, feed_dict={
									self.t: time,
									self.contrast: contrast,
									self.lr: rate,
									self.beta: beta,
									self.gamma: 0.})
				# cost_values3 = sess.run(self.cost1, feed_dict={
				# 					self.t: time,
				# 					self.contrast: contrast,
				# 					self.lr: rate,
				# 					self.beta: 0})
				# print('SGD #: {}, total: {:.3f}, cost1: {:.3f}, cost2: {:.3e}, regularization: {:.3f}'.format(0, 
				# 					cost_values2, cost_values3, cost_values-cost_values3, cost_values2-cost_values))
				print('SGD #: {}, total: {:.3f}, regularization: {:.3e}'.format(0, 
									cost_values, cost_values2-cost_values))
			for i in range(int(Nitr)):
				sess.run(self.optimizer, feed_dict={
									self.t: time,
									self.contrast: contrast,
									self.lr: rate,
									self.beta: beta,
									self.gamma: gamma})
				if print_flag and ((i+1)%200==0 or (i+1)==Nitr):
					cost_values = sess.run(self.cost, feed_dict={
									self.t: time,
									self.contrast: contrast,
									self.lr: rate,
									self.beta: 0.,
									self.gamma: 0.})
					cost_values2 = sess.run(self.cost, feed_dict={
									self.t: time,
									self.contrast: contrast,
									self.lr: rate,
									self.beta: beta,
									self.gamma: 0.})
					# cost_values3 = sess.run(self.cost1, feed_dict={
					# 				self.t: time,
					# 				self.contrast: contrast,
					# 				self.lr: rate,
					# 				self.beta: 0})
					# print('SGD #: {}, total: {:.3f}, cost1: {:.3f}, cost2: {:.3e}, regularization: {:.3f}'.format(i, 
					# 					cost_values2, cost_values3, cost_values-cost_values3, cost_values2-cost_values))
					print('SGD #: {}, total: {:.3f}, regularization: {:.3e}'.format(i, 
									cost_values, cost_values2-cost_values))
			u1p_values = sess.run(self.u1p)
		return u1p_values, cost_values

class Empirical_probe:
	def __init__(self, model, params_values, img_number, pair_wise=True, probe_area=[1, 17, -17, 17], method='rot'):
		self.params_values = params_values
		self.img_number = img_number
		self.probe_area = probe_area
		self.model = model
		self.pair_wise = True
		self.method = method

		if self.pair_wise:
			self.pair_num = self.img_number // 2
			self.omega = 0.5 * np.pi / self.pair_num
		else:
			self.omega = 0.5 * np.pi / self.img_number

		dx = model.widthDM / model.DMmesh[1]
		dy = model.widthDM / model.DMmesh[0]
		xs = np.arange(-model.DMmesh[1]/2+0.5, model.DMmesh[1]/2+0.5) * dx
		ys = np.arange(-model.DMmesh[0]/2+0.5, model.DMmesh[0]/2+0.5) * dy
		[self.XS, self.YS] = np.meshgrid(xs, ys)

		self.mx = (probe_area[1] - probe_area[0]) / self.model.SPwidth
		self.my = (probe_area[3] - probe_area[2]) / self.model.SPwidth
		self.wx = (probe_area[1] + probe_area[0]) / self.model.SPwidth

	def Probe_command(self, contrast, itr, index=1, R_coef=[]):
		if len(R_coef) == 3:
			# an optimal law to determine the probe contrast based on noise analysis
			# probeContrast = np.sqrt(R_coef[0]/R_coef[2] + R_coef[1]/R_coef[2]*contrast + contrast**2)
			probeContrast = np.sqrt(R_coef[0]/R_coef[2] + R_coef[1]/R_coef[2]*contrast)
		else:
			# a heuristic law to determine the probe contrast
			probeContrast = np.min([np.sqrt(np.mean(contrast) * 1e-5), 5e-4])
		print('The chosen probe contrast is {}'.format(probeContrast))

		if self.pair_wise:
			offsets = self.omega * itr + np.arange(0, self.pair_num) * np.pi / self.pair_num
			count = int(self.pair_num)
		else:
			offsets = self.omega * itr + np.arange(0, self.img_number) * np.pi / self.img_number
			count = int(self.img_number)

		u_p = np.empty((self.model.Nact, self.model.Nact, count), dtype=float)
		for k in range(count):
			if self.method.lower() == 'rot':
				probeSP = self.ProbeShape(offsets[k], probeContrast)
			if self.method.lower() == 'rot_alter':
				probeSP = self.ProbeShape(offsets[k], probeContrast, axis=itr%2)
			elif self.method.lower() == 'alter':
				probeSP = self.ProbeShape(offsets[k], probeContrast, axis=k%2)

			marginWidth = (self.model.widthDM - self.model.SPwidth) / 2
			marginNpixel = int(np.round(marginWidth / self.model.widthDM * self.model.DMmesh[0]))
			probeSPresized = sp.misc.imresize(probeSP, (int(self.model.DMmesh[0] - 2 * marginNpixel), int(self.model.DMmesh[1] - 2 * marginNpixel)), interp='bilinear', mode='F')
			probeDM = np.zeros(self.model.DMmesh)
			probeDM[marginNpixel:-marginNpixel, marginNpixel:-marginNpixel] = probeSPresized
			u_p[:, :, k] = self.Height_to_voltage(probeDM, index)

		contrast_p_now = np.mean(np.abs(np.matmul(np.squeeze(self.params_values['G1']), u_p[self.model.DMind1, self.model.DMind2, :]))**2)
		u_p = np.sqrt(probeContrast/contrast_p_now) * u_p
		
		return u_p

	def ProbeShape(self, offset, probeContrast, axis=0):
		
		SincAmp = np.mean(self.model.wavelength) * np.sqrt(probeContrast) * np.sqrt(2 * np.pi)
		if self.model.coronagraph_type.lower() == 'splc':
			surf = SincAmp * np.sinc(self.mx * self.XS) * np.sinc(self.my * self.YS + 2 * np.pi) * np.cos(np.pi * self.wx * self.XS + offset)
		elif self.model.coronagraph_type.lower() == 'vortex':
			if axis == 0:
				surf = SincAmp * np.sinc(self.mx * self.XS) * np.sinc(self.my * self.YS) * np.cos(np.pi * self.wx * self.XS + offset)
			elif axis == 1:
				surf = SincAmp * np.sinc(self.mx * self.YS) * np.sinc(self.my * self.XS) * np.cos(np.pi * self.wx * self.YS + offset)
		surf = np.pi * surf
		return surf

	def Height_to_voltage(self, surf, index):
		if index == 1:
			gain = self.model.DM1gain
		elif index == 2:
			gain = self.model.DM2gain
		else:
			print('We only have two DMs!')
		command = self.model.DMstop * sp.misc.imresize(surf, gain.shape, interp='bicubic', mode='F') / gain
		for k in range(5):
			currentSurf = self.model.DMsurf(command, index)
			surfBias = surf - currentSurf[26:-26, 26:-26]
			command += sp.misc.imresize(surfBias, gain.shape, interp='bicubic', mode='F') / gain
			command *= self.model.DMstop
		return command