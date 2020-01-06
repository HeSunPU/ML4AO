"""
created on Sun Apr. 21, 2019

@author: He Sun, Princeton University

model of a simple vortex coronagraph
adapted from the A.J.'s FALCO codes

"""
import os
import numpy as np
import scipy.io as spio
import scipy.signal as spsig
import scipy.interpolate as spint
import numpy.fft as fft
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform as transform
import imp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
plt.ion()

def imresize(img, new_shape):
	return 255.0 * transform.resize(img, new_shape, order=3, anti_aliasing=True, mode='reflect')

class Optical_Model:
	def __init__(self, wavelength, DM1gain, DM2gain, wfe=False, error_maps=[]):
		# target information
		self.wavelength = wavelength
		self.normalization = np.ones(wavelength.shape)

		# DM information
		self.DMmesh = [442, 442]
		self.Nact = 34
		self.pitchDM = 301e-6
		self.widthDM = self.pitchDM * self.Nact
		self.DM1gain = DM1gain
		self.DM2gain = DM2gain
		self.zDM1toDM2 = 0.23
		self.voltageLimit = 50
		self.DMstop = np.zeros((self.Nact, self.Nact))
		for m in range(self.Nact):
			for n in range(self.Nact):
				if np.linalg.norm([m-self.Nact/2+0.5, n-self.Nact/2+0.5]) < self.Nact/2+0.5:
					self.DMstop[m, n] = 1
		self.DMind1, self.DMind2 = np.nonzero(self.DMstop)
		sigma = 1.125 # influence function width in # of actuators
		width = 5 # influence function width in # of actuators
		dx = self.Nact * self.pitchDM / self.DMmesh[1]
		dy = self.Nact * self.pitchDM / self.DMmesh[0]
		sigmax = sigma * self.pitchDM
		sigmay = sigma * self.pitchDM
		widthx = np.ceil(width * self.pitchDM / dx)
		widthy = np.ceil(width * self.pitchDM / dy)
		if widthx%2 == 1:
			xs = np.arange(-np.floor(widthx/2), np.floor(widthx/2)+1, 1) * dx
		else:
			xs = np.arange(-widthx/2+1, widthx/2+1, 1) * dx
		if widthy%2 == 1:
			ys = np.arange(-np.floor(widthy/2), np.floor(widthy/2)+1, 1) * dy
		else:
			ys = np.arange(-widthy/2+1, widthy/2+1, 1) * dy
		[YS, XS] = np.meshgrid(ys, xs)
		self.infFunc = np.exp(-4 * np.log(2) * ((XS/sigmax)**2 + (YS/sigmay)**2))

		# coronagraph information
		self.coronagraph_type = 'vortex'
		self.SPwidth = 0.01
		self.Nsp = 432
		self.focalLength = 1.1642
		self.apertureWidth = self.SPwidth
		self.Naperture = self.Nsp
		inVal = 0.3
		outVal = 5
		lambdaOverD = 4
		vortexD = 350
		charge = 6
		self.Nfpm = lambdaOverD * vortexD
		self.lyotWidth = self.SPwidth
		self.Nlyot = self.Nsp
		self.SPshape = fits.getdata('./masks/vortex_apodizer.fits', ext=0)
		self.LyotStop = fits.getdata('./masks/vortex_lyot.fits', ext=0)
		self.windowMask1 = fits.getdata('./masks/vortex_window_mask1.fits', ext=0)
		self.windowMask2 = fits.getdata('./masks/vortex_window_mask2.fits', ext=0)
		self.FPmask = self.Gen_Vortex_mask(charge, self.Nfpm)
		self.SPshape = imresize(self.SPshape, [self.Nsp, self.Nsp]) / 255
		self.LyotStop = imresize(self.LyotStop, [self.Nlyot, self.Nlyot]) / 255
		# wavefront error information
		self.wfe = wfe
		if wfe:
			if error_maps:
				self.DM1_error = imresize(error_maps[0], [494, 494]) / 255
				self.DM2_error = imresize(error_maps[1], [494, 494]) / 255
				self.SP_error = imresize(error_maps[2], [self.Nsp, self.Nsp]) / 255
			else:
				self.DM1_error = 2 * imresize(spio.loadmat('./err_maps/PSD_DM1.mat')['PSD_DM1'], [494, 494]) / 255
				self.DM2_error = 2 * imresize(spio.loadmat('./err_maps/PSD_DM2.mat')['PSD_DM2'], [494, 494]) / 255
				self.SP_error = 2 * imresize(spio.loadmat('./err_maps/PSD_SP.mat')['PSD_SP'], [self.Nsp, self.Nsp]) / 255

		# camera information
		self.camera_pitch = 4.54e-6
		self.camera_binXi = 8# 4
		self.camera_binEta = 8# 4
		self.camera_Nxi = 49# 99
		self.camera_Neta = 41# 83

		self.Ein = np.ones((self.Naperture, self.Naperture))

		# Fresnel propagation from DM1 to DM2 (pupil1 to pupil2)
		Ndm = 494
		L = self.widthDM * 3 * 494 / self.DMmesh[0]
		M = Ndm * 3
		dx =  L / M
		fx = np.arange(-1/(2*dx), 1/(2*dx), 1/L)
		[FX, FY] = np.meshgrid(fx, fx)
		self.p1_to_p2 = np.empty((M, M, len(self.wavelength)), dtype=np.complex)
		for k in range(len(self.wavelength)):
			wl = self.wavelength[k]
			self.p1_to_p2[:, :, k] = fft.fftshift(np.exp(-1j * np.pi * wl * self.zDM1toDM2 * (FX**2 + FY**2)))
		# Fresnel propagation from DM2 back to DM1 (pupil2 to pupil3)
		self.p2_to_p3 = np.empty((M, M, len(self.wavelength)), dtype=np.complex)
		for k in range(len(self.wavelength)):
			wl = self.wavelength[k]
			self.p2_to_p3[:, :, k] = fft.fftshift(np.exp(-1j * np.pi * wl * (-self.zDM1toDM2) * (FX**2 + FY**2)))
		# Fourier transform from DM1 to the focal plane mask (pupil3 to focal1)
		xp = np.arange((-self.Nsp/2+0.5), (self.Nsp/2+0.5)) / vortexD
		xf1_1 = np.arange((-self.Nfpm/2+0.5), (self.Nfpm/2+0.5)) / lambdaOverD
		xf1_2 = np.arange((-self.Nfpm/2+0.5), (self.Nfpm/2+0.5)) * 2 * outVal / self.Nfpm

		self.p3_to_f1_scaler1 = 1. / (vortexD * lambdaOverD)
		self.p3_to_f1_pre1 = np.exp(-2 * np.pi * 1j * np.outer(xf1_1, xp))
		self.p3_to_f1_post1 = np.exp(-2 * np.pi * 1j * np.outer(xp, xf1_1))

		self.p3_to_f1_scaler2 = 2. * outVal / (vortexD * self.Nfpm)
		self.p3_to_f1_pre2 = np.exp(-2 * np.pi * 1j * np.outer(xf1_2, xp))
		self.p3_to_f1_post2 = np.exp(-2 * np.pi * 1j * np.outer(xp, xf1_2))

		# Fourier transform from the focal plane mask to the Lyot stop (focal1 to pupil4)
		self.f1_to_p4_pre1 = self.p3_to_f1_post1
		self.f1_to_p4_post1 = self.p3_to_f1_pre1
		self.f1_to_p4_scaler1 = self.p3_to_f1_scaler1

		self.f1_to_p4_pre2 = self.p3_to_f1_post2
		self.f1_to_p4_post2 = self.p3_to_f1_pre2
		self.f1_to_p4_scaler2 = self.p3_to_f1_scaler2

		# Fourier transform from the Lyot stop to camera (pupil4 to focal2)
		self.p4_to_f2_pre = np.empty((self.camera_Neta, self.Nlyot, len(self.wavelength)), dtype=np.complex)
		self.p4_to_f2_post = np.empty((self.Nlyot, self.camera_Nxi, len(self.wavelength)), dtype=np.complex)
		self.p4_to_f2_scaler = np.empty(len(self.wavelength), dtype=np.complex)
		dx = self.lyotWidth / self.Nlyot
		xp = np.arange((-self.Nlyot/2+0.5)*dx, (self.Nlyot/2+0.5)*dx, dx)
		dxi = self.camera_pitch * self.camera_binXi
		deta = self.camera_pitch * self.camera_binEta
		xis = np.arange(-(self.camera_Nxi-1)/2, (self.camera_Nxi-1)/2+1, 1) * dxi
		etas = np.arange(-(self.camera_Neta-1)/2, (self.camera_Neta-1)/2+1, 1) * deta
		for k in range(len(self.wavelength)):
			wl = self.wavelength[k]
			self.p4_to_f2_pre[:, :, k] = np.exp(-2 * np.pi * 1j * np.outer(etas, xp) / (wl * self.focalLength))
			self.p4_to_f2_post[:, :, k] = np.exp(-2 * np.pi * 1j * np.outer(xp, xis) / (wl * self.focalLength))
			self.p4_to_f2_scaler[k] = dx * np.sqrt(dxi * deta) * np.exp(2 * np.pi * 1j * self.focalLength / wl) / (1j * wl * self.focalLength)

		# define the computation graph for DM surface
		input_shape = (34, 34)
		kernel_shape = (65, 65)
		strides = (13, 13)
		output_shape = (kernel_shape[0] + strides[0] * (input_shape[0]-1),
		        kernel_shape[1] + strides[1] * (input_shape[1]-1))
		self.kernel = tf.placeholder(tf.float32, shape = kernel_shape)
		self.height_map = tf.placeholder(tf.float32, shape = input_shape)
		kernel_tensor = tf.reshape(self.kernel, [kernel_shape[0], kernel_shape[1], 1, 1])
		height_map_tensor = tf.reshape(self.height_map, [1, input_shape[0], input_shape[1], 1])
		self.DM_surface = tf.nn.conv2d_transpose(height_map_tensor, kernel_tensor, 
								output_shape=[1, output_shape[0], output_shape[1], 1],
		                       	strides=[1,strides[0],strides[1],1], 
		                       	padding='VALID')
		self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

		# recompute the normalization factors
		separation = 7
		minAngle = 0
		maxAngle = separation * 2 * np.pi
		gap = (maxAngle - minAngle) / self.Naperture
		phase = np.matmul(np.ones((self.Naperture, self.Naperture)), np.diag(np.arange(minAngle, maxAngle, gap)))
		Ein_offaxis = np.exp(1j * phase)
		Ef = self.Propagate(np.zeros((34, 34)), np.zeros((34, 34)), Ein=Ein_offaxis)
		If = np.abs(Ef)**2
		for k in range(len(self.normalization)):
			self.normalization[k] = np.max(If[:, :, k])

	def DMsurf(self, command_map, index):
		# calculate the DM surf shape shape given the 2D command map
		if index == 1:
			height_map = self.DM1gain * self.DMstop * command_map
		elif index == 2:
			height_map = self.DM2gain * self.DMstop * command_map
		else:
			print('We only have DM #1 or #2 available!!')

		surf = self.sess.run(self.DM_surface, feed_dict={self.kernel: self.infFunc.reshape((65, 65)),
													self.height_map: height_map})
		return surf.reshape((surf.shape[1], surf.shape[2]))

	def Propagate(self, command1, command2, Ein=None, to_pupil=False):
		# the propagation model compute the focal plane electric field after giving specific commands
		mirrorFactor = 2
		DM1surf = self.DMsurf(command1, 1)
		DM2surf = self.DMsurf(command2, 2)
		Ndm = DM1surf.shape[0]
		Npupil_pad = 3 * Ndm
		Nsp = self.Nsp
		if self.wfe:
			DM1surf += self.DM1_error
			DM2surf += self.DM2_error
		if Ein is None:
			Ein = self.Ein

		if to_pupil:
			pupil3_bb = np.empty((self.Nsp, self.Nsp, len(self.wavelength)), dtype=complex)
		else:
			focal2_bb = np.empty((self.camera_Neta, self.camera_Nxi, len(self.wavelength)), dtype=complex)
		
		for k in range(len(self.wavelength)):
			wl = self.wavelength[k]
			pupil1 = Ein
			pupil1 = np.pad(pupil1, (int((Ndm -self.Naperture)/2), int((Ndm -self.Naperture)/2)), 'constant')
			pupil1 = pupil1 * np.exp(2 * 1j * np.pi * mirrorFactor * DM1surf / wl)

			pupil1_pad = np.pad(pupil1, (Ndm, Ndm), 'constant')

			pupil2_pad = fft.ifftshift(fft.ifft2(fft.fft2(fft.fftshift(pupil1_pad)) * self.p1_to_p2[:, :, k]))
			pupil2_pad = pupil2_pad * np.exp(2 * 1j * np.pi * mirrorFactor * np.pad(DM2surf, (Ndm, Ndm), 'constant') / wl)

			pupil3_pad = fft.ifftshift(fft.ifft2(fft.fft2(fft.fftshift(pupil2_pad)) * self.p2_to_p3[:, :, k]))

			pupil3 = pupil3_pad[int(Npupil_pad/2-Nsp/2):int(Npupil_pad/2+Nsp/2), int(Npupil_pad/2-Nsp/2):int(Npupil_pad/2+Nsp/2)]
			pupil3m = pupil3 * self.SPshape

			if to_pupil:
				pupil3_bb[:, :, k] = pupil3
			else:
				focal1_1 = self.p3_to_f1_scaler1 * np.matmul(np.matmul(self.p3_to_f1_pre1, pupil3m), self.p3_to_f1_post1)
				focal1_2 = self.p3_to_f1_scaler2 * np.matmul(np.matmul(self.p3_to_f1_pre2, pupil3m), self.p3_to_f1_post2)
				
				focal1m_1 = focal1_1 * self.FPmask * (1-self.windowMask1)
				focal1m_2 = focal1_2 * self.FPmask * self.windowMask2

				pupil4_1 = self.f1_to_p4_scaler1 * np.matmul(np.matmul(self.f1_to_p4_pre1, focal1m_1), self.f1_to_p4_post1)
				pupil4_2 = self.f1_to_p4_scaler2 * np.matmul(np.matmul(self.f1_to_p4_pre2, focal1m_2), self.f1_to_p4_post2)

				pupil4 = pupil4_1 + pupil4_2

				pupil4m = pupil4 * self.LyotStop

				focal2 = self.p4_to_f2_scaler[k] * np.matmul(np.matmul(self.p4_to_f2_pre[:, :, k], pupil4m), self.p4_to_f2_post[:, :, k])

				focal2_bb[:, :, k] = focal2 / np.sqrt(self.normalization[k])
		if to_pupil:
			out = pupil3_bb
		else:
			out = focal2_bb
		#return pupil1_pad, pupil2_pad, pupil3_pad, focal1, pupil4, focal2
		return out
	
	def Gen_Vortex_mask(self, charge, N):
		[X, Y] = np.meshgrid(np.arange(-N/2+0.5, N/2+0.5), np.arange(-N/2+0.5, N/2+0.5))
		V = np.exp(1j * charge * np.arctan2(Y, X))
		return V