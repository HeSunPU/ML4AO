"""
created on Sun Apr. 21, 2019

@author: He Sun, Princeton University

model of WFIRST shaped pupil lyot coronagraph (SPLC)

adapted from the A.J. Riggs's FALCO codes and Jessica 
Gersh-Range's coronagraph design codes

"""

import os
import numpy as np
import scipy.io as spio
import numpy.fft as fft
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import imp
import tensorflow as tf
import skimage.transform as transform

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
		self.coronagraph_type = 'splc'
		self.SPwidth = 0.01
		self.Nsp = 432
		self.focalLength = 1.1642
		self.apertureWidth = self.SPwidth
		self.Naperture = self.Nsp
		self.Nfpm = 160
		self.FPMpitch = 8.3168e-6
		self.FPMwidth = self.Nfpm * self.FPMpitch
		self.lyotWidth = self.SPwidth
		self.Nlyot = self.Nsp
		self.apertureMask = fits.getdata('./FPWCpy/optical_layout/masks/splc_aperture.fits', ext=0)
		self.SPshape = fits.getdata('./FPWCpy/optical_layout/masks/splc_apodizer.fits', ext=0)
		self.FPmask = fits.getdata('./FPWCpy/optical_layout/masks/splc_FPM_990by990.fits', ext=0)
		self.LyotStop = fits.getdata('./FPWCpy/optical_layout/masks/splc_lyot_1000by1000.fits', ext=0)
		self.apertureMask = imresize(self.apertureMask, [self.Naperture, self.Naperture]) / 255
		self.SPshape = imresize(self.SPshape, [self.Nsp, self.Nsp]) / 255
		self.LyotStop = imresize(self.LyotStop, [self.Nlyot, self.Nlyot]) / 255
		self.FPmask = imresize(self.FPmask, [self.Nfpm, self.Nfpm]) / 255

		# wavefront error information
		self.wfe = wfe
		if wfe:
			if error_maps:
				self.DM1_error = imresize(error_maps[0], [494, 494]) / 255
				self.DM2_error = imresize(error_maps[1], [494, 494]) / 255
				self.SP_error = imresize(error_maps[2], [self.Nsp, self.Nsp]) / 255
			else:
				self.DM1_error = 2 * imresize(spio.loadmat('./FPWCpy/optical_layout/err_maps/PSD_DM1.mat')['PSD_DM1'], [494, 494]) / 255
				self.DM2_error = 2 * imresize(spio.loadmat('./FPWCpy/optical_layout/err_maps/PSD_DM2.mat')['PSD_DM2'], [494, 494]) / 255
				self.SP_error = 2 * imresize(spio.loadmat('./FPWCpy/optical_layout/err_maps/PSD_SP.mat')['PSD_SP'], [self.Nsp, self.Nsp]) / 255

		# camera information

		self.camera_pitch = 4.54e-6
		self.camera_binXi = 4
		self.camera_binEta = 4
		self.camera_Nxi = 99
		self.camera_Neta = 83

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
		self.p3_to_f1_pre = np.empty((self.Nfpm, self.Nsp, len(self.wavelength)), dtype=np.complex)
		self.p3_to_f1_post = np.empty((self.Nsp, self.Nfpm, len(self.wavelength)), dtype=np.complex)
		self.p3_to_f1_scaler = np.empty(len(self.wavelength), dtype=np.complex)
		dx = self.SPwidth / self.Nsp
		xp = np.arange((-self.Nsp/2+0.5)*dx, (self.Nsp/2+0.5)*dx, dx)
		dxi = self.FPMpitch
		xf1 = np.arange((-self.Nfpm/2+0.5)*dxi, (self.Nfpm/2+0.5)*dxi, dxi)
		for k in range(len(self.wavelength)):
			wl = self.wavelength[k]
			self.p3_to_f1_pre[:, :, k] = np.exp(-2 * np.pi * 1j * np.outer(xf1, xp) / (wl * self.focalLength))
			self.p3_to_f1_post[:, :, k] = np.exp(-2 * np.pi * 1j * np.outer(xp, xf1) / (wl * self.focalLength))
			self.p3_to_f1_scaler[k] = dx * dxi * np.exp(2 * np.pi * 1j * self.focalLength / wl) / (1j * wl * self.focalLength)
		# Fourier transform from the focal plane mask to the Lyot stop (focal1 to pupil4)
		self.f1_to_p4_pre = self.p3_to_f1_post
		self.f1_to_p4_post = self.p3_to_f1_pre
		self.f1_to_p4_scaler = self.p3_to_f1_scaler
		# Fourier transform from the Lyot stop to camera (pupil4 to focal2)
		self.p4_to_f2_pre = np.empty((self.camera_Neta, self.Nsp, len(self.wavelength)), dtype=np.complex)
		self.p4_to_f2_post = np.empty((self.Nsp, self.camera_Nxi, len(self.wavelength)), dtype=np.complex)
		self.p4_to_f2_scaler = np.empty(len(self.wavelength), dtype=np.complex)
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
		FPmask = self.FPmask
		self.FPmask = np.ones(FPmask.shape)
		Ef = self.Propagate(np.zeros((34, 34)), np.zeros((34, 34)))
		If = np.abs(Ef)**2
		for k in range(len(self.normalization)):
			self.normalization[k] = np.max(If[:, :, k])
		self.FPmask = FPmask

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
			pupil1 = self.apertureMask * Ein
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
				focal1 = self.p3_to_f1_scaler[k] * np.matmul(np.matmul(self.p3_to_f1_pre[:, :, k], pupil3m), self.p3_to_f1_post[:, :, k])
				focal1m = focal1 * self.FPmask

				pupil4 = self.f1_to_p4_scaler[k] * np.matmul(np.matmul(self.f1_to_p4_pre[:, :, k], focal1m), self.f1_to_p4_post[:, :, k])
				pupil4m = pupil4 * self.LyotStop

				focal2 = self.p4_to_f2_scaler[k] * np.matmul(np.matmul(self.p4_to_f2_pre[:, :, k], pupil4m), self.p4_to_f2_post[:, :, k])

				focal2_bb[:, :, k] = focal2 / np.sqrt(self.normalization[k])
		if to_pupil:
			out = pupil3_bb
		else:
			out = focal2_bb
		#return pupil1_pad, pupil2_pad, pupil3_pad, focal1, pupil4, focal2
		return out
		