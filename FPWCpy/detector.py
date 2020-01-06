"""
created on Mon Apr. 23, 2019

@author: He Sun, Princeton University

Define the detector noise models
"""
import numpy as np

class CCD:
	def __init__(self, flux=2e9, readout_std=12, readout=True, photon=True, exp_time=1):
		self.flux = flux # number of photons reaching the detector in unit time (1 second)
		self.readout_std = readout_std # the std of gaussian noise
		self.readout = readout # the flag indicating whether include readout noise
		self.photon = photon # the flag indicating whether include photon noise
		self.exp_time = exp_time # detector exposure time
		self.detector_type = 'CCD' # detector type
		self.detector_name = 'QSI_RS_6_1' # detector model, the QSI RS.6.1 is used in Princeton HCIL

	def set_exposure(self, exp_time):
		# reset the camera exposure time
		self.exp_time = exp_time

	def Add_noise(self, img):
		# add noises to the camera images
		if self.photon:
			Iphoton = 1.0 * np.random.poisson(lam=img * self.flux * self.exp_time)
		else:
			Iphoton = img * self.flux * self.exp_time
		if self.readout:
			readout_noise = self.readout_std * np.random.normal(size=img.shape)
			Iphoton += readout_noise
		I = 1.0 * Iphoton / (self.flux * self.exp_time)
		return I

