import numpy as np

def Zernike(index=0, wfe_shape=100):
	half_size = 0.5 * (wfe_shape - 1)
	x = np.arange(wfe_shape) - half_size
	y = np.arange(wfe_shape) - half_size
	xv, yv = np.meshgrid(x, y)
	rv = np.sqrt(xv**2 + yv**2) / wfe_shape
	thetav = np.arctan2(xv, yv)
	if index == 0:
		return np.ones([wfe_shape, wfe_shape])
	elif index == 1:
		return 2 * rv * np.sin(thetav)
	elif index == 2:
		return 2 * rv * np.cos(thetav)
	elif index == 3:
		return np.sqrt(6) * rv**2 * np.sin(2*thetav)
	elif index == 4:
		return np.sqrt(3) * (2*rv**2 - 1)
	elif index == 5:
		return np.sqrt(6) * rv**2 * np.cos(2*thetav)
	elif index == 6:
		return np.sqrt(8) * rv**3 * np.sin(3*thetav)
	elif index == 7:
		return np.sqrt(8) * (3*rv**3-2*rv) * np.sin(thetav)
	elif index == 8:
		return np.sqrt(8) * (3*rv**3-2*rv) * np.cos(thetav)
	elif index == 9:
		return np.sqrt(8) * rv**3 * np.cos(3*thetav)
	elif index == 10:
		return np.sqrt(10) * rv**4 * np.sin(4*thetav)
	elif index == 11:
		return np.sqrt(10) * (4*rv**4 - 3*rv**2) * np.sin(2*thetav)
	elif index == 12:
		return np.sqrt(5) * (6*rv**4 - 6*rv**2 + 1)
	elif index == 13:
		return np.sqrt(10) * (4*rv**4 - 3*rv**2) * np.cos(2*thetav)
	elif index == 14:
		return np.sqrt(10) * rv**4 * np.cos(4*thetav)


class PhaseErr:
	def __init__(self, order=15, wfe_shape=100, coef=np.zeros(15), drift_dynamics=np.zeros(15)):
		self.order = order
		self.wfe_shape = wfe_shape
		self.coef = coef
		self.drift_dynamics = drift_dynamics
		self.phase_map = np.zeros([wfe_shape, wfe_shape])
		for k in range(order):
			self.phase_map += Zernike(k , wfe_shape) * coef[k]

	def drift(self, drift_coef=[]):
		for k in range(self.order):
			if len(drift_coef) == 0:
				self.coef[k] += np.random.normal(loc=0.0, scale=self.drift_dynamics[k])
			else:
				self.coef[k] += drift_coef[k]
		self.phase_map = np.zeros([self.wfe_shape, self.wfe_shape])
		for k in range(self.order):
			self.phase_map += Zernike(k , self.wfe_shape) * self.coef[k]
