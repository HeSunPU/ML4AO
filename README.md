# ML4AO - Machine Learning for Adaptive Optics system identification
![overview image](https://github.com/HeSunPU/ML4AO/blob/master/overview/EM_overview.png)
[Identification and adaptive control of a high-contrast focal plane wavefront correction system](https://www.spiedigitallibrary.org/journals/Journal-of-Astronomical-Telescopes-Instruments-and-Systems/volume-4/issue-4/049006/Identification-and-adaptive-control-of-a-high-contrast-focal-plane/10.1117/1.JATIS.4.4.049006.full?SSO=1)

> All coronagraphic instruments for exoplanet high-contrast imaging need wavefront correction systems to reject optical aberrations and create sufficiently dark holes. Since the most efficient wavefront correction algorithms (controllers and estimators) are usually model-based, the modeling accuracy of the system influences the ultimate wavefront correction performance. Currently, wavefront correction systems are typically approximated as linear systems using Fourier optics. However, the Fourier optics model is usually biased due to inaccuracies in the layout measurements, the imperfect diagnoses of inherent optical aberrations, and a lack of knowledge of the deformable mirrors (actuator gains and influence functions). Moreover, the telescope optical system varies over time because of instrument instabilities and environmental effects. We present an expectation–maximization (E-M) approach for identifying and real-time adapting the linear telescope model from data. By iterating between the E-step (a Kalman filter and a Rauch smoother) and the M-step (analytical or gradient-based optimization), the algorithm is able to recover an accurate system state space model, even if the model depends on the electric fields, which are unmeasurable hidden variables.

Further mathematical and implementation details are described in our paper:
```
@article{sun2018identification,
  title={Identification and adaptive control of a high-contrast focal plane wavefront correction system},
  author={Sun, He and Kasdin, N Jeremy and Vanderbei, Robert},
  journal={Journal of Astronomical Telescopes, Instruments, and Systems},
  volume={4},
  number={4},
  pages={049006},
  year={2018},
  publisher={International Society for Optics and Photonics}
}

```
If you make use of the code, please cite the paper in any resulting publications.

## Setup
The EM system identification codes ("EMsystemID.py") are developed based on python package "numpy", "scipy" and "tensorflow". 

It is originally tested using Focal Plane Wavefront Control Python package ("FPWCpy"), as also included in this repository. However, it could be used independently in identifying the state space model of any WFSC system.

## Run EM system identification
### The formula of the state space model (SSM) we are identifying
```
x_k = x_{k-1} + G1 u1_k + G2 u2_k + w_k,
y_k = H_k x_k + n_k + n_k, H_k = 4 (G1 u1p_k + G2 u2p_k).T,
```
where "x_k" is the electric field, "y_k" is the pair-wise difference image, "G1" and "G2" are the Jacobian matrices of DM1 and DM2, "u1_k" and "u2_k" are the control commands of DM1 and DM2, "H_k" is the observation matrix, "u1p_k" and "u2p_k" are the probe commands of DM1 and DM2, "w_k" is the process noise and "n_k" is the observation noise.

The covariance matrices of process and observation noises are approximated as, 
```
w_k ~ N(0, Q_k),
Q_k = Q0 + Q1 * mean(|G1 u1_k + G2 u2_k|^2)
n_k ~ N(0, R_k),
R_k = R0 + R1 * mean(|G1 u1p_k + G2 u2p_k|^2) * mean(|x_k|^2)
```

1. Define parameters of the original state space model (SSM)
```
params_values = {}
params_values['G1'] = G1
params_values['G2'] = G2
params_values['Q0'] = Q0
params_values['Q1'] = Q1
params_values['R0'] = R0
params_values['R1'] = R1
```

2. Define a EM identifier
```
em_identifier = em.linear_em(params_values, n_pair, model_type='normal')
```
"n_pair" is the number of probe pairs, "model_type" is typically 'normal'. 

(
Otherwise, we can define a reduced-dimensional EM identification. For example, if we only want to identify the 1000 largest modes of Jacobian matrices,
```
em_identifier = em.linear_em(params_values, n_pair, model_type='reduced', dim=1000)
```
)

3. Save the WFSC data
```
data_train = {}
data_train['u1'] = u1
data_train['u2'] = u2
data_train['u1p'] = u1p
data_train['u2p'] = u2p
data_train['I'] = I
```
where "u1" and "u2" have dimension (n_act * n_step), "u1p" and "u2p" have dimension (n_act * (2*n_pair) * n_step), "I" has dimension (n_act * (2*n_pair+1) * n_step). 

"n_act" is the number of actuators on a DM and "n_step" is the total control step. 

"u1p" and "u2p" should be organized as (1st positive probe, 1st negative probe, ...), I should be organized as (non-probed image, 1st positive probed image, 1st negative probed image).

4. Run system identification
```
mse_list = em_identifier.train_params(data_train, lr=1e-7, lr2=1e-2, epoch=10, mstep_itr=2, print_flag=True, params_trainable='Jacobian')
```
where "lr" and "lr2" are the learning rates of the EM algorithm, "epoch" is number of EM iterations, "mstep_itr" is number of optimization iterations in the M-step, "print_flag" represents whether display optimization process, "params_trianable" defines different training methods.

Above command gives representative values for these tuning parameters. Typically, "lr2", "mstep_itr" and "params_trianable" need not to be changed. Only "lr" and "epoch" need to be well tuned.

**Please check "run_simulation.py for an example!!!"**
