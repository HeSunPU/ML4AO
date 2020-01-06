# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:02:55 2018

@author: He Sun, Princeton University

Filters for estimating the hidden states in the linear hidden Markov model (HMM),

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

Filters include:

Batch least square filter: 
x_hat = lse(y, H)
    
Kalman filter: 
x_new1, P_new1, x_new0, P_new0 = Kalman_filter(y, u, x_old, P_old, F, G, H, Q, R)

Rauch smoother: 
x_old2, P_old2 = Rauch_smoother(x_old1, P_old1, x_new0, P_new0, x_new2, P_new2, F)

Extended Kalman filer:
x_new1, P_new1, x_new0, P_new0 = EKF(y, delta_xc, delta_xp, x_old, P_old, Q, R)

Rauch smoother for EKF:
x_old2, P_old2 = Rauch_smoother_EKF(x_old1, P_old1, x_new0, P_new0, x_new2, P_new2)
    
Here in this files, all the input vectors/matrices should be np.array.
"""

import numpy as np

def lse(y, H, R):
    # least square estimation, x_hat = pinv(H) * y
    # the type of all inputs should be np.array
    H_inv = np.linalg.inv(H.T.dot(H)).dot(H.T)
    #H_inv = np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T) # pseudo-inverse of H
    #H_inv = np.linalg.pinv(H)
    x_hat = H_inv.dot(y.reshape((-1, 1)))
    P_hat = H_inv.dot(R.dot(H_inv.T))
    return x_hat.reshape(len(x_hat)), P_hat

def Kalman_filter(y, u, x_old, P_old, F, G, H, Q, R):
    # Kalman filter estimation
    # the type of all inputs should be np.array
    n_states = len(x_old) # the dimension of the hidden states
    n_observations = len(y) # the dimension of the observations
    n_controls = len(u) # the dimension of the control commands
    
    x_old = x_old.reshape((n_states, 1))
    y = y.reshape((n_observations, 1))
    u = u.reshape((n_controls, 1))
    
    x_new0 = F.dot(x_old) + G.dot(u) # predict the new states
    P_new0 = F.dot(P_old).dot(F.T) + Q # predict the new covariance
    S = R+H.dot(P_new0).dot(H.T)
    K = P_new0.dot(H.T.dot(np.linalg.inv(S))) # optimal Kalman gain
    x_new1 = x_new0 + K.dot(y-H.dot(x_new0)) # update the state estimate
    P_new1 = (np.eye(n_states)-K.dot(H)).dot(P_new0)
    
    return x_new1.reshape(n_states), P_new1, x_new0.reshape(n_states), P_new0

def Rauch_smoother(x_old1, P_old1, x_new0, P_new0, x_new2, P_new2, F):
    # Rauch smoother
    # the type of all inputs should be np.array
    n_states = len(x_old1)
    x_old1 = x_old1.reshape((n_states, 1))
    x_new0 = x_new0.reshape((n_states, 1))
    x_new2 = x_new2.reshape((n_states, 1))
    
    C = P_old1.dot(F.T).dot(np.linalg.inv(P_new0))
    x_old2 = x_old1 + C.dot(x_new2 - x_new0)
    P_old2 = P_old1 + C.dot(P_new2 - P_new0).dot(C.T)
    return x_old2.reshape(n_states), P_old2

def EKF(y, delta_xc, delta_xp, x_old, P_old, Q, R):
    # Kalman filter estimation
    # the type of all inputs should be np.array
    x_old = x_old.reshape((-1, 1))
    y = y.reshape((-1, 1))
    #delta_xc = delta_xc.reshape((-1, 2))
    #delta_xp = delta_xp.reshape((-1, 2))
    
    x_new0 = x_old + delta_xc # predict the new state
    P_new0 = P_old + Q # predict the new covariance
    
    H = np.empty((y.shape[0], x_old.shape[0]))
    for i in range(delta_xp.shape[0]):
        H[i, :] = np.array([2*(x_new0[0]+delta_xp[i, 0]), 2*(x_new0[1]+delta_xp[i, 1]), 1])
    
    y_new0 = []
    for i in range(delta_xp.shape[0]):
        y_new0.append((x_new0[0]+delta_xp[i, 0])**2+(x_new0[1]+delta_xp[i, 1])**2+x_new0[2])
    y_new0 = np.array(y_new0).reshape((-1, 1))
    
    S = R + H.dot(P_new0).dot(H.T)
    K = P_new0.dot(H.T.dot(np.linalg.inv(S))) # optimal Kalman gain
    x_new1 = x_new0 + K.dot(y-y_new0) # update the state estimation based on observations
    P_new1 = (np.eye(x_old.shape[0]) - K.dot(H)).dot(P_new0) # update the 
        
    return x_new1.reshape(x_new1.shape[0]), P_new1, x_new0.reshape(x_new0.shape[0]), P_new0


def IEKF(y, delta_xc, delta_xp, x_old, P_old, Q, R, iterations):
    # Kalman filter estimation
    # the type of all inputs should be np.array
    x_old = x_old.reshape((-1, 1))
    y = y.reshape((-1, 1))
    #delta_xc = delta_xc.reshape((-1, 2))
    #delta_xp = delta_xp.reshape((-1, 2))
    
    x_new0 = x_old + delta_xc # predict the new state
    P_new0 = P_old + Q # predict the new covariance
    
    H = np.empty((y.shape[0], x_old.shape[0]))
    for i in range(delta_xp.shape[0]):
        H[i, :] = np.array([2*(x_new0[0]+delta_xp[i, 0]), 2*(x_new0[1]+delta_xp[i, 1]), 1])
    
    y_new0 = []
    for i in range(delta_xp.shape[0]):
        y_new0.append((x_new0[0]+delta_xp[i, 0])**2+(x_new0[1]+delta_xp[i, 1])**2+x_new0[2])
    y_new0 = np.array(y_new0).reshape((-1, 1))
    
    S = R + H.dot(P_new0).dot(H.T)
    K = P_new0.dot(H.T.dot(np.linalg.inv(S))) # optimal Kalman gain
    x_new1 = x_new0 + K.dot(y-y_new0) # update the state estimation based on observations
    P_new1 = (np.eye(x_old.shape[0]) - K.dot(H)).dot(P_new0) # update the covariance
    
    # start iterated Kalman filter
    for k in range(iterations):
        for i in range(delta_xp.shape[0]):
            H[i, :] = np.array([2*(x_new1[0]+delta_xp[i, 0]), 2*(x_new1[1]+delta_xp[i, 1]), 1])
        y_new1 = []
        for i in range(delta_xp.shape[0]):
            y_new1.append((x_new1[0]+delta_xp[i, 0])**2+(x_new1[1]+delta_xp[i, 1])**2+x_new1[2])
        y_new1 = np.array(y_new1).reshape((-1, 1))
        S = R + H.dot(P_new0).dot(H.T)
        K = P_new0.dot(H.T.dot(np.linalg.inv(S))) # optimal Kalman gain
        x_new1 = x_new0 + K.dot(y - y_new1 - H.dot(x_new0 - x_new1))
        P_new1 = (np.eye(x_old.shape[0]) - K.dot(H)).dot(P_new0)
    
    return x_new1.reshape(x_new1.shape[0]), P_new1, x_new0.reshape(x_new0.shape[0]), P_new0


def IEKF_DH(y, delta_xc, x_old, P_old, Q, R, iterations):
    # Kalman filter estimation
    # the type of all inputs should be np.array
    x_old = x_old.reshape((-1, 1))
    y = y.reshape((-1, 1))
    #delta_xc = delta_xc.reshape((-1, 2))
    #delta_xp = delta_xp.reshape((-1, 2))
    
    x_new0 = x_old + delta_xc # predict the new state
    P_new0 = P_old + Q # predict the new covariance
    
    H = np.empty((y.shape[0], x_old.shape[0]))
    for i in range(y.shape[0]):
        H[i, :] = np.array([2*x_new0[0], 2*x_new0[1], 1])
    
    y_new0 = []
    for i in range(y.shape[0]):
        y_new0.append((x_new0[0])**2+(x_new0[1])**2+x_new0[2])
    y_new0 = np.array(y_new0).reshape((-1, 1))
    
    S = R + H.dot(P_new0).dot(H.T)
    K = P_new0.dot(H.T.dot(np.linalg.inv(S))) # optimal Kalman gain
    x_new1 = x_new0 + K.dot(y-y_new0) # update the state estimation based on observations
    P_new1 = (np.eye(x_old.shape[0]) - K.dot(H)).dot(P_new0) # update the covariance
    
    # start iterated Kalman filter
    for k in range(iterations):
        for i in range(y.shape[0]):
            H[i, :] = np.array([2*x_new1[0], 2*x_new1[1], 1])
        y_new1 = []
        for i in range(y.shape[0]):
            y_new1.append((x_new1[0])**2+(x_new1[1])**2+x_new1[2])
        y_new1 = np.array(y_new1).reshape((-1, 1))
        S = R + H.dot(P_new0).dot(H.T)
        K = P_new0.dot(H.T.dot(np.linalg.inv(S))) # optimal Kalman gain
        x_new1 = x_new0 + K.dot(y - y_new1 - H.dot(x_new0 - x_new1))
        P_new1 = (np.eye(x_old.shape[0]) - K.dot(H)).dot(P_new0)
    
    return x_new1.reshape(x_new1.shape[0]), P_new1, x_new0.reshape(x_new0.shape[0]), P_new0

def Rauch_smoother_EKF(x_old1, P_old1, x_new0, P_new0, x_new2, P_new2):
    # Rauch smoother
    # the type of all inputs should be np.array
    x_old1 = x_old1.reshape((-1, 1))
    x_new0 = x_new0.reshape((-1, 1))
    x_new2 = x_new2.reshape((-1, 1))
    
    C = P_old1.dot(np.linalg.inv(P_new0))
    x_old2 = x_old1 + C.dot(x_new2 - x_new0)
    P_old2 = P_old1 + C.dot(P_new2 - P_new0).dot(C.T)
    
    return x_old2.reshape(x_old2.shape[0]), P_old2