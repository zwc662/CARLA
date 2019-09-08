import argparse
import os
import sys

import numpy as np
import theano.tensor as T

import pytorch
import os
import sys
import csv
import pickle

import time
timestr = time.strftime("%Y%m%d-%H%M%S")

import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.autograd import Variable
from scipy.linalg import block_diag

from scipy.sparse import coo_matrix, vstack, linalg
from cvxopt import spmatrix, spdiag, matrix, solvers, sparse


from .ilqr import CarDynamics
from .cost import Cost, PathQRCost
from .controller import iLQR
from .utils import num_parameters, on_iterations

# Global variables defined for monitoring the iLQR processing 


class EnhanceCost(Cost):
    """ Quadratic Regulator Instantaneous Cost for trajectory following and barrier function."""

    def __init__(self, model_0, model_1, \
            Q, q, R, r, b12, b34, q1, q2, q3, q4, x_path, x_avoids, Q_terminal = None, u_path = None):
        # Construct the basic cost parameters
        self.Q  = np.array(Q)
        self.q = np.array(q)
        self.R = np.array(R)
        self.r = np.array(r)

        # Configuration for the barrier cost
        self.b12 = b12
        self.b34 = b34
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4

        """ F,f are the quadratic and linear costs for the state barrier cost"""
        self.F = np.zeros(self.Q.shape)
        self.f = np.zeros(self.q.shape)
        
        # State, action and path size
        self.state_size = self.Q.shape[0]
        self.action_size = self.R.shape[0]
        self.path_length = self.x_path.shape[0]

        # The other basic parameters
        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(self.Q_terminal)

        if u_path is None:
            self.u_path = np.zeros([path_length - 1, action_size])
        else:
            self.u_path = np.array(u_path)

        if x_avoids is None:
            self.x_avoids = np.zeros([path_length, state_size])
        else:
            self.x_avoids = np.array(x_avoids)

        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.q.shape[0] == self.Q.shape[0], "q mismatch"

        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        assert self.r.shape[0] == self.R.shape[0], "r mismatch"


        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch {} vs {}".format(R.shape, u_path.shape)

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T
        self._R_plus_R_T = self.R + self.R.T
        self._F_plus_F_T = self.F + self.F.T

        # Numbers of the models' parameters should be the same
        self.theta_size = num_parameters(model_0)
        assert self.theta_size == num_parameters(model_1)


        # Constant cost
        self.l1s = np.empty([self.path_length, 1, 1])
        # Jacobians of the costs w.r.t x and u
        self.lxs = np.empty([self.path_length, self.state_size, 1])  
        self.lus = np.empty([self.path_length, self.action_size, 1])
        # Hessians of the costs w.r.t x and u
        self.lxxs = np.empty([self.path_length, self.state_size, self.state_size])
        self.luus = np.empty([self.path_length, self.action_size, self.action_size])
        self.luxs = np.empty([self.path_length, self.action_size, self.state_size])

        # The difference between the outputs of model_1 and model_0
        self.u_diffs = np.empty([self.path_length, self.action_size, 1])
        # Jacobians of model_0 w.r.t x
        self.pixs = np.empty([self.path_length, self.state_size, self.action_size])
        # Jacobians of model_1 w.r.t \theta
        self.pithetas = np.empty([self.path_length, self.theta_size, self.action_size])


        super(PolicyEnhanceCost, self).__init__()

    def l(self, x, u, i, terminal = False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        r = self.r
        q = self.q
        R = self.R

        x_diff = x - self.x_path[i]
        x_dist = x - self.x_avoids[i] 
        x_diff[2:] = 0.0
        x_dist[2:] = 0.0
        barrier_cost = self.q3 * np.exp(self.q4 * (- x_dist.T.dot(x_dist) + self.b34 * self.b34))
        barrier_cost += self.q1 * np.exp(self.q2 * (x_diff.T.dot(x_diff) - self.b12 * self.b12))

        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)
        if terminal:
            return squared_x_cost

        u_diff = u - self.u_path[i]
        squared_u_cost = u_diff.T.dot(R).dot(u_diff)

        linear_cost = x_diff.T.dot(q) + u_diff.T.dot(r)


        
        return squared_x_cost + squared_u_cost + linear_cost + barrier_cost + 
        




class EnhanceController()
    """ An enhancer that uses similar algorithm as iLQR with a different way of defininig the cost
    """
    def __init__(self, steps_ahead = 10, dt = 0.25, model_0, model_1, N, max_reg = 1e10, hessian = False):
        """Construct an enhancer.

        Args:
            dynamics: Plant dynamics
            model_0: the initial model
            model_1: the model to be enhanced

    @staticmethod
    def QP(model, P_qp, q_qp, G_qp, h_qp, A_gp = None, b_qp = None):
        P_qp = numpy_sparse_to_spmatrix(P_qp)
        q_qp = matrix(q_qp)
        G_qp = numpy_sparse_to_spmatrix(G_qp)
        h_qp = matrix(h_qp)
        print("Starting solving QP")
        solvers.options['feastol'] = 1e-5
        sol = solvers.qp(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp)
        
        theta_diffs = list(sol['x'])
        if theta_diffs is None:
            theta_diffs = np.zeros(num_parameters(model))
        return theta_diffs

