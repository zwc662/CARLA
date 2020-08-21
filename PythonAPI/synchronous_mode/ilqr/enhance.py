import argparse
import os
import sys

import numpy as np
import theano.tensor as T

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
from .utils import num_parameters, on_iteration_enhance

# Global variables defined for monitoring the iLQR processing 

# set MACROS for data normalization
max_pos_val = 500
max_yaw_val = 180
max_speed_val = 40



class EnhanceCost(Cost):
    """ Quadratic Regulator Instantaneous Cost for trajectory following and barrier function."""

    def __init__(self, model_0, model_1, trajectory, b0, b12, b34, q1, q2, q3, q4, x_path, x_avoids, lagrange):

        # State, action and path size
        self.x_path =x_path
        self.state_size = self.x_path.shape[-1]
        self.action_size = trajectory.us[0].shape[-1]
        self.path_length = self.x_path.shape[0]
        # Configuration for the barrier cost
        self.b0 = b0
        self.b12 = b12
        self.b34 = b34
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4

        self.lagrange = lagrange
        """ F,f are the quadratic and linear costs for the state barrier cost"""
        self.F = np.zeros([self.state_size, self.state_size])
        self.f = np.zeros([self.state_size, 1])
        # Precompute some common constants.
        self._F_pQus_F_T = self.F + self.F.T
        

        # The other basic parameters
        if x_avoids is None:
            self.x_avoids = np.zeros([path_length, state_size])
        else:
            self.x_avoids = np.array(x_avoids)

        self.model_0 = model_0
        self.model_1 = model_1
        self.theta_size = num_parameters(model_0)
        assert self.theta_size == num_parameters(model_1)

        # Constant cost
        self.Q1s = np.empty([self.path_length, 1, 1])
        # Jacobians of the costs w.r.t x and u
        self.Qxs = np.empty([self.path_length, self.state_size, 1])  
        self.Qus = np.empty([self.path_length, self.action_size, 1])
        # Hessians of the costs w.r.t x and u
        self.Qxxs = np.empty([self.path_length, self.state_size, self.state_size])
        self.Quus = np.empty([self.path_length, self.action_size, self.action_size])
        self.Quxs = np.empty([self.path_length, self.action_size, self.state_size])

        # The difference between the outputs of model_1 and model_0
        self.xs = np.asarray(trajectory.states)
        self.xs = torch.from_numpy(self.xs).float().to('cpu')

        self.u_0s = np.reshape(self.model_0(self.xs)[:self.path_length].data.cpu().numpy(), [self.path_length, self.action_size, 1])
        
        # The output of model_1 and its Jacobians w.r.t x
        self.u_1s = np.empty([self.path_length, self.action_size, 1])
        pixs = np.empty([self.path_length, self.xs.shape[-1], self.action_size])
        for i in range(self.path_length):
            i_x = torch.tensor(self.xs[i, :].view(1, -1), requires_grad = True)
            i_u_1 = self.model_1(i_x)
            for k in range(self.action_size):
                pixs[i, :, k] = torch.autograd.grad(i_u_1[0, k], i_x, retain_graph = True)[0].cpu().numpy().T[0:self.xs.shape[-1], 0]
            self.u_1s[i] = i_u_1.detach().cpu().numpy().T
        self.pixs = np.empty([self.path_length, self.state_size, self.action_size])
        # Get gradient w.r.t input x[1, 2, 5, 0] <<<< pix 
        indices = [1, 2, 5, 0]
        scales = {1: 1./max_pos_val, 2: 1./max_pos_val, 5: 1./max_yaw_val, 0: 1./max_speed_val}
        for j in range(len(indices)):
            self.pixs[:, j, :] = pixs[:, indices[j], :] * scales[indices[j]] 
        self.u_diffs = self.u_1s - self.u_0s


        super(EnhanceCost, self).__init__()

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
        if terminal:
            return 0
        x_diff = x - self.x_path[i]
        x_dist = x - self.x_avoids[i] 
        x_diff[2:] = 0.0
        x_dist[2:] = 0.0
        barrier_cost = self.q3 * np.exp(self.q4 * (- x_dist.T.dot(x_dist) + self.b34 * self.b34))
        barrier_cost += self.q1 * np.exp(self.q2 * (x_diff.T.dot(x_diff) - self.b12 * self.b12))

        policy_cost =  2.0 * np.linalg.norm(u - self.u_0s[i, :, 0]) * self.b0

        
        return barrier_cost + policy_cost  

    def l_x(self, x, u, i, terminal = False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return 0
        x_diff = np.reshape(x - self.x_path[i], (self.state_size, 1))
        x_diff[2:] = 0.0
        x_dist = np.reshape(x - self.x_avoids[i], (self.state_size, 1))
        x_dist[2:] = 0.0


        self.f = - self.q3 * self.q4 * np.exp(self.q4 * (- x_dist.T.dot(x_dist) + self.b34 * self.b34)) * 2 * x_dist
        self.f +=  self.q1 * self.q2 * np.exp(self.q2 * (x_diff.T.dot(x_diff) - self.b12 * self.b12)) * 2 * x_diff


        self.Qxs[i] = 2. * self.pixs[i].dot(self.u_diffs[i]) * self.b0 / 4.0


        return self.f.T + self.Qxs[i].T * 2.0

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            return 0

        self.Qus[i] = 2. * self.u_diffs[i] * self.b0 /4.0
        return self.Qus[i].T * 2.0


    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            return 0
        x_diff = np.reshape(x - self.x_path[i], (self.state_size, 1))
        x_diff[2:] = 0.0
        x_dist = np.reshape(x - self.x_avoids[i], (self.state_size, 1))
        x_dist[2:] = 0.0


        self.F =  - self.q3 * self.q4 * np.exp(self.q4 * (-x_dist.T.dot(x_dist) + self.b34 * self.b34)) * 2  + self.q3 * self.q4**2 * np.exp(self.q4 * (- x_dist.T.dot(x_dist) + self.b34 * self.b34)) * 4 * x_dist.dot(x_dist.T)
        self.F += self.q1 * self.q2 * np.exp(self.q2 * (x_diff.T.dot(x_diff) - self.b12 * self.b12)) * 2  + self.q1 * self.q2**2 * np.exp(self.q2 * (x_diff.T.dot(x_diff) - self.b12 * self.b12)) * 4 * x_diff.dot(x_diff.T)
        self._F_pQus_F_T = self.F + self.F.T


        self.Qxxs[i] = self.lagrange * (self.pixs[i].dot(self.u_diffs[i])).dot(self.u_diffs[i].T.dot(self.pixs[i].T)) * self.b0/4.

        return self._F_pQus_F_T + self.Qxxs[i] * 2.0


    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            return 0
        self.Quxs[i] = self.lagrange * self.u_diffs[i].dot(self.u_diffs[i].T.dot(self.pixs[i].T)) * self.b0/4.
        return self.Quxs[i] * 2.

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return 0
        self.Quus[i] = self.lagrange * self.u_diffs[i].dot(self.u_diffs[i].T) * self.b0/4.0

        return self.Quus[i] * 2.0
    

        
        




class PolicyEnhancer():
    """ An enhancer that uses similar algorithm as iLQR with a different way of defininig the cost
    """
    def __init__(self, model_0, model_1, steps_ahead = 10, dt = 0.25, l = 1.0, half_width = 2.0, lagrange = 1.0):
        """Construct an enhancer.

        Args:
            dynamics: Plant dynamics
            model_0: the initial model
            model_1: the model to be enhanced
        """
        self.model_0 = model_0
        self.model_1 = model_1
        # Number of steps to be optimized
        self.steps_ahead = steps_ahead
        # Step length
        self.dt = dt
        # Front wheel diameter
        self.l = l
        # Define car dynamics
        self.dynamics = CarDynamics(self.dt, self.l)
        # Lack of variables to build car cost
        self.cost = None
        # lack of variales to build nominal trajectory, lower and upper bounds
        self.x_path = np.zeros([self.steps_ahead, self.dynamics.state_size])
        # Obstacles to avoid
        self.x_avoids = np.zeros([self.steps_ahead, self.dynamics.state_size])
        # Half of the road width
        self.half_width = half_width 
        # Current state
        self.measurements = {'posx': None, 'posy': None, 'v': None, 'theta': None}

        # Lagrange parameter 
        self.lagrange = lagrange

       

    def get_state(self, measurements):
        """Get the state variables from measurements"""
        v = np.linalg.norm([measurements.v.x, measurements.v.y], ord = 2)
        theta = measurements.t.rotation.yaw * np.pi / 180.
        posx, posy = measurements.t.location.x, measurements.t.location.y
        self.measurements['posx'] = posx
        self.measurements['posy'] = posy
        self.measurements['v'] = v
        self.measurements['theta'] = theta
        #print("ilqr measurements:", self.measurements)

    def generate_nominal(self, future_wps_np = None):
        # Along the dense nominal trajectory, select the first self.steps_ahead waypoints 

        # Find the closest reference waypoints among the dense set of waypoints
        # Extracrt the reference waypoints and calculate the barrier sequences
        """ (Optional)Coordinate transform. The ego car is the original point
        x0[:4] = np.asarray([0.0, 0.0, 0.0, self.measurements['v']])
        future_wps_np_ = ILQRController.transform_into_cars_coordinate_system(future_wps_np, \
                self.measurements['posx'], self.measurements['posy'], \
                np.cos(self.measurements['theta']), np.sin(self.measurements['theta']))
        #self.x_path[:, :2] = future_wps_np_[:self.steps_ahead, :2]
        """

        """(Optional)Turn the sparse waypoints into dense waypoints
        future_wps_np_dense = ILQRController.transform_into_dense_points(future_wps_np, self.x_path_density)
        """

        """ Find the closest reference waypoints among the dense set of waypoints
            Extracrt the reference waypoints and calculate the barrier sequences
        self.generate_nominal(future_wps_np_dense)
        self.x_path[:, :2] = future_wps_np_dense[:self.steps_ahead, :2]
        """
        self.x_path = np.zeros([self.steps_ahead, self.dynamics.state_size])
        self.x_path[:self.steps_ahead, :2] = future_wps_np[:self.steps_ahead, :2]
        
    def generate_avoidance(self, locations = None):
        if locations is not None:
            for i in range(self.steps_ahead):
                if len(locations) > 1:
                    self.x_avoids[i, 0] = locations[i].x
                    self.x_avoids[i, 1] = locations[i].y
                else:
                    self.x_avoids[i, 0] = locations[0].x
                    self.x_avoids[i, 1] = locations[0].y
        print("Distance to obstacle: {}".format(\
                np.linalg.norm([self.measurements['posx'] - self.x_avoids[0, 0], \
                self.measurements['posy'] - self.x_avoids[0, 1]])))


    def generate_cost(self, trajectory):
        b0 = 10.0
        # For staying in lane
        q1 = 1.0E-1
        q2 = 1.0E-4

        # For avoidance
        q3 = 10.0 * 1.
        q4 = 10.0 * 2.0

        b12 = self.half_width
        b34 = self.half_width

        q = np.zeros([self.dynamics.state_size, 1])
        r = np.zeros([self.dynamics.action_size, 1])

        lagrange = self.lagrange

        self.model_0.zero_grad()
        self.model_1.zero_grad()
        self.cost = EnhanceCost(self.model_0, self.model_1, trajectory = trajectory, b0 = b0, \
            b12 = b12, b34 = b34, q1 = q1, q2 = q2, q3 = q3, q4 = q4, \
            x_path = self.x_path, x_avoids = self.x_avoids, lagrange = lagrange)

    def enhance(self, x0, us_init):
        assert us_init.shape[0] == self.steps_ahead

        # Initialize ilqr solver
        
        """ Run ilqr to obtain sequence of state and control
            xs: [1 + self.steps_ahead, state_size]
            us: [self.steps_ahead, state_size]
        """
        #ilqr = iLQR(self.dynamics, self.cost, self.steps_ahead)
        #xs, us = ilqr.fit(x0, us_init, n_iterations = 100, on_iteration=on_iteration_enhance)
        xs, us = self.fit(x0, us_init, n_iterations = 10)

        return xs, us

    def control(self, future_wps_np, trajectory, avoidances = None):
        measurements = trajectory.measurements[0]
        # Collect state variables
        self.get_state(measurements)
        # Initial state is x0
        x0 = np.zeros([self.dynamics.state_size])
        x0[:4] = np.asarray([self.measurements['posx'], \
                self.measurements['posy'], \
                self.measurements['theta'], \
                self.measurements['v']])
        print("Enhance policy from state ", x0)

        # Define noimal path
        self.generate_nominal(future_wps_np)

        # Define obstacles
        self.generate_avoidance(avoidances)

        # Define the cost
        self.generate_cost(trajectory)

        us_init = np.asarray(trajectory.us)[:self.steps_ahead, :]
        # Run enhancement
        xs, us = self.enhance(x0, us_init)

        return xs, us
        
        
    def fit(self, x0, us_init, n_iterations):
        xs_opt = np.empty([self.steps_ahead, self.dynamics.state_size])
        us_opt = np.empty([self.steps_ahead, self.dynamics.action_size])

        Q1 = np.zeros([self.steps_ahead, 1, 1])
        Qx = np.empty([self.steps_ahead, self.dynamics.state_size, 1])
        Qu = np.empty([self.steps_ahead, self.dynamics.action_size, 1])
        Qxx = np.empty([self.steps_ahead, self.dynamics.state_size, self.dynamics.state_size])
        Qxu = np.empty([self.steps_ahead, self.dynamics.state_size, self.dynamics.action_size])
        Quu = np.empty([self.steps_ahead, self.dynamics.action_size, self.dynamics.action_size])
        self.K = np.empty([self.steps_ahead, self.dynamics.action_size, self.dynamics.state_size])
        self.k = np.empty([self.steps_ahead, self.dynamics.action_size, 1])

        xs_init = np.empty([self.steps_ahead, self.dynamics.state_size])
        xs_init[0] = x0
        for i in range(1, self.steps_ahead):
            xs_init[i] = self.dynamics.f(xs_init[i - 1], us_init[i], i)
            x_u_i = (xs_init[i], us_init[i], i)
            Qx[i] = self.cost.l_x(*(x_u_i)).T
            Qu[i] = self.cost.l_u(*(x_u_i)).T
            Qxx[i] = self.cost.l_xx(*(x_u_i))
            Qxu[i] = self.cost.l_ux(*(x_u_i)).T
            Quu[i] = self.cost.l_uu(*(x_u_i)) 

        J_opt = float('inf')
        converged = False
        alphas = 0.9**(np.arange(n_iterations)**2)
        mus = 1.1**(np.arange(n_iterations)**2)
        tol = 1e-5
        
        for i_itr in range(n_iterations**2):
            alpha = alphas[int(i_itr/n_iterations)]
            mu = mus[int(i_itr % n_iterations)]
    
            J_new = 0.0

            xs = np.empty([self.steps_ahead, self.dynamics.state_size])
            xs[0] = x0
            us = np.empty([self.steps_ahead, self.dynamics.action_size])

            for i in range(self.steps_ahead):
                Quu[i] += self.cost.b0 * mu * np.eye(Quu.shape[-1])
                self.K[i] = -np.linalg.solve(Quu[i], Qxu[i].T)
                self.k[i] = -np.linalg.solve(Quu[i], Qu[i])
                us[i] = us_init[i] + \
                        (alpha * self.k[i] + self.K[i].dot(np.expand_dims(xs[i] - xs_init[i], axis = 1))).flatten()
                J_new += self.cost.l(xs[i], us[i], i)
                
                if i < self.steps_ahead - 1:
                    xs[i + 1] = self.dynamics.f(xs[i], us[i], i)

            if J_opt > J_new:
                if np.abs((J_opt - J_new)/J_opt) < tol:
                    converged = True
                    print("Converged.")
                    break
                #print("Found new optimal solution.")
                J_opt = J_new
                xs_opt = xs[:, :]
                us_opt = us[:, :]
            else:
                if i_itr % 10 == 0:
                    #print("Iteration {} cost {}".format(i_itr, J_new))
                    pass

        return xs_opt, us_opt


