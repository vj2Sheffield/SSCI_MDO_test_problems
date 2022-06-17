#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:57:16 2022

@author: victoria
"""

import openmdao.api as om
import numpy as np
import time
import random
import pandas as pd
import sys

i = sys.argv[1]
print('******************************' + str(sys.argv[1]) + '******************************')

path = "[Run " + str(i) + "] DV 2 disciplines"

# Start timer
t0 = time.time()

xLower = np.zeros(5)
xUpper = np.ones(5)
zLower = np.zeros(10)
zUpper = np.ones(10)

# Set parameters
N = 2
n_z = 10 # Number of global design variables
n_x1 = 5 # Number of local design variables in discipline 1
n_y1 = 5 # Number of linking variables in discipline 1
n_x2 = 5 # Number of local design variables in discipline 1
n_y2 = 5 # Number of linking variables in discipline 1

# Set initial guesses
seedVar = random.random()
z_initial = np.array(random.sample(range(100), n_z))/100
x1_initial = np.array(random.sample(range(100), n_x1))/100
x2_initial = np.array(random.sample(range(100), n_x2))/100

# Set dummy values
z_dummy = np.ones((n_z, 1))
x1_dummy = np.ones((n_x1, 1))
y1_dummy = np.ones((n_y1, 1))
x2_dummy = np.ones((n_x2, 1))
y2_dummy = np.ones((n_y2, 1))

d1Matrices = np.zeros((N*n_x1, n_x1 + n_z))
b1Matrices = np.zeros((N*n_x1, n_y1))

bArray = np.array([[ 8,  5, 10,  6,  4],
     [ 6,  5,  9, 10, 10],
     [ 2,  6,  2,  4,  6],
     [ 2,  9,  4,  3,  3],
     [ 5,  9,  9,  2,  4],
     [10,  8,  9,  7,  4],
     [ 4,  2, 10,  7,  4],
     [ 3,  5,  8, 10,  3],
     [ 1,  8,  2,  9,  3],
     [ 5,  1,  6,  9,  2]])
dArray = np.array([[1, 2, 2, 1, 8, 9, 3, 9, 1, 6, 10, 2, 4, 9, 1],
     [4, 7, 9, 1, 2, 3, 1, 6, 8, 6, 2, 8, 6, 6, 5],
     [9, 3, 4, 2, 1, 5, 6, 1, 8, 6, 6, 9, 3, 5, 1],
     [10, 5, 10, 9, 9, 1, 4, 10, 6, 3, 2, 8, 3, 2, 8],
     [10, 7, 7, 8, 3, 10, 2, 10, 5, 1, 5, 5, 8, 9, 8],
     [2, 3, 10, 6, 8, 9, 1, 7, 9, 9, 4, 5, 4, 2, 3],
     [7, 3, 8, 7, 4, 8, 6, 4, 4, 4, 9, 8, 2, 4, 10],
     [3, 8, 7, 4, 2, 10, 3, 3, 5, 3, 1, 7, 5, 4, 9],
     [4, 1, 10, 10, 4, 6, 8, 5, 6, 3, 8, 1, 1, 6, 3],
     [9, 4, 4, 1, 1, 7, 8, 4, 1, 3, 6, 1, 9, 6, 7]])

# A and D arrays..
for i in range(0, N*n_x1):
    d1Matrices[i] = dArray[i:(i+1), :]/np.sum(dArray[i:(i+1), :])
    
# B arrays
for i in range(0, N*n_x1):
    b1Matrices[i] = 2*bArray[i:(i+1), :]/np.sum(bArray[i:(i+1), :])

d1Matrices = np.concatenate((b1Matrices, d1Matrices), axis = 1)

B12 = d1Matrices[0:5, 0:5]
D1 = d1Matrices[0:5, 5:10]
C1 = d1Matrices[0:5, 10:19]

B21 = d1Matrices[5:10, 0:5]
D2 = d1Matrices[5:10, 5:10]
C2 = d1Matrices[5:10, 10:19]

class disc1(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x1", val = x1_dummy)
		self.add_input("y2", val = y2_dummy)

		self.add_output("y1", val = y1_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x1 = inputs["x1"]
		y2 = inputs["y2"]
        
		Cz = np.reshape(np.matmul(C1, z[1:10]), [C1.shape[0], 1])
		Dx = np.matmul(D1, x1)
		B1y = np.matmul(B12, y2)
        
		outputs["y1"] = -Cz - Dx + B1y

class disc2(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x2", val = x2_dummy)
		self.add_input("y1", val = y1_dummy)

		self.add_output("y2", val = y2_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x2 = inputs["x2"]
		y1 = inputs["y1"]
        
		Cz = np.reshape(np.matmul(C2, z[1:10]), [C2.shape[0], 1])
		Dx = np.matmul(D2, x2)
		B2y = np.matmul(B21, y1)
        
		outputs["y2"] = -Cz - Dx + B2y
        

class MDA(om.Group):
	def setup(self):
		cycle = self.add_subsystem("cycle", om.Group(), promotes=["*"])
		cycle.add_subsystem("d1", disc1(), promotes_inputs = ["z", "y2", "x1"], promotes_outputs = ["y1"])
		cycle.add_subsystem("d2", disc2(), promotes_inputs = ["z", "y1", "x2"], promotes_outputs = ["y2"])
        
		self.add_subsystem("obj2", om.ExecComp("f2 = (1 + (9/29)*(sum(x1) + sum(x2) + sum(y1) + sum(y2) + z[1] + z[2] + z[3] + z[4] + z[5] + z[6] + z[7] + z[8] + z[9]))*(1 - (z[0]/(1 + (9/29)*(sum(x1) + sum(x2) + sum(y1) + sum(y2) + z[1] + z[2] + z[3] + z[4] + z[5] + z[6] + z[7] + z[8] + z[9])))**0.5)",
                                            x1 = x1_dummy, x2 = x2_dummy, z = z_dummy, y1 = y1_dummy, y2 = y2_dummy), 
                     promotes = ["x1", "x2", "y1", "y2", "z", "f2"])
		self.add_subsystem("obj1", om.ExecComp("f1 = z[0]", z = z_dummy), promotes = ["z", "f1"])

		cycle.set_input_defaults("z", z_initial)
		cycle.set_input_defaults("x1", x1_initial)
		cycle.set_input_defaults("x2", x2_initial)
        
		cycle.nonlinear_solver = om.NewtonSolver(maxiter = 1000, solve_subsystems = True)
		cycle.linear_solver = om.DirectSolver()

prob = om.Problem()
prob.model = MDA()

# Specify problem driver (SLSQP)
# prob.driver = om.ScipyOptimizeDriver(optimizer="COBYLA", tol = 1e-3, disp = True)
prob.driver = om.pyOptSparseDriver(optimizer = 'NSGA2') 
prob.driver.opt_settings["maxGen"] = 100
prob.driver.opt_settings["PopSize"] = 100
prob.driver.opt_settings["seed"] = seedVar

# Add design variables
prob.model.add_design_var("z", lower = zLower, upper = zUpper)
prob.model.add_design_var("x1", lower = xLower, upper = xUpper)
prob.model.add_design_var("x2", lower = xLower, upper = xUpper)

prob.model.add_constraint("y1", lower = xLower, upper = xUpper)
prob.model.add_constraint("y2", lower = xLower, upper = xUpper)

# Add objective
prob.model.add_objective("f1")
prob.model.add_objective("f2")

recorder = om.SqliteRecorder('cases_var_params.sql')
prob.add_recorder(recorder) # Attach recorder to the problem
prob.driver.add_recorder(recorder) # Attach recorder to the driver

prob.setup()

prob.model.obj1.add_recorder(recorder) # Attach recorder to a subsystem
prob.model.obj2.add_recorder(recorder) # Attach recorder to a subsystem
prob.model.cycle.nonlinear_solver.add_recorder(recorder)
prob.set_solver_print(level = 0)

prob.run_driver()
prob.run_model()

prob.record("final_state")
prob.cleanup()

# Instantiate your CaseReader
cr = om.CaseReader("cases_var_params.sql")
driver_cases = cr.get_cases('driver', recurse=False) # List driver cases (do not recurse to system/solver cases)
solver_cases = cr.list_cases('root.cycle.nonlinear_solver', out_stream=None)

# Plot the path the design variables took to convergence
dv_obj1_values = []
dv_obj2_values = []
dv_z_values = []
dv_x1_values = []
dv_x2_values = []
y1_vals = []
y2_vals = []

for case in driver_cases:
    dv_obj1_values.append(case['obj1.f1'])
    dv_obj2_values.append(case['obj2.f2'])
    dv_z_values.append(case['z'])
    dv_x1_values.append(case['x1'])
    dv_x2_values.append(case['x2'])

for case_id in solver_cases:
    case = cr.get_case(case_id)
    y1_vals.append(case['y1'])
    y2_vals.append(case['y2'])

y1_vals = np.reshape(np.array(y1_vals), (len(y1_vals), 5))
y2_vals = np.reshape(np.array(y2_vals), (len(y2_vals), 5))

dv_obj1_values = np.array(dv_obj1_values)
dv_obj2_values = np.array(dv_obj2_values)
objs = np.array(np.concatenate((dv_obj1_values, dv_obj2_values), axis = 1))
np.savetxt(path + 'obj.csv', objs, delimiter=',')

dv_z_values = np.array(dv_z_values)
zs = np.reshape(np.array(dv_z_values), (len(dv_z_values), 10))
np.savetxt(path + 'z.csv', zs, delimiter=',')

dv_x1_values = np.array(dv_x1_values)
dv_x2_values = np.array(dv_x2_values)

y1_vals = np.array(y1_vals)
y2_vals = np.array(y2_vals)

x1s = np.reshape(np.array(dv_x1_values), (len(dv_x1_values), 5))
x2s = np.reshape(np.array(dv_x2_values), (len(dv_x2_values), 5))

y1s = np.reshape(np.array(y1_vals), (len(y1_vals), 5))
y2s = np.reshape(np.array(y2_vals), (len(y2_vals), 5))

np.savetxt(path + 'x1.csv', x1s, delimiter=',')
np.savetxt(path + 'x2.csv', x2s, delimiter=',')

np.savetxt(path + 'y1.csv', y1s, delimiter=',')
np.savetxt(path + 'y2.csv', y2s, delimiter=',')

# np.savetxt(path + 'seed.out', np.array(seedVar))
data = pd.read_csv('nsga2_best_pop.out', sep="\t", header=1)
data.to_csv(path + 'nsga2_best_pop.csv')
