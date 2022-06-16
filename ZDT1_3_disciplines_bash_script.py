#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:32:52 2022

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
path = "[Run " + str(i) + "] DV 3 disciplines"

# Set parameters
N = 3
n_z = 6 # Number of global design variables
n_x1 = 4 # Number of local design variables in discipline 1
n_y1 = 4 # Number of linking variables in discipline 1
n_x2 = 4 # Number of local design variables in discipline 1
n_y2 = 4 # Number of linking variables in discipline 1
n_x3 = 4 # Number of local design variables in discipline 1
n_y3 = 4 # Number of linking variables in discipline 1

xLower = np.zeros(n_x1)
xUpper = np.ones(n_x1)
zLower = np.zeros(n_z)
zUpper = np.ones(n_z)

# Set initial guesses
z_initial = np.array(random.sample(range(100), n_z))/100
x1_initial = np.array(random.sample(range(100), n_x1))/100
x2_initial = np.array(random.sample(range(100), n_x2))/100
x3_initial = np.array(random.sample(range(100), n_x3))/100

# Set dummy values
z_dummy = np.zeros((n_z, 1))
x1_dummy = np.ones((n_x1, 1))
y1_dummy = np.ones((n_y1, 1))
x2_dummy = np.ones((n_x2, 1))
y2_dummy = np.ones((n_y2, 1))
x3_dummy = np.ones((n_x3, 1))
y3_dummy = np.ones((n_y3, 1))

d1Matrices = np.zeros((N*n_x1, n_x1 + n_z))
b1Matrices = np.zeros((N*n_x1, n_y1))

dArray = np.array([[ 8,  7,  4,  8,  6,  4,  4,  4,  9,  8],
       [ 2,  4, 10,  3,  8,  7,  4,  2, 10,  3],
       [ 3,  5,  3,  1,  7,  5,  4,  9,  4,  1],
       [10, 10,  4,  6,  8,  5,  6,  3,  8,  1],
       [ 1,  6,  3,  9,  4,  4,  1,  1,  7,  8],
       [ 4,  1,  3,  6,  1,  9,  6,  7,  8,  5],
       [10,  6,  4,  6,  5,  9, 10, 10,  2,  6],
       [ 2,  4,  6,  2,  9,  4,  3,  3,  5,  9],
       [ 9,  2,  4, 10,  8,  9,  7,  4,  4,  2],
       [10,  7,  4,  3,  5,  8, 10,  3,  1,  8],
       [ 2,  9,  3,  5,  1,  6,  9,  2,  9,  1],
       [ 4,  3,  6,  7, 10,  2,  6,  8,  3,  4]])

bArray = np.array([[ 1, 10,  5, 10],
       [ 9,  9,  1,  4],
       [10,  6,  3,  2],
       [ 8,  3,  2,  8],
       [10,  7,  7,  8],
       [ 3, 10,  2, 10],
       [ 5,  1,  5,  5],
       [ 8,  9,  8,  2],
       [ 3, 10,  6,  8],
       [ 9,  1,  7,  9],
       [ 9,  4,  5,  4],
       [ 2,  3,  7,  3]])

# A and D arrays..
for i in range(0, N*n_x1):
    d1Matrices[i] = dArray[i:(i+1), :]/np.sum(dArray[i:(i+1), :])
    
# B arrays
for i in range(0, N*n_x1):
    b1Matrices[i] = 2*bArray[i:(i+1), :]/np.sum(bArray[i:(i+1), :])

d1Matrices = np.concatenate((b1Matrices, d1Matrices), axis = 1)

B12 = d1Matrices[0:4, 0:4]
D1 = d1Matrices[0:4, 4:8]
C1 = d1Matrices[0:4, 8:14]

B23 = d1Matrices[4:8, 0:4]
D2 = d1Matrices[4:8, 4:8]
C2 = d1Matrices[4:8, 8:14]

B31 = d1Matrices[8:12, 0:4]
D3 = d1Matrices[8:12, 4:8]
C3 = d1Matrices[8:12, 8:14]


class disc1(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x1", val = x1_dummy)
		self.add_input("y3", val = y2_dummy)

		self.add_output("y1", val = y1_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x1 = inputs["x1"]
		y3 = inputs["y3"]
        
		Cz = np.reshape(np.matmul(C1, z), [C1.shape[0], 1])
		Dx = np.matmul(D1, x1)
		B1y = np.matmul(B12, y3)
        
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
        
		Cz = np.reshape(np.matmul(C2, z), [C2.shape[0], 1])
		Dx = np.matmul(D2, x2)
		B2y = np.matmul(B23, y1)
        
		outputs["y2"] = -Cz - Dx + B2y

class disc3(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x3", val = x2_dummy)
		self.add_input("y2", val = y1_dummy)

		self.add_output("y3", val = y2_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x3 = inputs["x3"]
		y2 = inputs["y2"]
        
		Cz = np.reshape(np.matmul(C3, z), [C3.shape[0], 1])
		Dx = np.matmul(D3, x3)
		B2y = np.matmul(B31, y2)
        
		outputs["y3"] = -Cz - Dx + B2y      


class MDA(om.Group):
	def setup(self):
		cycle = self.add_subsystem("cycle", om.Group(), promotes=["*"])
		cycle.add_subsystem("d1", disc1(), promotes_inputs = ["z", "y3", "x1"], promotes_outputs = ["y1"])
		cycle.add_subsystem("d2", disc2(), promotes_inputs = ["z", "y1", "x2"], promotes_outputs = ["y2"])
		cycle.add_subsystem("d3", disc3(), promotes_inputs = ["z", "y2", "x3"], promotes_outputs = ["y3"])
        
		self.add_subsystem("obj2", om.ExecComp("f2 = (1 + (9/29)*(sum(x1) + sum(x2) + sum(x3) + sum(y1) + sum(y2) + sum(y3) + z[1] + z[2] + z[3] + z[4] + z[5]))*(1 - (z[0]/(1 + (9/29)*(sum(x1) + sum(x2) + sum(x3) + sum(y1) + sum(y2) + sum(y3) + z[1] + z[2] + z[3] + z[4] + z[5])))**0.5)",
                                            x1 = x1_dummy, x2 = x2_dummy, x3 = x2_dummy, z = z_dummy, y1 = y1_dummy, y2 = y2_dummy, y3 = y1_dummy), 
                     promotes = ["x1", "x2", "x3", "y1", "y2", "y3", "z", "f2"])
		self.add_subsystem("obj1", om.ExecComp("f1 = z[0]", z = z_dummy), promotes = ["z", "f1"])

		cycle.set_input_defaults("z", z_initial)
		cycle.set_input_defaults("x1", x1_initial)
		cycle.set_input_defaults("x2", x2_initial)
		cycle.set_input_defaults("x3", x2_initial)
        
		cycle.nonlinear_solver = om.NewtonSolver(maxiter = 1000, solve_subsystems = True)
		cycle.linear_solver = om.DirectSolver()

prob = om.Problem()
prob.model = MDA()

# Specify problem driver (SLSQP)
# prob.driver = om.ScipyOptimizeDriver(optimizer="COBYLA", tol = 1e-3, disp = True)
prob.driver = om.pyOptSparseDriver(optimizer = 'NSGA2') 
prob.driver.opt_settings["maxGen"] = 100
prob.driver.opt_settings["PopSize"] = 100
seedVar = random.random()
prob.driver.opt_settings["seed"] = seedVar

# Add design variables
prob.model.add_design_var("z", lower = zLower, upper = zUpper)
prob.model.add_design_var("x1", lower = xLower, upper = xUpper)
prob.model.add_design_var("x2", lower = xLower, upper = xUpper)
prob.model.add_design_var("x3", lower = xLower, upper = xUpper)

prob.model.add_constraint("y1", lower = 0, upper = 1)
prob.model.add_constraint("y2", lower = 0, upper = 1)
prob.model.add_constraint("y3", lower = 0, upper = 1)

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
dv_x3_values = []

y1_vals = []
y2_vals = []
y3_vals = []

for case in driver_cases:
    dv_obj1_values.append(case['obj1.f1'])
    dv_obj2_values.append(case['obj2.f2'])
    dv_z_values.append(case['z'])
    dv_x1_values.append(case['x1'])
    dv_x2_values.append(case['x2'])
    dv_x3_values.append(case['x3'])
    
for case_id in solver_cases:
    case = cr.get_case(case_id)
    y1_vals.append(case['y1'])
    y2_vals.append(case['y2'])
    y3_vals.append(case['y3'])

dv_obj1_values = np.array(dv_obj1_values)
dv_obj2_values = np.array(dv_obj2_values)
objs = np.array(np.concatenate((dv_obj1_values, dv_obj2_values), axis = 1))
np.savetxt(path + 'obj.csv', objs, delimiter=',')

dv_z_values = np.array(dv_z_values)
zs = np.reshape(np.array(dv_z_values), (len(dv_z_values), 6))
np.savetxt(path + 'z.csv', zs, delimiter=',')

dv_x1_values = np.array(dv_x1_values)
dv_x2_values = np.array(dv_x2_values)
dv_x3_values = np.array(dv_x3_values)

y1_vals = np.array(y1_vals)
y2_vals = np.array(y2_vals)
y3_vals = np.array(y3_vals)

x1s = np.reshape(np.array(dv_x1_values), (len(dv_x1_values), 4))
x2s = np.reshape(np.array(dv_x2_values), (len(dv_x2_values), 4))
x3s = np.reshape(np.array(dv_x3_values), (len(dv_x3_values), 4))

y1s = np.reshape(np.array(y1_vals), (len(y1_vals), 4))
y2s = np.reshape(np.array(y2_vals), (len(y2_vals), 4))
y3s = np.reshape(np.array(y3_vals), (len(y3_vals), 4))

np.savetxt(path + 'x1.csv', x1s, delimiter=',')
np.savetxt(path + 'x2.csv', x2s, delimiter=',')
np.savetxt(path + 'x3.csv', x3s, delimiter=',')

np.savetxt(path + 'y1.csv', y1s, delimiter=',')
np.savetxt(path + 'y2.csv', y2s, delimiter=',')
np.savetxt(path + 'y3.csv', y3s, delimiter=',')

# np.savetxt(path + 'seed.out', np.array(seedVar))

data = pd.read_csv('nsga2_best_pop.out', sep="\t", header=1)
data.to_csv(path + 'nsga2_best_pop.csv')