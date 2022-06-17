#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:40:43 2022

@author: victoria
"""

import openmdao.api as om
import numpy as np
import random
import pandas as pd
import sys

i = sys.argv[1]
print('******************************' + str(sys.argv[1]) + '******************************')

path = "[Run " + str(i) + "] DV 4 disciplines"

# Set parameters
N = 4
n_z = 6 # Number of global design variables
n_x1 = 3 # Number of local design variables in discipline 1
n_y1 = 3 # Number of linking variables in discipline 1
n_x2 = 3 # Number of local design variables in discipline 1
n_y2 = 3 # Number of linking variables in discipline 1
n_x3 = 3 # Number of local design variables in discipline 1
n_y3 = 3 # Number of linking variables in discipline 1
n_x4 = 3 # Number of local design variables in discipline 1
n_y4 = 3 # Number of linking variables in discipline 1

xLower = np.zeros(n_x1)
xUpper = np.ones(n_x1)
zLower = np.zeros(n_z)
zUpper = np.ones(n_z)

# Set initial guesses
z_initial = np.array(random.sample(range(100), n_z))/100
x1_initial = np.array(random.sample(range(100), n_x1))/100
x2_initial = np.array(random.sample(range(100), n_x2))/100
x3_initial = np.array(random.sample(range(100), n_x3))/100
x4_initial = np.array(random.sample(range(100), n_x4))/100

# Set dummy values
z_dummy = np.zeros((n_z, 1))
x1_dummy = np.ones((n_x1, 1))
y1_dummy = np.ones((n_y1, 1))
x2_dummy = np.ones((n_x2, 1))
y2_dummy = np.ones((n_y2, 1))
x3_dummy = np.ones((n_x3, 1))
y3_dummy = np.ones((n_y3, 1))
x4_dummy = np.ones((n_x4, 1))
y4_dummy = np.ones((n_y4, 1))

d1Matrices = np.zeros((N*n_x1, n_x1 + n_z))
b1Matrices = np.zeros((N*n_x1, n_y1))

bArray = np.random.randint(10, size=(N*n_x1, n_y1))
dArray = np.random.randint(10, size=(N*n_x1, n_x1 + n_z))

bArray = np.array([[ 3,  1,  5],
       [ 5,  5,  8],
       [ 4,  6,  4],
       [ 2,  5,  1],
       [ 6,  3,  7],
       [ 8,  6,  6],
       [10,  1,  7],
       [10,  6,  9],
       [ 1,  7,  2],
       [ 3, 10,  3],
       [ 7,  4,  8],
       [ 3,  7,  9]])
dArray = np.array([[ 5,  6,  6,  5,  5,  6,  6,  7,  4],
       [ 9,  7,  2,  6,  4, 10,  6,  7, 10],
       [ 6,  1,  4,  1,  6,  1, 10,  7,  1],
       [ 9,  7,  9,  5,  7,  3,  1, 10,  6],
       [ 1,  5,  1,  3, 10,  5,  4, 10,  4],
       [ 4, 10,  1,  2,  1,  6,  7, 10,  7],
       [ 9,  1,  3,  9,  7,  7,  5,  8,  7],
       [ 4,  6,  3,  6,  5,  8,  1, 10,  6],
       [ 2,  5,  2,  8,  1, 10,  6, 10, 10],
       [ 4,  2,  9,  9,  7,  5,  1,  2,  5],
       [ 2,  9,  7,  2,  8,  8,  8,  5,  5],
       [ 2,  7,  6,  7,  7,  6,  8,  6,  5]])

# A and D arrays..
for i in range(0, N*n_x1):
    d1Matrices[i] = dArray[i:(i+1), :]/np.sum(dArray[i:(i+1), :])
    
# B arrays
for i in range(0, N*n_x1):
    b1Matrices[i] = 2*bArray[i:(i+1), :]/np.sum(bArray[i:(i+1), :])

d1Matrices = np.concatenate((b1Matrices, d1Matrices), axis = 1)

B1 = d1Matrices[0:3, 0:3]
D1 = d1Matrices[0:3, 3:6]
C1 = d1Matrices[0:3, 6:11]

B2 = d1Matrices[3:6, 0:3]
D2 = d1Matrices[3:6, 3:6]
C2 = d1Matrices[3:6, 6:11]

B3 = d1Matrices[6:9, 0:3]
D3 = d1Matrices[6:9, 3:6]
C3 = d1Matrices[6:9, 6:11]

B4 = d1Matrices[9:12, 0:3]
D4 = d1Matrices[9:12, 3:6]
C4 = d1Matrices[9:12, 6:11]


class disc1(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x1", val = x1_dummy)
		self.add_input("y4", val = y2_dummy)

		self.add_output("y1", val = y1_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x1 = inputs["x1"]
		y4 = inputs["y4"]
        
		Cz = np.reshape(np.matmul(C1, z[1:6]), [C1.shape[0], 1])
		Dx = np.matmul(D1, x1)
		B1y = np.matmul(B1, y4)
        
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
        
		Cz = np.reshape(np.matmul(C2, z[1:6]), [C2.shape[0], 1])
		Dx = np.matmul(D2, x2)
		B2y = np.matmul(B2, y1)
        
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
        
		Cz = np.reshape(np.matmul(C3, z[1:6]), [C3.shape[0], 1])
		Dx = np.matmul(D3, x3)
		B2y = np.matmul(B3, y2)
        
		outputs["y3"] = -Cz - Dx + B2y      

class disc4(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x4", val = x2_dummy)
		self.add_input("y3", val = y3_dummy)

		self.add_output("y4", val = y2_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x4 = inputs["x4"]
		y3 = inputs["y3"]
        
		Cz = np.reshape(np.matmul(C4, z[1:6]), [C4.shape[0], 1])
		Dx = np.matmul(D4, x4)
		B2y = np.matmul(B4, y3)
        
		outputs["y4"] = -Cz - Dx + B2y   
        

class MDA(om.Group):
	def setup(self):
		cycle = self.add_subsystem("cycle", om.Group(), promotes=["*"])
		cycle.add_subsystem("d1", disc1(), promotes_inputs = ["z", "y4", "x1"], promotes_outputs = ["y1"])
		cycle.add_subsystem("d2", disc2(), promotes_inputs = ["z", "y1", "x2"], promotes_outputs = ["y2"])
		cycle.add_subsystem("d3", disc3(), promotes_inputs = ["z", "y2", "x3"], promotes_outputs = ["y3"])
		cycle.add_subsystem("d4", disc4(), promotes_inputs = ["z", "y3", "x4"], promotes_outputs = ["y4"])
        
		self.add_subsystem("obj2", om.ExecComp("f2 = (1 + (9/29)*(sum(x1) + sum(x2) + sum(x3) + sum(x4) + sum(y1) + sum(y2) + sum(y3) + sum(y4) + z[1] + z[2] + z[3] + z[4] + z[5]))*(1 - (z[0]/(1 + (9/29)*(sum(x1) + sum(x2) + sum(x3) + sum(x4) + sum(y1) + sum(y2) + sum(y3) + sum(y4) + z[1] + z[2] + z[3] + z[4] + z[5])))**0.5)",
                                            x1 = x1_dummy, x2 = x2_dummy, x3 = x3_dummy, x4 = x4_dummy, z = z_dummy, y1 = y1_dummy, y2 = y2_dummy, y3 = y1_dummy, y4 = y4_dummy), 
                     promotes = ["x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4", "z", "f2"])
		self.add_subsystem("obj1", om.ExecComp("f1 = z[0]", z = z_dummy), promotes = ["z", "f1"])

		cycle.set_input_defaults("z", z_initial)
		cycle.set_input_defaults("x1", x1_initial)
		cycle.set_input_defaults("x2", x2_initial)
		cycle.set_input_defaults("x3", x3_initial)
		cycle.set_input_defaults("x4", x4_initial)
        
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
prob.model.add_design_var("x4", lower = xLower, upper = xUpper)

prob.model.add_constraint("y1", lower = 0, upper = 1)
prob.model.add_constraint("y2", lower = 0, upper = 1)
prob.model.add_constraint("y3", lower = 0, upper = 1)
prob.model.add_constraint("y4", lower = 0, upper = 1)

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
dv_x4_values = []

y1_vals = []
y2_vals = []
y3_vals = []
y4_vals = []

for case in driver_cases:
    dv_obj1_values.append(case['obj1.f1'])
    dv_obj2_values.append(case['obj2.f2'])
    dv_z_values.append(case['z'])
    dv_x1_values.append(case['x1'])
    dv_x2_values.append(case['x2'])
    dv_x3_values.append(case['x3'])
    dv_x4_values.append(case['x4'])
    
for case_id in solver_cases:
    case = cr.get_case(case_id)
    y1_vals.append(case['y1'])
    y2_vals.append(case['y2'])
    y3_vals.append(case['y3'])
    y4_vals.append(case['y4'])

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
dv_x4_values = np.array(dv_x4_values)

y1_vals = np.array(y1_vals)
y2_vals = np.array(y2_vals)
y3_vals = np.array(y3_vals)
y4_vals = np.array(y4_vals)

x1s = np.reshape(np.array(dv_x1_values), (len(dv_x1_values), 3))
x2s = np.reshape(np.array(dv_x2_values), (len(dv_x2_values), 3))
x3s = np.reshape(np.array(dv_x3_values), (len(dv_x3_values), 3))
x4s = np.reshape(np.array(dv_x4_values), (len(dv_x4_values), 3))

y1s = np.reshape(np.array(y1_vals), (len(y1_vals), 3))
y2s = np.reshape(np.array(y2_vals), (len(y2_vals), 3))
y3s = np.reshape(np.array(y3_vals), (len(y3_vals), 3))
y4s = np.reshape(np.array(y4_vals), (len(y4_vals), 3))

np.savetxt(path + 'x1.csv', x1s, delimiter=',')
np.savetxt(path + 'x2.csv', x2s, delimiter=',')
np.savetxt(path + 'x3.csv', x3s, delimiter=',')
np.savetxt(path + 'x4.csv', x4s, delimiter=',')

np.savetxt(path + 'y1.csv', y1s, delimiter=',')
np.savetxt(path + 'y2.csv', y2s, delimiter=',')
np.savetxt(path + 'y3.csv', y3s, delimiter=',')
np.savetxt(path + 'y4.csv', y4s, delimiter=',')

# np.savetxt(path + 'seed.out', np.array(seedVar))
data = pd.read_csv('nsga2_best_pop.out', sep="\t", header=1)
data.to_csv(path + 'nsga2_best_pop.csv')