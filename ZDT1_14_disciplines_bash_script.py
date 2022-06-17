#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:29:28 2022

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

t0 = time.time()
path = "[Run " + str(i) + "] DV 14 disciplines"
xLower = 0
xUpper = 1
zLower = 0
zUpper = 1

# Set parameters
N = 14
n_z = 2 # Number of global design variables
n_x1 = 1 # Number of local design variables in discipline 1
n_y1 = 1 # Number of linking variables in discipline 1

# Set initial guesses
seedVar = random.random()
z_initial = np.array(random.sample(range(100), n_z))/100
x_initial = np.array(random.sample(range(100), n_x1))/100

# Set dummy values
z_dummy = np.ones((n_z, 1))
x_dummy = np.ones((n_x1, 1))
y_dummy = np.ones((n_y1, 1))

d1Matrices = np.zeros((N*n_x1, n_x1 + n_z))
b1Matrices = np.zeros((N*n_x1, n_y1))

dArray = np.array([[ 7,  1,  5],
       [10,  5,  5],
       [ 4,  7,  3],
       [10,  4,  2],
       [ 9,  8,  4],
       [ 5,  7,  5],
       [ 2,  7,  1],
       [ 9,  4,  2],
       [10,  1,  9],
       [ 2,  3,  2],
       [ 7,  8, 10],
       [ 7,  1,  7],
       [ 2,  3, 10],
       [ 5,  5,  5]])
bArray = np.array([[10],
       [ 7],
       [ 9],
       [ 2],
       [ 9],
       [ 9],
       [ 5],
       [ 2],
       [ 8],
       [ 8],
       [ 7],
       [ 5],
       [ 7],
       [ 6]])

# A and D arrays..
for i in range(0, N*n_x1):
    d1Matrices[i] = dArray[i:(i+1), :]/np.sum(dArray[i:(i+1), :])
    
# B arrays
for i in range(0, N*n_x1):
    b1Matrices[i] = 2*bArray[i:(i+1), :]/np.sum(bArray[i:(i+1), :])

d1Matrices = np.concatenate((b1Matrices, d1Matrices), axis = 1)

B1 = d1Matrices[0, 0]
C1 = d1Matrices[0, 1:2]
D1 = d1Matrices[0, 3]

B2 = d1Matrices[1, 0]
C2 = d1Matrices[1, 1:2]
D2 = d1Matrices[1, 3]

B3 = d1Matrices[2, 0]
C3 = d1Matrices[2, 1:2]
D3 = d1Matrices[2, 3]

B4 = d1Matrices[3, 0]
C4 = d1Matrices[3, 1:2]
D4 = d1Matrices[3, 3]

B5 = d1Matrices[4, 0]
C5 = d1Matrices[4, 1:2]
D5 = d1Matrices[4, 3]

B6 = d1Matrices[5, 0]
C6 = d1Matrices[5, 1:2]
D6 = d1Matrices[5, 3]

B7 = d1Matrices[6, 0]
C7 = d1Matrices[6, 1:2]
D7 = d1Matrices[6, 3]

B8 = d1Matrices[7, 0]
C8 = d1Matrices[7, 1:2]
D8 = d1Matrices[7, 3]

B9 = d1Matrices[8, 0]
C9 = d1Matrices[8, 1:2]
D9 = d1Matrices[8, 3]

B10 = d1Matrices[9, 0]
C10 = d1Matrices[9, 1:2]
D10 = d1Matrices[9, 3]

B11 = d1Matrices[10, 0]
C11 = d1Matrices[10, 1:2]
D11 = d1Matrices[10, 3]

B12 = d1Matrices[11, 0]
C12 = d1Matrices[11, 1:2]
D12 = d1Matrices[11, 3]

B13 = d1Matrices[12, 0]
C13 = d1Matrices[12, 1:2]
D13 = d1Matrices[12, 3]

B14 = d1Matrices[13, 0]
C14 = d1Matrices[13, 1:2]
D14 = d1Matrices[13, 3]


class disc1(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x1", val = x_dummy)
		self.add_input("y14", val = y_dummy)

		self.add_output("y1", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x1 = inputs["x1"]
		y14 = inputs["y14"]
        
		outputs["y1"] = -D1*x1 - np.matmul(C1, z[1:2]) + B1*y14

class disc2(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x2", val = x_dummy)
		self.add_input("y1", val = y_dummy)

		self.add_output("y2", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x2 = inputs["x2"]
		y1 = inputs["y1"]
        
		outputs["y2"] = -D2*x2 - np.matmul(C2, z[1:2]) + B2*y1
        
class disc3(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x3", val = x_dummy)
		self.add_input("y2", val = y_dummy)

		self.add_output("y3", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x3 = inputs["x3"]
		y2 = inputs["y2"]
        
		outputs["y3"] = -D3*x3 - np.matmul(C3, z[1:2]) + B3*y2

class disc4(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x4", val = x_dummy)
		self.add_input("y3", val = y_dummy)

		self.add_output("y4", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x4 = inputs["x4"]
		y3 = inputs["y3"]
        
		outputs["y4"] = -D4*x4 - np.matmul(C4, z[1:2]) + B4*y3

class disc5(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x5", val = x_dummy)
		self.add_input("y4", val = y_dummy)

		self.add_output("y5", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x5"]
		y = inputs["y4"]
        
		outputs["y5"] = -D5*x - np.matmul(C5, z[1:2]) + B5*y

class disc6(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x6", val = x_dummy)
		self.add_input("y5", val = y_dummy)

		self.add_output("y6", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x6"]
		y = inputs["y5"]
        
		outputs["y6"] = -D6*x - np.matmul(C6, z[1:2]) + B6*y

class disc7(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x7", val = x_dummy)
		self.add_input("y6", val = y_dummy)

		self.add_output("y7", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x7"]
		y = inputs["y6"]
        
		outputs["y7"] = -D7*x - np.matmul(C7, z[1:2]) + B7*y

class disc8(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x8", val = x_dummy)
		self.add_input("y7", val = y_dummy)

		self.add_output("y8", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x8"]
		y = inputs["y7"]
        
		outputs["y8"] = -D8*x - np.matmul(C8, z[1:2]) + B8*y

class disc9(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x9", val = x_dummy)
		self.add_input("y8", val = y_dummy)

		self.add_output("y9", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x9"]
		y = inputs["y8"]
        
		outputs["y9"] = -D9*x - np.matmul(C9, z[1:2]) + B9*y

class disc10(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x10", val = x_dummy)
		self.add_input("y9", val = y_dummy)

		self.add_output("y10", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x10"]
		y = inputs["y9"]
        
		outputs["y10"] = -D10*x - np.matmul(C10, z[1:2]) + B10*y

class disc11(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x11", val = x_dummy)
		self.add_input("y10", val = y_dummy)

		self.add_output("y11", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x11"]
		y = inputs["y10"]
        
		outputs["y11"] = -D11*x - np.matmul(C11, z[1:2]) + B11*y

class disc12(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x12", val = x_dummy)
		self.add_input("y11", val = y_dummy)

		self.add_output("y12", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x12"]
		y = inputs["y11"]
        
		outputs["y12"] = -D12*x - np.matmul(C12, z[1:2]) + B12*y

class disc13(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x13", val = x_dummy)
		self.add_input("y12", val = y_dummy)

		self.add_output("y13", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x = inputs["x13"]
		y = inputs["y12"]
        
		outputs["y13"] = -D13*x - np.matmul(C13, z[1:2]) + B13*y

class disc14(om.ExplicitComponent):
	def setup(self):
		self.add_input("z", val = z_dummy)
		self.add_input("x14", val = x_dummy)
		self.add_input("y13", val = y_dummy)

		self.add_output("y14", val = y_dummy)

	def setup_partials(self):
		self.declare_partials("*", "*", method = "fd")

	def compute(self, inputs, outputs):
		z = inputs["z"]
		x14 = inputs["x14"]
		y = inputs["y13"]
        
		outputs["y14"] = -D14*x14 - np.matmul(C14, z[1:2]) + B14*y

class MDA(om.Group):
	def setup(self):
		cycle = self.add_subsystem("cycle", om.Group(), promotes=["*"])
		cycle.add_subsystem("d1", disc1(), promotes_inputs = ["z", "y14", "x1"], promotes_outputs = ["y1"])
		cycle.add_subsystem("d2", disc2(), promotes_inputs = ["z", "y1", "x2"], promotes_outputs = ["y2"])
		cycle.add_subsystem("d3", disc3(), promotes_inputs = ["z", "y2", "x3"], promotes_outputs = ["y3"])
		cycle.add_subsystem("d4", disc4(), promotes_inputs = ["z", "y3", "x4"], promotes_outputs = ["y4"])
		cycle.add_subsystem("d5", disc5(), promotes_inputs = ["z", "y4", "x5"], promotes_outputs = ["y5"])
		cycle.add_subsystem("d6", disc6(), promotes_inputs = ["z", "y5", "x6"], promotes_outputs = ["y6"])
		cycle.add_subsystem("d7", disc7(), promotes_inputs = ["z", "y6", "x7"], promotes_outputs = ["y7"])
		cycle.add_subsystem("d8", disc8(), promotes_inputs = ["z", "y7", "x8"], promotes_outputs = ["y8"])
		cycle.add_subsystem("d9", disc9(), promotes_inputs = ["z", "y8", "x9"], promotes_outputs = ["y9"])
		cycle.add_subsystem("d10", disc10(), promotes_inputs = ["z", "y9", "x10"], promotes_outputs = ["y10"])
		cycle.add_subsystem("d11", disc11(), promotes_inputs = ["z", "y10", "x11"], promotes_outputs = ["y11"])
		cycle.add_subsystem("d12", disc12(), promotes_inputs = ["z", "y11", "x12"], promotes_outputs = ["y12"])
		cycle.add_subsystem("d13", disc13(), promotes_inputs = ["z", "y12", "x13"], promotes_outputs = ["y13"])
		cycle.add_subsystem("d14", disc14(), promotes_inputs = ["z", "y13", "x14"], promotes_outputs = ["y14"])
        
		self.add_subsystem("obj2", om.ExecComp("f2 = (1 + (9/29)*(x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + z[1]))*(1 - (z[0]/(1 + (9/29)*(x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + z[1])))**0.5)",
                                            x1 = x_dummy, x2 = x_dummy, x3 = x_dummy, x4 = x_dummy, x5 = x_dummy, x6 = x_dummy, x7 = x_dummy, x8 = x_dummy, x9 = x_dummy, x10 = x_dummy, x11 = x_dummy, x12 = x_dummy, x13 = x_dummy, x14 = x_dummy, 
                                            
                                            z = z_dummy), promotes = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10", "y11", "y12", "y13", "y14", "z", "f2"])
		self.add_subsystem("obj1", om.ExecComp("f1 = z[0]", z = z_dummy), promotes = ["z", "f1"])

		cycle.set_input_defaults("z", z_initial)
		cycle.set_input_defaults("x1", x_initial)
		cycle.set_input_defaults("x2", x_initial)
		cycle.set_input_defaults("x3", x_initial)
		cycle.set_input_defaults("x4", x_initial)
		cycle.set_input_defaults("x5", x_initial)
		cycle.set_input_defaults("x6", x_initial)
		cycle.set_input_defaults("x7", x_initial)
		cycle.set_input_defaults("x8", x_initial)
		cycle.set_input_defaults("x9", x_initial)
		cycle.set_input_defaults("x10", x_initial)
		cycle.set_input_defaults("x11", x_initial)
		cycle.set_input_defaults("x12", x_initial)
		cycle.set_input_defaults("x13", x_initial)
		cycle.set_input_defaults("x14", x_initial)
        
		cycle.nonlinear_solver = om.NewtonSolver(maxiter = 1000, solve_subsystems = True)
		cycle.linear_solver = om.ScipyKrylov()

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
prob.model.add_design_var("x3", lower = xLower, upper = xUpper)
prob.model.add_design_var("x4", lower = xLower, upper = xUpper)
prob.model.add_design_var("x5", lower = xLower, upper = xUpper)
prob.model.add_design_var("x6", lower = xLower, upper = xUpper)
prob.model.add_design_var("x7", lower = xLower, upper = xUpper)
prob.model.add_design_var("x8", lower = xLower, upper = xUpper)
prob.model.add_design_var("x9", lower = xLower, upper = xUpper)
prob.model.add_design_var("x10", lower = xLower, upper = xUpper)
prob.model.add_design_var("x11", lower = xLower, upper = xUpper)
prob.model.add_design_var("x12", lower = xLower, upper = xUpper)
prob.model.add_design_var("x13", lower = xLower, upper = xUpper)
prob.model.add_design_var("x14", lower = xLower, upper = xUpper)


prob.model.add_constraint("y1", lower = xLower, upper = xUpper)
prob.model.add_constraint("y2", lower = xLower, upper = xUpper)
prob.model.add_constraint("y3", lower = xLower, upper = xUpper)
prob.model.add_constraint("y4", lower = xLower, upper = xUpper)
prob.model.add_constraint("y5", lower = xLower, upper = xUpper)
prob.model.add_constraint("y6", lower = xLower, upper = xUpper)
prob.model.add_constraint("y7", lower = xLower, upper = xUpper)
prob.model.add_constraint("y8", lower = xLower, upper = xUpper)
prob.model.add_constraint("y9", lower = xLower, upper = xUpper)
prob.model.add_constraint("y10", lower = xLower, upper = xUpper)
prob.model.add_constraint("y11", lower = xLower, upper = xUpper)
prob.model.add_constraint("y12", lower = xLower, upper = xUpper)
prob.model.add_constraint("y13", lower = xLower, upper = xUpper)
prob.model.add_constraint("y14", lower = xLower, upper = xUpper)

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
dv_x5_values = []
dv_x6_values = []
dv_x7_values = []
dv_x8_values = []
dv_x9_values = []
dv_x10_values = []
dv_x11_values = []
dv_x12_values = []
dv_x13_values = []
dv_x14_values = []

y1_vals = []
y2_vals = []
y3_vals = []
y4_vals = []
y5_vals = []
y6_vals = []
y7_vals = []
y8_vals = []
y9_vals = []
y10_vals = []
y11_vals = []
y12_vals = []
y13_vals = []
y14_vals = []
dv_obj1_values = []
dv_obj2_values = []

for case in driver_cases:
    dv_obj1_values.append(case['obj1.f1'])
    dv_obj2_values.append(case['obj2.f2'])
    dv_z_values.append(case['z'])
    dv_x1_values.append(case['x1'])
    dv_x2_values.append(case['x2'])
    dv_x3_values.append(case['x3'])
    dv_x4_values.append(case['x4'])
    dv_x5_values.append(case['x5'])
    dv_x6_values.append(case['x6'])
    dv_x7_values.append(case['x7'])
    dv_x8_values.append(case['x8'])
    dv_x9_values.append(case['x9'])
    dv_x10_values.append(case['x10'])
    dv_x11_values.append(case['x11'])
    dv_x12_values.append(case['x12'])
    dv_x13_values.append(case['x13'])
    dv_x14_values.append(case['x14'])


for case_id in solver_cases:
    case = cr.get_case(case_id)
    y1_vals.append(case['y1'])
    y2_vals.append(case['y2'])
    y3_vals.append(case['y3'])
    y4_vals.append(case['y4'])
    y5_vals.append(case['y5'])
    y6_vals.append(case['y6'])
    y7_vals.append(case['y7'])
    y8_vals.append(case['y8'])
    y9_vals.append(case['y9'])
    y10_vals.append(case['y10'])
    y11_vals.append(case['y11'])
    y12_vals.append(case['y12'])
    y13_vals.append(case['y13'])
    y14_vals.append(case['y14'])

dv_obj1_values = np.array(dv_obj1_values)
dv_obj2_values = np.array(dv_obj2_values)
objs = np.array(np.concatenate((dv_obj1_values, dv_obj2_values), axis = 1))
np.savetxt(path + 'obj.csv', objs, delimiter=',')

dv_z_values = np.array(dv_z_values)
zs = np.reshape(np.array(dv_z_values), (len(dv_z_values), 2))
np.savetxt(path + 'z.csv', zs, delimiter=',')

dv_x1_values = np.array(dv_x1_values)
dv_x2_values = np.array(dv_x2_values)
dv_x3_values = np.array(dv_x3_values)
dv_x4_values = np.array(dv_x4_values)
dv_x5_values = np.array(dv_x5_values)
dv_x6_values = np.array(dv_x6_values)
dv_x7_values = np.array(dv_x7_values)
dv_x8_values = np.array(dv_x8_values)
dv_x9_values = np.array(dv_x9_values)
dv_x10_values = np.array(dv_x10_values)
dv_x11_values = np.array(dv_x11_values)
dv_x12_values = np.array(dv_x12_values)
dv_x13_values = np.array(dv_x13_values)
dv_x14_values = np.array(dv_x14_values)

y1_vals = np.array(y1_vals)
y2_vals = np.array(y2_vals)
y3_vals = np.array(y3_vals)
y4_vals = np.array(y4_vals)
y5_vals = np.array(y5_vals)
y6_vals = np.array(y6_vals)
y7_vals = np.array(y7_vals)
y8_vals = np.array(y8_vals)
y9_vals = np.array(y9_vals)
y10_vals = np.array(y10_vals)
y11_vals = np.array(y11_vals)
y12_vals = np.array(y12_vals)
y13_vals = np.array(y13_vals)
y14_vals = np.array(y14_vals)

x1s = np.reshape(np.array(dv_x1_values), (len(dv_x1_values), 1))
x2s = np.reshape(np.array(dv_x2_values), (len(dv_x2_values), 1))
x3s = np.reshape(np.array(dv_x3_values), (len(dv_x3_values), 1))
x4s = np.reshape(np.array(dv_x4_values), (len(dv_x4_values), 1))
x5s = np.reshape(np.array(dv_x5_values), (len(dv_x5_values), 1))
x6s = np.reshape(np.array(dv_x6_values), (len(dv_x6_values), 1))
x7s = np.reshape(np.array(dv_x7_values), (len(dv_x7_values), 1))
x8s = np.reshape(np.array(dv_x8_values), (len(dv_x8_values), 1))
x9s = np.reshape(np.array(dv_x9_values), (len(dv_x9_values), 1))
x10s = np.reshape(np.array(dv_x10_values), (len(dv_x10_values), 1))
x11s = np.reshape(np.array(dv_x11_values), (len(dv_x11_values), 1))
x12s = np.reshape(np.array(dv_x12_values), (len(dv_x12_values), 1))
x13s = np.reshape(np.array(dv_x13_values), (len(dv_x13_values), 1))
x14s = np.reshape(np.array(dv_x14_values), (len(dv_x14_values), 1))

y1s = np.reshape(np.array(y1_vals), (len(y1_vals), 1))
y2s = np.reshape(np.array(y2_vals), (len(y2_vals), 1))
y3s = np.reshape(np.array(y3_vals), (len(y3_vals), 1))
y4s = np.reshape(np.array(y4_vals), (len(y4_vals), 1))
y5s = np.reshape(np.array(y5_vals), (len(y5_vals), 1))
y6s = np.reshape(np.array(y6_vals), (len(y6_vals), 1))
y7s = np.reshape(np.array(y7_vals), (len(y7_vals), 1))
y8s = np.reshape(np.array(y8_vals), (len(y8_vals), 1))
y9s = np.reshape(np.array(y9_vals), (len(y9_vals), 1))
y10s = np.reshape(np.array(y10_vals), (len(y10_vals), 1))
y11s = np.reshape(np.array(y11_vals), (len(y11_vals), 1))
y12s = np.reshape(np.array(y12_vals), (len(y12_vals), 1))
y13s = np.reshape(np.array(y13_vals), (len(y13_vals), 1))
y14s = np.reshape(np.array(y14_vals), (len(y14_vals), 1))

np.savetxt(path + 'x1.csv', x1s, delimiter=',')
np.savetxt(path + 'x2.csv', x2s, delimiter=',')
np.savetxt(path + 'x3.csv', x3s, delimiter=',')
np.savetxt(path + 'x4.csv', x4s, delimiter=',')
np.savetxt(path + 'x5.csv', x5s, delimiter=',')
np.savetxt(path + 'x6.csv', x6s, delimiter=',')
np.savetxt(path + 'x7.csv', x7s, delimiter=',')
np.savetxt(path + 'x8.csv', x8s, delimiter=',')
np.savetxt(path + 'x9.csv', x9s, delimiter=',')
np.savetxt(path + 'x10.csv', x10s, delimiter=',')
np.savetxt(path + 'x11.csv', x11s, delimiter=',')
np.savetxt(path + 'x12.csv', x12s, delimiter=',')
np.savetxt(path + 'x13.csv', x13s, delimiter=',')
np.savetxt(path + 'x14.csv', x14s, delimiter=',')

np.savetxt(path + 'y1.csv', y1s, delimiter=',')
np.savetxt(path + 'y2.csv', y2s, delimiter=',')
np.savetxt(path + 'y3.csv', y3s, delimiter=',')
np.savetxt(path + 'y4.csv', y4s, delimiter=',')
np.savetxt(path + 'y5.csv', y5s, delimiter=',')
np.savetxt(path + 'y6.csv', y6s, delimiter=',')
np.savetxt(path + 'y7.csv', y7s, delimiter=',')
np.savetxt(path + 'y8.csv', y8s, delimiter=',')
np.savetxt(path + 'y9.csv', y9s, delimiter=',')
np.savetxt(path + 'y10.csv', y10s, delimiter=',')
np.savetxt(path + 'y11.csv', y11s, delimiter=',')
np.savetxt(path + 'y12.csv', y12s, delimiter=',')
np.savetxt(path + 'y13.csv', y13s, delimiter=',')
np.savetxt(path + 'y14.csv', y14s, delimiter=',')

data = pd.read_csv('nsga2_best_pop.out', sep="\t", header=1)
data.to_csv(path + 'nsga2_best_pop.csv')