import os
from maraboupy import Marabou
from maraboupy import MarabouCore
import numpy as np
import pandas as pd
import time


def set_constraints(network, inputVars, outputVars, target):
	if target == "square":
		# Set input interval
		network.setLowerBound(inputVars, -10.0)
		network.setUpperBound(inputVars, 10.0)

		# Set output interval
		network.setLowerBound(outputVars, 0)
		network.setUpperBound(outputVars, 100)

	if target == "cosine":

		# Set input interval
		network.setLowerBound(inputVars, -10.0)
		network.setUpperBound(inputVars, 10.0)

		# Set output interval

		# upper bound on cosine: validate that no output is greater than 1
		upperBound = MarabouCore.Equation(MarabouCore.Equation.GE);
		upperBound.addAddend(1.0, outputVars);
		upperBound.setScalar(0);

		# lower bound on cosine: validate that no output is lower than -1
		lowerBound = MarabouCore.Equation(MarabouCore.Equation.LE);
		lowerBound.addAddend(-1.0, outputVars);
		lowerBound.setScalar(0);

		# ( output > 1) \/ ( output < -1 )
		disjunction = [[lowerBound], [upperBound]]
		network.addDisjunctionConstraint(disjunction)

	if target == "inverse":
		# Set input interval
		network.setLowerBound(inputVars, 0.0)
		network.setUpperBound(inputVars, 10.0)

	   	# Set output interval
		network.setUpperBound(outputVars, -0.5)


targets = ['square', 'cosine', 'log']
model_dirs = []
model_stats = {}

if not os.path.exists('models'):
	print("no models repository")
	sys.exit(2)

for target in targets:
	path = os.path.join('models', f'model_{target}')
	if not os.path.exists(path):
		print(f'models for target {target} don\'t exist')
	else:
		print(f'models for target {target} will go through validation')
		model_dirs.append((path, target))

for model_dir, target in model_dirs:
	models = []
	for file in os.listdir(model_dir):
		path = os.path.join(model_dir, file)
		if file.startswith("model") and os.path.isdir(path):
			models.append(path)

	for model in models:
		model_name = model.split("/")[-1]
		name = os.path.join(model_dir, model_name+'.log')
		
		print('*'*50)
		print(f'validating {name}')
		
		network = Marabou.MarabouNetworkTF(model, modelType="savedModel_v2")

		inputVars = network.inputVars[0][0]
		outputVars = network.outputVars[0]

		set_constraints(network, inputVars[0], outputVars[0], target)

		print("Input Lower Bounds: ", network.lowerBounds.get(inputVars[0], "None"))
		print("Input Upper Bounds: ", network.upperBounds.get(inputVars[0], "None"))
		print("Output Lower Bounds: ", network.lowerBounds.get(outputVars[0], "None"))
		print("Output Upper Bounds: ", network.upperBounds.get(outputVars[0], "None"))

		vals, stats = network.solve(name, verbose=True)

		input_sol = vals.get(inputVars[0], None)
		output_sol = vals.get(outputVars[0], None)

		print("input solution: ", input_sol)
		print("output solution: ", output_sol)

		model_stats[name] = (stats.getTotalTime(), stats.getNumSplits(),
							 stats.hasTimedOut(), stats.getMaxStackDepth(),
							 stats.getNumMainLoopIterations(), input_sol,
							 output_sol)

print('*'*50)

columns_label = ['running_time', 'num_splits', 'timed_out',
				 'stack_depth', 'main_loop_iter',
				 'input_solution', 'output_solution']
results = pd.DataFrame.from_dict(
	model_stats, orient='index', columns=columns_label)
results.to_csv('results_exp2.csv')
