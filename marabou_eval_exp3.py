import os
from maraboupy import Marabou
import numpy as np
import pandas as pd


def set_constraints(network, inputVars, outputVars, target):
	if target == "square":
		# Set input interval
		network.setLowerBound(inputVars[0], -10.0)
		network.setUpperBound(inputVars[0], 10.0)

		# Set output interval
        network.setLowerBound(outputVars[0], -10)
		network.setUpperBound(outputVars[0], -0.5)

targets = ['square']

model_dirs = []
model_stats = {}

if not os.path.exists('models_exp3'):
	print("no models repository")
	sys.exit(2)

for target in targets:
	path = os.path.join('models_exp3', f'model_{target}')
	if not os.path.exists(path):
		print(f'models for target {target} don\'t exist')
	else:
		print(f'models for target {target} will go through validation')
		model_dirs.append(path)

for model_dir in model_dirs:
	models = []
	for file in os.listdir(model_dir):
		path = os.path.join(model_dir, file)
		if file.startswith("model") and os.path.isdir(path):
			models.append(path)

	for model in models:
		model_name = model.split("/")[-1]+'.log'
		# print("validating model: ", model_name, flush=True)
		network = Marabou.MarabouNetworkTF(model, modelType="savedModel_v2")

		inputVars = network.inputVars[0][0]
		outputVars = network.outputVars[0]

		# print("input vars: ", inputVars)
		# print("output vars: ", outputVars)

		set_constraints(network, inputVars, outputVars, target)

		name = os.path.join(model_dir, model_name)

		vals, stats = network.solve(name, verbose=True)

		input_sol = vals.get(inputVars[0], None)
		output_sol = vals.get(outputVars[0], None)

		model_stats[name] = (stats.getTotalTime(), stats.getNumSplits(),
							 stats.hasTimedOut(), stats.getMaxStackDepth(),
							 stats.getNumMainLoopIterations(), input_sol,
							 output_sol)

columns_label = ['running_time', 'num_splits', 'timed_out',
				 'stack_depth', 'main_loop_iter',
				 'input_solution', 'output_solution']
results = pd.DataFrame.from_dict(
	model_stats, orient='index', columns=columns_label)
results.to_csv('results_exp3.csv')
