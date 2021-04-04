import os
from maraboupy import Marabou
import numpy as np
import pandas as pd 

model = 'models/model_cosine/model2'

print("validating model: ", model, flush=True)

network = Marabou.MarabouNetworkTF(model, modelType="savedModel_v2")

inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0]

print("inputVars: ", inputVars)
print("outputVars: ", outputVars)

network.setLowerBound(inputVars[0], -10.0)
network.setUpperBound(inputVars[0], 10.0)

print(network.evaluateWithMarabou(np.array([0.0])))

print("Input Lower bounds: ", network.lowerBounds.get(inputVars[0],None))
print("Input Upper bounds: ", network.upperBounds.get(inputVars[0],None))

# network.setUpperBound(outputVars[0], 100.0)

print("Output Lower bounds: ", network.lowerBounds.get(outputVars[0],None))
print("Output Upper bounds: ", network.upperBounds.get(outputVars[0],None))

print(network.findError(np.array([5.0]), options=None, filename='evaluateWithMarabou.log'))

network.saveQuery("test_query.txt")

print(network.getMarabouQuery())	

vals, stats = network.solve(verbose=True)

input_sol = vals.get(inputVars[0], None)
output_sol = vals.get(outputVars[0], None)

# print("input_sol: ", input_sol)
# print("output_sol: ", output_sol)





