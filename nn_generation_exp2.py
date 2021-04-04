import os
import sys
import getopt
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

targets = ['square', 'cosine', 'inverse']


def usage(message):
	print(message)
	signature = f'{os.path.basename(__file__)} -f <function_to_approximate> -n <number of networks>'
	print('\nThe program needs to be called in this way:')
	print(signature)
	print('\t-f')
	print('\t\tFollowing options:\n\t\t' + '\n\t\t'.join(targets))
	print('\t-n')
	print('\t\tAn integer between 1 and 10:\n\t\t')

	sys.exit(2)


def main(argv):
	try:
		opts, args = getopt.getopt(argv, "hf:n:", ["help", "f_target=", "num_networks="])
	except getopt.GetoptError as err:
		print(err)
		usage()
		sys.exit(2)

	found_f = False
	found_n = False

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		if opt in ("-f", "--f_target"):
			found_f = True
			target = arg
			if target not in targets:
				usage('function not found')
		
		if opt in ("-n", "--num_networks"):
			found_n = True
			try:
				n = int(arg)
			except ValueError:
				usage('n has to be an integer')

	if not found_f:
		usage('Missing target function')
	if not found_n:
		usage('Missing number of networks')

	print(f'\nGenerating {n} networks to appriximate {target}\n')
	
	# remove and create a new repo for the models
	if not os.path.exists('models_exp2'):
		os.makedirs('models_exp2')
	model_repo = f'models_exp2/model_{target}'
	os.system(f"rm -rf {model_repo} && mkdir {model_repo}")
	generate_networks(target, n, model_repo)


def target_function(x, target='square'):
	if target == 'square':
		out = np.square(x)
	if target == 'cosine':
		out = np.cos(x)
	if target == 'inverse':
		out = 1 / x

	return np.reshape(out, (-1, 1))

def generate_networks(target='square', num=2, model_repo='model'):
	# Set the parameters
	gamma = 0.001
	min_x = -10.0
	max_x = 10.0
	range_x = max_x-min_x
	number_train_samples = int(1e4)
	number_test_samples = int(1e3)


	# Prepare the training set for all the networks 
	X_train = np.reshape(np.random.rand(number_train_samples)*range_x+min_x, (-1, 1))
	X_test = np.reshape(np.linspace(min_x, max_x, num=number_test_samples), (-1, 1))

	Y_train = target_function(X_train, target)
	Y_test = target_function(X_test, target)

	# Network definition

	for i in range(num):
		# Network structure
		inputs = keras.Input(shape=(1,), name="Inputs")
		x = keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
		for j in range(i):
			x = keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

		outputs = keras.layers.Dense(1, name="Outputs")(x)

		# Network optmisers
		loss = tf.keras.losses.MeanSquaredError()
		optimizer = keras.optimizers.Adam(learning_rate=gamma)

		model = keras.Model(inputs = inputs, outputs = outputs, name=f'{target}')
		model.compile(optimizer = optimizer, loss = loss, metrics = ["mean_squared_error"])

		with open(f'{model_repo}/model{i}_summary.txt','w') as f:
			# Pass the file handle in as a lambda function to make it callable
			model.summary(print_fn=lambda x: f.write(x + '\n'))


		model.fit(x=X_train, y=Y_train, batch_size=128, epochs=10, verbose=1)

		preds = model.predict(X_test)

		fig, ax = plt.subplots(figsize=(15, 10))
		ax.plot(X_test, preds, color='red', linestyle='-', markersize=0.01, label='Predictions')
		ax.plot(X_test, Y_test, color='blue', linestyle='--', markersize=0.01, label='GroundTruth')
		ax.legend(loc='lower right')
		ax.set(title='Predictions vs GroundTruth', ylabel='y', xlabel='x')
		ax.xaxis.set(ticks=range(int(min_x), int(max_x), int(range_x/5)))
		ax.tick_params(axis='y', direction='inout', length=10)
		fig.savefig(f'{model_repo}/model{i}_results.png', dpi=fig.dpi)
		
		model.save(f"{model_repo}/model{i}")

	# The marabou framework is installed on a VM so the results are sent there
	# Set to false to reproduce
	copy2vm = False
	if copy2vm:
		os.system(f"scp -r {model_repo}/ alex@192.168.1.18:~/Reluplex/experiments")


if __name__ == "__main__":
	main(sys.argv[1:])

