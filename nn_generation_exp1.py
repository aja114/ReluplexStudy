import os
import sys
import getopt
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def target_function(x, target='square'):
	if target == 'square':
		out = np.square(x)
	if target == 'cosine':
		out = np.cos(x)
	if target == 'inverse':
		out = 1 / x

	return np.reshape(out, (-1, 1))


def generate_networks(model_repo, target='square'):
    # Set the parameters
    gamma = 0.001
    min_x = -10.0
    max_x = 10.0
    range_x = max_x-min_x
    number_train_samples = int(1e4)
    number_test_samples = int(1e3)

    # Prepare the training set for all the networks
    X_train = np.reshape(np.random.rand(
        number_train_samples)*range_x+min_x, (-1, 1))
    X_test = np.reshape(np.linspace(
        min_x, max_x, num=number_test_samples), (-1, 1))

    Y_train = target_function(X_train, target)
    Y_test = target_function(X_test, target)

    # Network definition

    # Network structure
    inputs = keras.Input(shape=(1,), name="Inputs")
    x = keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    outputs = keras.layers.Dense(1, name="Outputs")(x)

    # Network optmisers
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=gamma)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{target}')
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=["mean_squared_error"])

    with open(f'{model_repo}/model_summary.txt', 'w') as f:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.fit(x=X_train, y=Y_train, batch_size=128, epochs=30, verbose=1)

    preds = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(X_test, preds, color='red', linestyle='-',
            markersize=0.01, label='Predictions')
    ax.plot(X_test, Y_test, color='blue', linestyle='--',
            markersize=0.01, label='GroundTruth')
    ax.legend(loc='lower right')
    ax.set(title='Predictions vs GroundTruth', ylabel='y', xlabel='x')
    ax.xaxis.set(ticks=range(int(min_x), int(max_x), int(range_x/5)))
    ax.tick_params(axis='y', direction='inout', length=10)
    fig.savefig(f'{model_repo}/model_results.png', dpi=fig.dpi)

    model.save(f"{model_repo}/model")


if __name__ == "__main__":
    target = sys.argv[1]

    # remove and create a new repo for the models
    if not os.path.exists('models_exp1'):
        os.makedirs('models_exp1')
    model_repo = f'models_exp1/model_{target}'
    os.system(f"rm -rf {model_repo} && mkdir {model_repo}")

    generate_networks(model_repo=model_repo, target=target)
