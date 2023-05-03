import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def normalize(x) -> np.ndarray:
	mean = np.sum(x) / len(x)
	std = np.sqrt(np.sum((x - mean)**2)/ len(x))
	x_norm = (x - mean) / (std)
	return x_norm, mean, std

def denormalize(x_norm, mean, std) -> np.ndarray:
	x = x_norm * std + mean
	return x

def loss_function(y_hat, y) -> float:
	loss = (np.sum((y_hat - y) ** 2)) / (2*len(y))
	return loss

def predict(theta, x) -> np.ndarray:
	X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	y_hat = X.dot(theta)
	return y_hat

def train(x, y, theta, lr=0.01, iter=1000) -> np.ndarray:
	X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	for _ in tqdm(range(iter)):
		y_hat = predict(theta, x)
		gradient = (1 / len(y)) * (X.T.dot((y_hat - y)))
		theta = theta - lr * gradient
	return theta

if __name__ == "__main__":
	## RECUPERATION DES DONNÉES
	df = pd.read_csv('data.csv')
	X = np.array(df['km']).reshape(-1, 1).astype('float64')
	y = np.array(df['price']).reshape(-1, 1).astype('float64')

	# INITIALISATION DE THETA
	theta = np.zeros((2, 1)).astype('float64')

	# NORMALISATION DES DONNÉES
	x_norm, x_mean, x_std = normalize(X)
	y_norm, y_mean, y_std = normalize(y)

	# PREMIERE PREDICTION SANS AVOIR ENTRAINER L'ALGO
	y_hat = predict(theta, X)

	# ENTRAINEMENT DES DONNÉES
	theta = train(x_norm, y_norm, theta)

	# SECONDE PREDICTION APRES ENTRAINEMENT DES DONNÉES
	y_hat2 = predict(theta, x_norm)

	# CALCUL DE LA FONCTION COUT
	loss = loss_function(y_hat2, y_norm)

	# DENORMALISATION DE NOS DONNÉES
	y_hat2 = denormalize(y_hat2, y_mean, y_std)

	# AFFICHAGE DES RESULTATS
	fig, ax = plt.subplots()
	ax.scatter(X, y, label='Données réelles')
	ax.plot(X, y_hat, c='orange', label="Droite de prediction avant entrainement")
	ax.plot(X, y_hat2, c='blue', label="Droite de prediction apres entrainement")
	ax.set_title("Calcul des erreurs (mse) : {}".format(loss))
	ax.legend()
	plt.show()