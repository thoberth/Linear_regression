import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import sys

 
def normalize(x) -> np.ndarray:
	mean = np.sum(x) / len(x)
	std = np.sqrt(np.sum((x - mean)**2)/ len(x))
	x_norm = (x - mean) / (std)
	return x_norm, mean, std

def denormalize(x_norm, mean, std) -> np.ndarray:
	x = x_norm * std + mean
	return x

def loss_mse(y, y_hat) -> float:
	loss = (np.sum((y_hat - y) ** 2)) / (len(y))
	return loss

def loss_mae(y, y_hat) -> float:
	absolute_error = abs(y_hat - y)
	sum_absolute_error = sum(absolute_error)
	mae = sum_absolute_error / (y.size)
	return mae

def loss_r2score(y, y_hat) -> float:  # coeff de determination
	sum_squared_regression = sum((y - y_hat)**2)
	total_sum_of_squares = sum((y - (sum(y) / y.shape[0]))**2)
	r2score_var = 1 - (sum_squared_regression/total_sum_of_squares)
	return r2score_var

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

def load_thetas(path='result_thetas.pickle'):
	with open(path, 'rb') as f:
		data = pickle.load(f)
	return data

def save_thetas(thetas, path='result_thetas.pickle'):
	with open(path, 'wb') as f:
		pickle.dump(thetas, f)
