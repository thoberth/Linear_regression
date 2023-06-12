import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import sys

#lady radia

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