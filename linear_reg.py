from utils import *

if __name__ == "__main__":
	## RECUPERATION DES DONNÉES
	df = pd.read_csv('data.csv')
	X = np.array(df['km']).reshape(-1, 1).astype('float64')
	y = np.array(df['price']).reshape(-1, 1).astype('float64')

	# INITIALISATION DE THETA0 ET THETA1
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
