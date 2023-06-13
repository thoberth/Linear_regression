from utils import *

if __name__ == "__main__":
	## RECUPERATION DES DONNÉES
	df = pd.read_csv('data.csv')
	X = np.array(df['km']).reshape(-1, 1).astype('float64')
	y = np.array(df['price']).reshape(-1, 1).astype('float64')

	# INITIALISATION DE THETA0 ET THETA1
	thetas = np.zeros((2, 1)).astype('float64')

	# NORMALISATION DES DONNÉES
	x_norm, x_mean, x_std = normalize(X)
	y_norm, y_mean, y_std = normalize(y)

	# ENTRAINEMENT DES DONNÉES
	thetas = train(x_norm, y_norm, thetas)

	# ENREGISTREMENT DES THETAS ET DES VALEURS POUR DENORMALISER LE RESULTAT
	to_save = {
		"thetas": thetas, 
		"y_mean": y_mean,
		"y_std": y_std,
		"x_mean": x_mean,
		"x_std": x_std
		}
	save_thetas(to_save)