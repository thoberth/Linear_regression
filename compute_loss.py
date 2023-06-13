from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import *

if __name__=="__main__":
	# RECUPERATION DES DONNÉES
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

	# DENORMALISATION ET PREDICTION DES RESULTATS
	y_hat = denormalize(predict(thetas, x_norm), y_mean, y_std)

	# CALCUL DES PERFORMANCES DU MODEL
	print("###MSE###")
	print(f"My mse {loss_mse(y, y_hat)}")
	print(f"sklearn mse {mean_squared_error(y, y_hat)}")
	print("###MAE###")
	print(f"My mae {loss_mae(y, y_hat)}")
	print(f"sklearn mae {mean_absolute_error(y, y_hat)}")
	print("###R2SCORE###")
	print(f"My r2score {loss_r2score(y, y_hat)}")
	print(f"sklearn r2score {r2_score(y, y_hat)}")
