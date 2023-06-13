from utils import *


if __name__=="__main__":
	# RECUPERATION DES DONNÉES
	df = pd.read_csv('data.csv')
	X = np.array(df['km']).reshape(-1, 1).astype('float64')
	y = np.array(df['price']).reshape(-1, 1).astype('float64')

	# NORMALISATION DES DONNÉES
	x_norm, x_mean, x_std = normalize(X)
	y_norm, y_mean, y_std = normalize(y)

	# PREDICTION DES DONNÉES
	theta = np.zeros((2, 1))
	y_hat = predict(theta, x_norm)
	y_hat = denormalize(y_hat, y_mean, y_std)

	# AFFICHAGE DES RESULTATS (Partie Bonus)
	fig, ax = plt.subplots()
	ax.scatter(X, y, label='Données réelles')
	line, = ax.plot(X, y_hat, c='blue', label="Droite de prediction apres entrainement")
	ax.legend()
	plt.xlabel("Kilometrage")
	plt.ylabel("Prix")
	plt.draw()
	plt.pause(1)
	for _ in range(10):
		line.set_ydata(y_hat)
		theta = train(x_norm, y_norm, theta, lr=0.01, iter=50)
		y_hat = predict(theta, x_norm)
		y_hat = denormalize(y_hat, y_mean, y_std)
		line.set_ydata(y_hat)
		plt.draw()
		plt.pause(1)
	plt.show()