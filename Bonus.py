from utils import *

# AFFICHAGE DES RESULTATS (Partie Bonus)
	fig, ax = plt.subplots()
	ax.scatter(X, y, label='Données réelles')
	ax.plot(X, y_hat, c='orange', label="Droite de prediction avant entrainement")
	ax.plot(X, y_hat2, c='blue', label="Droite de prediction apres entrainement")
	ax.set_title("Calcul des erreurs (mse) : {}".format(loss))
	ax.legend()
	plt.show()