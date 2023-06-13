from utils import *

if __name__=="__main__":
	mileage = input('Entrez un Kilometrage : ')
	while (mileage.isnumeric() != True):
		mileage = input("Veuillez entrer une valeur correcte : ")
	try:
		data = load_thetas()
		theta = np.array(data['thetas'])
		y_mean = data['y_mean']
		y_std = data['y_std']
		x_mean = data['x_mean']
		x_std = data['x_std']
		mileage = (int(mileage) - x_mean) / x_std
		prix = float(theta[0] + (theta[1] * mileage))
		prix = denormalize(prix, y_mean, y_std)
		print('Chargement de la valeur des thetas apres entrainements ...')
	except Exception as e:
		print('Aucun entrainement détecté, la valeur des thetas est 0')
		theta = np.zeros((2, 1))
		prix = float(theta[0] + (theta[1] * int(mileage)))
	print(f'Le prix estimé est {prix:.2f}.')