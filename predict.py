from utils import *

def load_thetas(path='result_thetas.pickle'):
	pass

if __name__=="__main__":
	try:
		thetha = load_thetas()
	except:
		theta = np.zeros((2, 1))