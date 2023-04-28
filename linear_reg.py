import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    X = df['km']
    y = df['price']
    print(X, y)