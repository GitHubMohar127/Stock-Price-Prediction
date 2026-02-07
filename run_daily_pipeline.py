import os
import warnings
warnings.filterwarnings("ignore")

os.system("python fetch_data.py")
os.system("python preprocessing.py")
os.system("python train_ML.py")

print("All stocks pipeline completed")
