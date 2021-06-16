import pandas as pd
import numpy as np

df = pd.read_csv("/home/aisys/checkpoint.csv").drop(['Unnamed: 0'], axis = 1)

#print(round(df['time'][0],6))
#print(df)

pvalue = 1 

print("\n\n")
print("                      SATA        NVME        PMEM\n")
vvvv = 0
print("DPN92 - TrainLoad: ","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
vvvv = 1
#print("DPN92 - TestLoad:  ","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
vvvv = 2
print("DPN92 - torch.load:","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
print("")
vvvv = 9
print("  VGG - TrainLoad: ","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
vvvv = 10
#print("  VGG - TestLoad:  ","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
vvvv = 11
print("  VGG - torch.load:","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
print("")
vvvv = 18
print("S_DLA - TrainLoad: ","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
vvvv = 19
#print("S_DLA - TestLoad:  ","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
vvvv = 20
print("S_DLA - torch.load:","{:.6f}".format(df['time'][vvvv]),"  ","{:.6f}".format(df['time'][vvvv+3]),"  ","{:.6f}".format(df['time'][vvvv+6]*pvalue))
