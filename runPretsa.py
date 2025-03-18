import sys
from pretsa import Pretsa
from pretsa_star import Pretsa_star
import pandas as pd
import profile

filePath = "./dataset/bpic2013_homogeneous_dataset.csv"

# Chose these values of k and t since they were suggested in the paper. Lower value of t raises ValueError
k = 8
t = 1.0
sys.setrecursionlimit(3000)
targetFilePath = filePath.replace(".csv","_t%s_k%s_l4_pretsa_star.csv" % (t,k))

print("Load Event Log")
eventLog = pd.read_csv(filePath, delimiter=";")
print("Starting experiments")
pretsa = Pretsa_star(eventLog)
nonce = pretsa._generate_nonce()
cutOutCases = pretsa.runPretsa(int(k),float(t),nonce)
print("Modified " + str(len(cutOutCases)) + " cases for k=" + str(k))
privateEventLog = pretsa.getPrivatisedEventLog()
privateEventLog.to_csv(targetFilePath, sep=";",index=False)