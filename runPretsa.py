import sys
from pretsa import Pretsa
from pretsa_star import Pretsa_star
import pandas as pd
import profile

filePath = "./baselogs/test1.csv"
k = 8
t = 1.0
sys.setrecursionlimit(3000)
targetFilePath = filePath.replace(".csv","_t%s_k%s_pretsa_star.csv" % (t,k))


print("Load Event Log")
eventLog = pd.read_csv(filePath, delimiter=";")
print("Starting experiments")
pretsa = Pretsa_star(eventLog)
nonce = pretsa._generate_nonce()
cutOutCases = pretsa.runPretsa(int(k),float(t),nonce)
print("Modified " + str(len(cutOutCases)) + " cases for k=" + str(k))
privateEventLog = pretsa.getPrivatisedEventLog()
privateEventLog.to_csv(targetFilePath, sep=";",index=False)