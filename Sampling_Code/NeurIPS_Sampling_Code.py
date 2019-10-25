from GPyOpt.methods import BayesianOptimization
import numpy as np
import UserFun
import os
from datetime import datetime

api = 'IBM'
global alpha 
alpha = 0.6

timestr = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
resultsDIR = ''+api+'_'+str(alpha)+'_'+timestr
os.mkdir(resultsDIR)

imageDIR = resultsDIR+'\\images'
os.mkdir(imageDIR)

A = UserFun.UserFun(api, resultsDIR)

def f(x):
    # Small alpha means that the exploration is minimal.  If alpha is close to 1 then exploration is greater.

    # Run the face classification step:
    FDsuccess, GDsuccess, ImageLocation = A.UserFun([x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]])

    # Calculate the distances between the current point and the 50 most recently sampled points:
    dists = distCalculation(x)
    
    # Calculate MI_factor which is the exploration factor:
    if len(dists)>0:
        MI_factor = 1 - np.exp(-5*min(dists))
    else:
        MI_factor = 1

    # Calculate the final loss which is a trade-off between exploration (MI_factor) and exploitation (face classification loss).
    # Face classication was a success do:
    if GDsuccess == True:
        Xset.append(x)
        y = (1-alpha)*1 - alpha*MI_factor
    # If the face generation was a failure do (exception handling):
    elif FDsuccess==-1:
        y = 0*MI_factor
    # IF the face classiication was a failure do:
    else:
        Xset.append(x)
        y = -(1-alpha)*40 - alpha*MI_factor
    
    if len(Xset)>50:
        Xset.pop(0)
        
    return np.array(y)

# Distance calculation:
def distCalculation(x):
    result = np.zeros((len(Xset),1))
    count=0
    for xd in Xset:
        dist = np.sum(np.multiply((x[0] - xd),(x[0] - xd)))
        result[count] = dist
        count=count+1
    return result

# Bounds:
bounds =[{'name': 'Region1', 'type': 'continuous', 'domain':(0,1)},
        {'name': 'Region2', 'type': 'continuous', 'domain':(0,1)},
        {'name': 'Region3', 'type': 'continuous', 'domain':(0,1)},
        {'name': 'Region4', 'type': 'continuous', 'domain':(0,1)},
        {'name': 'Gender1', 'type': 'continuous', 'domain':(0,1)},
        {'name': 'Gender2', 'type': 'continuous', 'domain':(0,1)}]

Xset = "global"
Yset = "global"
Xset = []
myBopt = BayesianOptimization(f, domain=bounds)
myBopt.run_optimization(max_iter= 1000)
    
np.savetxt(resultsDIR + '\\X' + api + '_bayesian.txt', myBopt.X)
np.savetxt(resultsDIR + '\\Y' + api + '_bayesian.txt', myBopt.Y)