import matplotlib.pyplot as plt
import numpy as np
import csv

scored = []
Gdiff = []
goalkRate = []
with open('/home/michael/Documents/Mcgill_Courses/PHYS 321/PHYS321-final-project/cristiano-ronaldo-stats.csv') as csvfile:

    reader = csv.reader(csvfile, delimiter=',')

    firstLine = True
    for row in reader:
        if firstLine:
            firstLine=False
            continue
        scored.append(row[0])
        Gdiff.append(row[1])
        goalkRate.append(row[2])

temp = np.array(scored)
scored = temp.astype(np.float)
temp = np.array(Gdiff)
Gdiff = temp.astype(np.float)
temp = np.array(goalkRate)
goalkRate = temp.astype(np.float)
miB = -23
maB = 23

def likelihood(Rate,Scored,minB,maxB,log=True):
    def p(beta, data):
        return 1/(1+np.exp(-beta*data))

    betas = np.linspace(minB,maxB,1000)
    l = []
    for b in betas:
        l.append(sum(Scored*np.log(p(b,Rate)) + (1-Scored)*np.log(1-p(b,Rate))))

    li = np.array(l)

    if(log):
        return li
    else:
        return np.exp(li)


plt.figure()
plt.plot(np.linspace(miB,maB,1000), likelihood(goalkRate,scored,miB,maB,False))
plt.show()