import numpy as np

####For the One variable MCMCs:

#The probabiltiy function for p(scoring) given the data and a value of beta
def p(beta, data):
    return 1/(1+np.exp(-beta*data))

#Linear prior
def log_prior(beta):
    if -1.0 < beta < 14:
        return 0.0 # the constant doesn't matter since MCMCs only care about *ratios* of probabilities
    return -np.inf # log(0) = -inf

#Define the likelihood function from wikipedia
def log_likelihood(beta,Rate,Scored,log=True):
    l = (sum(Scored*np.log(p(beta,Rate)) + (1-Scored)*np.log(1-p(beta,Rate))))

    if(log):
        return l
    else:
        return np.exp(l)

#posterior adding the log prior and likelihoods togethers
def log_post(beta,Rate,Scored):
    return log_prior(beta) + log_likelihood(beta,Rate,Scored)


####For the Many Variable MCMCs:


#Our new probability function with many betas.
def pNew(beta, data):
    '''
    data: an array of data sets. The number of data sets defines the number of betas needed
    '''
    if len(beta)-1 != len(data):
        print(beta)
        raise ValueError("There is an incorrect balance of beta values and datasets.\nNote 'data' should be an array of datasets so that len(beta)-1 == len(data)")
    
    exponent = -beta[0]
    for i in range(len(data)):
        exponent -= beta[i+1]*data[i]
    return 1/(1+np.exp(exponent))

def log_priorNew(beta):
    for b in beta:
        if -500 < b < 500:
            continue
        else:
            return -np.inf # log(0) = -inf
    return 0.0 # the constant doesn't matter since MCMCs only care about *ratios* of probabilities

#Define the likelihood function from wikipedia
def log_likelihoodNew(betas,data,Scored):
    l = (sum(Scored*np.log(pNew(betas,data)) + (1-Scored)*np.log(1-pNew(betas,data))))
    
    if(np.isnan(l)):
        #If the answer is -infinity, convert nan to this value
        l = -np.inf

    return l

#The log posterior
def log_postNew(betas,Data,Scored):
    return log_priorNew(betas) + log_likelihoodNew(betas,Data,Scored)