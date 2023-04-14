import unittest
import numpy as np
import pytest
import likelihood

v=np.array([100e9,200e9,50e9])

'''def test_prior():
    param = [0, 50]
    I = FILETOTEST.prior(param[0],param[1])

    areaUnderGraph = sum(I)*(I[1]-I[0])
    assert areaUnderGraph == pytest.approx(1)
    assert I[0] == pytest.approx(param[0])
    assert I[-1] == pytest.approx(param[1] - I[1] + I[0])
'''
def test_likelihood():
    goalkeeperRate = np.array([0.161290322581, 0.231707317073,0.222222222222,0.153846153846,0.142857142857,0.308641975309,0.269230769231,0.231707317073,0.181818181818,0.30612244898,0.142857142857,0.112676056338,0.112676056338,0.188405797101,0.3,0.0,0.0454545454545,0.230769230769,0.344,0.172413793103,0.204081632653,0.333333333333,0.245098039216,0.333333333333,0.270833333333,0.4])
    goalScored = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
    minB = -500
    maxB= 500
    
    I = likelihood.likelihood(goalkeeperRate[13:18],goalScored[13:18],minB,maxB)
    
    def p(beta, data):
        return 1/(1+np.exp(beta*data))

    betas = np.linspace(minB,maxB,1000)
    IShouldBe = []
    for b in betas:
        IShouldBe.append(sum(goalScored[13:18]*np.log(p(b,goalkeeperRate[13:18])) + (1-goalScored[13:18])*np.log(1-p(b,goalkeeperRate[13:18]))))

    assert np.nansum(I)/len(I) == pytest.approx(np.nansum(IShouldBe)/len(IShouldBe))

################################################################################

'''def test_picky():
    # Make sure that tools.picky raises an error if the
    # wrong type is inputted
    pytest.raises(TypeError, tools.picky, 'hey')
'''
test_likelihood()