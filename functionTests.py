import unittest
import numpy as np
import pytest
import functions

def test_p():
    prob = functions.p(6,3)
    assert prob == pytest.approx(1/(1+np.exp(-18)))
    probArr = functions.p(6,np.array([3,4]))
    assert probArr[0] == pytest.approx(1/(1+np.exp(-18)))
    assert probArr[1] == pytest.approx(1/(1+np.exp(-24)))
    prob2Arr = functions.p(np.array([6,5]),np.array([3,4]))
    assert prob2Arr[0] == pytest.approx(1/(1+np.exp(-18)))
    assert prob2Arr[1] == pytest.approx(1/(1+np.exp(-20)))

def test_pNew():
    prob = functions.pNew(np.array([6,2]),np.array([3]))
    assert prob == pytest.approx(1/(1+np.exp(-12)))
    probArr = functions.pNew(np.array([6,2]),np.array([[3,4]]))
    assert probArr[0] == pytest.approx(1/(1+np.exp(-12)))
    assert probArr[1] == pytest.approx(1/(1+np.exp(-14)))
    prob2Arr = functions.pNew(np.array([[6,5],[3,4]]),np.array([[3,4]]))
    assert prob2Arr[0] == pytest.approx(1/(1+np.exp(-15)))
    assert prob2Arr[1] == pytest.approx(1/(1+np.exp(-21)))
################################################################################

'''def test_picky():
    # Make sure that tools.picky raises an error if the
    # wrong type is inputted
    pytest.raises(TypeError, tools.picky, 'hey')
'''
test_p()
test_pNew()