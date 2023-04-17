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

def test_logPost():
    value = functions.log_post(5.,np.array([0.5]),np.array([1]))
    assert value == pytest.approx(-0.0788897)
    value = functions.log_post(5.,np.array([0.5]),np.array([0]))
    assert value == pytest.approx(-2.5788897)

def test_logPostNew():
    value = functions.log_postNew(np.array([5.,-3.]),np.array([0.5]),np.array([1]))
    assert value == pytest.approx(-0.0297504)
    value = functions.log_postNew(np.array([5.,-3.]),np.array([[0.5,0.75]]),np.array([1,0]))
    assert value == pytest.approx(-2.841718007)
################################################################################