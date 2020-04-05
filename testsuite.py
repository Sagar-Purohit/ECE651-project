
import unittest
import torch
import torch_testing as tt


class NaNTensorException(Exception):
  pass

class InfTensorException(Exception):
  pass


class TestSame(unittest.TestCase):
    def runTest(self,initial_param,after_param):
        count=0
        for i,a_val in enumerate(after_param):
            if(tt.assert_equal(a_val, initial_param[i])== None):
                count=count+1
        if(count==4):
            print('Unit Test 1 : Test for Same tensors passed')

class TestNan(unittest.TestCase) :
    def runTest(self,after_param):
        count=0
        for val in after_param:
         try:
            assert not torch.isnan(val).byte().any()
            count=count+1
         except AssertionError:
            raise NaNTensorException("There was a NaN value in tensor")
        if(count==4):
            print('Unit Test 2 : No NaN value in tensor. Test for NaN passed')

class TestInfinite(unittest.TestCase) :
    def runTest(self,after_param):
         count=0
         for val in after_param:
            try:
                assert torch.isfinite(val).byte().any()
                count=count+1
            except AssertionError:
                raise InfTensorException("There was an Inf value in tensor")
         if(count==4):
            print('Unit Test 3 : No infinite value in tensor. Test for infinite value passed')

suite = unittest.TestSuite()
suite.addTest(TestSame())
suite.addTests([TestNan(), TestInfinite()])

unittest.TextTestRunner().run(suite)


