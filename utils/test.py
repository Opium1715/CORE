import unittest
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn.functional as F


class MyTestCase(unittest.TestCase):
    def test_something(self):
        a = torch.randn(5, 5)
        print(a)
        a = torch.tril(a)
        print(a)
        condition = a == 0
        a = torch.where(condition, -torch.inf, 0)
        print(a)
        print(torch.softmax(a, dim=-1))
        F.multi_head_attention_forward()


if __name__ == '__main__':
    unittest.main()
