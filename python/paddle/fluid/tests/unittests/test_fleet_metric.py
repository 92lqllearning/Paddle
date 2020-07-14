#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test fleet metric."""

from __future__ import print_function
import numpy as np
import paddle
import paddle.fluid as fluid
import os
import unittest
import paddle.fleet.metrics.metric as metric
from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet as fleet


class TestFleetMetric(unittest.TestCase):
    """
    Test cases for fleet metric.
    """

    def setUp(self):
        """Set up, set envs."""

        class FakeFleet:
            """fake fleet only for test"""

            def __init__(self):
                """Init."""
                self.gloo = fluid.core.Gloo()
                self.gloo.set_rank(0)
                self.gloo.set_size(1)
                self.gloo.set_prefix("123")
                self.gloo.set_iface("lo")
                self.gloo.set_hdfs_store("./tmp_test_metric", "", "")
                self.gloo.init()
            
            def _all_reduce(self, input, output, mode="sum"):
                """all reduce using gloo"""
                input_list = [i for i in input]
                ans = self.gloo.all_reduce(input_list, mode)
                for i in range(len(ans)):
                    output[i] = 1

            def _barrier_worker(self):
                """fake barrier worker, do nothing"""
                pass

        self.fleet = FakeFleet()
        fleet._role_maker = self.fleet

    def test_metric_1(self):
        """test cases for metrics"""
        arr = np.array([1,2,3,4])
        metric.sum(arr)
        metric.max(arr)
        metric.min(arr)
        arr1 = np.array([[1,2,3,4]])
        arr2 = np.array([[1,2,3,4]])
        arr3 = np.array([1,2,3,4])
        metric.auc(arr1, arr2)
        metric.mae(arr, 3)
        metric.rmse(arr, 3)
        metric.mse(arr, 3)
        metric.acc(arr, arr3)


if __name__ == "__main__":
    unittest.main()
