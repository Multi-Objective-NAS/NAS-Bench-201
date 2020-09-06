import numpy as np
from nas_201_api import model_spec as _model_spec
ModelSpec = _model_spec.ModelSpec

class ModelSpecTest(unittest.TestCase):
  def setUp(self):
    self.input = ['|nor_conv_3x3~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_1x1~2|',
                  '|none~0|+|avg_pool_3x3~0|none~1|+|avg_pool_3x3~0|skip_connect~1|nor_conv_1x1~2|',
                  '|nor_conv_1x1~0|+|skip_connect~0|skip_connect~1|+|nor_conv_3x3~0|skip_connect~1|none~2|']

    self.result_matrix = [np.array([[0, 1, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0]]),
                          np.array([[0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0]]),
                          np.array([[0, 1, 1, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0]])]

    self.result_ops = [ ['input', 'nor_conv_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'nor_conv_3x3', 'nor_conv_1x1', 'output'],
                        ['input', 'avg_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1', 'output'],
                        ['input', 'nor_conv_1x1', 'skip_connect', 'skip_connect', 'nor_conv_3x3', 'skip_connect', 'output'] ]

  def test_convert_matrix(self):
    for idx, input in enumerate(self.input):
      m = ModelSpec(index=0, model_str=input)
      np.testing.assert_array_equal(m.matrix, self.result_matrix[idx])
      self.assertListEqual(m.ops, self.result_ops[idx])


if __name__ == '__main__':
  unittest.main()
