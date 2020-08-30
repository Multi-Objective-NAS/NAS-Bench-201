import copy

from nasbench.lib import graph_util
from collections import defaultdict
import unittest
import numpy as np

# Graphviz is optional and only required for visualization.
try:
  import graphviz   # pylint: disable=g-import-not-at-top
except ImportError:
  pass

class ModelSpec(object):
  """Model specification given adjacency matrix and labeling."""
  
  def __init__(self, index=None, model_str=None, matrix=None, ops=None, data_format='channels_last'):
    """Initialize the module spec.

    Args:
      matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.

    Raises:
      ValueError: invalid matrix or ops
    """
    self.index = index
    self.model_str = model_str
    self.valid_spec = True
    
    if not isinstance(matrix, np.ndarray):
      if not isinstance(matrix, list):
        matrix, ops = self._convert_to_matrix_ops(model_str) # when matrix and ops is None
      else:
        matrix = np.array(matrix)

    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('matrix must be square')
    if shape[0] != len(ops):
      raise ValueError('length of ops must match matrix dimensions')
    if not is_upper_triangular(matrix):
      raise ValueError('matrix must be upper triangular')
        
    self.original_matrix = copy.deepcopy(matrix)
    self.original_ops = copy.deepcopy(ops)
    self.matrix = copy.deepcopy(matrix)
    self.ops = copy.deepcopy(ops)

    self._prune()      
    
    self.num_nodes = np.shape(self.matrix)[0]
    self.num_edges = np.sum(self.matrix)
    
    self.data_format = data_format
  
  def _convert_to_matrix_ops(self, arch_str):
    # only for nasbench 201
    search_space = [ 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    node_strs = arch_str.split('+')
    num_nodes = len(node_strs) + 1
    org_end_node = []
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.int)

    converted_nodes = 1
    to_node_dict = defaultdict(list)
    to_node_dict[0] = [0]
    converted_ops = ['input']

    for i, node_str in enumerate(node_strs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      for xi in inputs:
        op, idx = xi.split('~')
        if op not in search_space: raise ValueError('this op ({:}) is not in {:}'.format(op, search_space))
        op_idx, node_idx = search_space.index(op), int(idx)
        matrix[i+1, node_idx] = op_idx
        if op_idx != 0:
          to_node_dict[i+1].append(converted_nodes)
          converted_ops.append(op)
          converted_nodes += 1
      if len(to_node_dict[i+1]) == 0 :
        to_node_dict[i+1].append(0)      

    for i in range(num_nodes):
      if np.count_nonzero(matrix[:,i]) == 0:
        org_end_node.append(i) 

    converted_ops.append('output')
    converted_nodes += 1    
    converted_end_node = converted_nodes - 1
    converted_matrix = np.zeros((converted_nodes, converted_nodes), dtype=np.int)
    for i in range(1, matrix.shape[0]):
      turn = 0
      for j in range(i):
        if matrix[i][j] != 0:
          to_node = to_node_dict[i][turn]
          for k in to_node_dict[j]:
            converted_matrix[k][to_node] = 1
            converted_matrix[to_node][k] = 1
          turn += 1
    for end_node in org_end_node:     
      for k in to_node_dict[end_node]:
        converted_matrix[k][converted_end_node] = 1
        converted_matrix[converted_end_node][k] = 1
    
    converted_matrix = np.triu(converted_matrix, 0)
    
    return converted_matrix, converted_ops

  def _prune(self):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(self.original_matrix)[0]

    # DFS forward from input
    visited_from_input = set([0])
    frontier = [0]
    while frontier:
      top = frontier.pop()
      for v in range(top + 1, num_vertices):
        if self.original_matrix[top, v] and v not in visited_from_input:
          visited_from_input.add(v)
          frontier.append(v)

    # DFS backward from output
    visited_from_output = set([num_vertices - 1])
    frontier = [num_vertices - 1]
    while frontier:
      top = frontier.pop()
      for v in range(0, top):
        if self.original_matrix[v, top] and v not in visited_from_output:
          visited_from_output.add(v)
          frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
      self.matrix = None
      self.ops = None
      self.valid_spec = False
      return

    self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
    self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
      del self.ops[index]

  def hash_spec(self, canonical_ops):
    """Computes the isomorphism-invariant graph hash of this spec.

    Args:
      canonical_ops: list of operations in the canonical ordering which they
        were assigned (i.e. the order provided in the config['available_ops']).

    Returns:
      MD5 hash of this spec which can be used to query the dataset.
    """
    # Invert the operations back to integer label indices used in graph gen.
    labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
    return graph_util.hash_module(self.matrix, labeling)

  def visualize(self):
    """Creates a dot graph. Can be visualized in colab directly."""
    num_vertices = np.shape(self.matrix)[0]
    g = graphviz.Digraph()
    g.node(str(0), 'input')
    for v in range(1, num_vertices - 1):
      g.node(str(v), self.ops[v])
    g.node(str(num_vertices - 1), 'output')

    for src in range(num_vertices - 1):
      for dst in range(src + 1, num_vertices):
        if self.matrix[src, dst]:
          g.edge(str(src), str(dst))

    return g


def is_upper_triangular(matrix):
  """True if matrix is 0 on diagonal and below."""
  for src in range(np.shape(matrix)[0]):
    for dst in range(0, src + 1):
      if matrix[src, dst] != 0:
        return False

  return True


class ModelSpecTest(unittest.TestCase):
  def setUp(self):
    self.input = ['|nor_conv_3x3~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_1x1~2|',\
                  '|none~0|+|avg_pool_3x3~0|none~1|+|avg_pool_3x3~0|skip_connect~1|nor_conv_1x1~2|',\
                  '|nor_conv_1x1~0|+|skip_connect~0|skip_connect~1|+|nor_conv_3x3~0|skip_connect~1|none~2|']

    self.result_matrix = [np.array([[0, 1, 1, 1, 0, 0, 0],\
                                    [0, 0, 0, 0, 1, 0, 0],\
                                    [0, 0, 0, 0, 0, 1, 0],\
                                    [0, 0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 0, 0]]),\
                          np.array([[0, 1, 1, 1, 0, 0],\
                                    [0, 0, 0, 0, 1, 0],\
                                    [0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 0]]),\
                          np.array([[0, 1, 1, 0, 1, 0, 0],\
                                    [0, 0, 0, 1, 0, 1, 0],\
                                    [0, 0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 0, 1],\
                                    [0, 0, 0, 0, 0, 0, 0]])]

    self.result_ops = [ ['input', 'nor_conv_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'nor_conv_3x3', 'nor_conv_1x1', 'output'],\
                        ['input', 'avg_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1', 'output'],\
                        ['input', 'nor_conv_1x1', 'skip_connect', 'skip_connect', 'nor_conv_3x3', 'skip_connect', 'output'] ]

  def test_convert_matrix(self):
    for idx, input in enumerate(self.input):
      m = ModelSpec(index=0, model_str=input)
      np.testing.assert_array_equal(m.matrix, self.result_matrix[idx])
      self.assertListEqual(m.ops, self.result_ops[idx])


if __name__ == '__main__':
  unittest.main()
