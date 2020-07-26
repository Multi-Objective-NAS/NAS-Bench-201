import copy

from nasbench.lib import graph_util
import numpy as np

# Graphviz is optional and only required for visualization.
try:
	import graphviz	 # pylint: disable=g-import-not-at-top
except ImportError:
	pass


class ModelSpec(object):
	"""Model specification given adjacency matrix and labeling."""

	def __init__(self, index, model_str, data_format='channels_last'):
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
		self.matrix, self.ops = self._convert_to_matrix_ops(model_str)
		self.valid_spec = True
        self.search_space = [ 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
		self._prune()

		self.data_format = data_format
        
        if not isinstance(matrix, np.ndarray):
			matrix = np.array(matrix)
		shape = np.shape(matrix)
		if len(shape) != 2 or shape[0] != shape[1]:
			raise ValueError('matrix must be square')
		if shape[0] != len(ops):
			raise ValueError('length of ops must match matrix dimensions')
		if not is_upper_triangular(matrix):
			raise ValueError('matrix must be upper triangular')
    
    def _convert_to_matrix_ops(self, model_str):
        node_strs = arch_str.split('+')
		num_nodes = len(node_strs) + 1
		matrix = np.zeros((num_nodes, num_nodes))
		for i, node_str in enumerate(node_strs):
			inputs = list(filter(lambda x: x != '', node_str.split('|')))
			for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
			for xi in inputs:
				op, idx = xi.split('~')
				if op not in self.search_space: raise ValueError('this op ({:}) is not in {:}'.format(op, self.search_space))
				op_idx, node_idx = self.search_space.index(op), int(idx)
				matrix[i+1, node_idx] = op_idx
            indegree = np.sum(matrix[i+1])
            num_nodes += (indegree-1)
        
        converted_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(1, matrix.shape[0]):
            

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

	def hash_spec(self):
		"""Computes the isomorphism-invariant graph hash of this spec.

		Args:
			canonical_ops: list of operations in the canonical ordering which they
				were assigned (i.e. the order provided in the config['available_ops']).

		Returns:
			MD5 hash of this spec which can be used to query the dataset.
		"""
		# Invert the operations back to integer label indices used in graph gen.
		labeling = [-1] + [self.search_space.index(op) for op in self.ops[1:-1]] + [-2]
		return graph_util.hash_module(self.matrix, labeling)