import numpy as np
import graphviz
from typing import Union, Tuple, Set, Dict, List, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os

# Global configuration for distributed computing
_DISTRIBUTED_CONFIG = {
    'enabled': False,
    'num_workers': mp.cpu_count(),
    'min_elements_for_parallel': 10000,  # Minimum elements to trigger parallel execution
    'use_processes': True,  # Use processes (True) or threads (False)
}


def set_distributed_config(
    enabled: bool = None,
    num_workers: int = None,
    min_elements_for_parallel: int = None,
    use_processes: bool = None
):
    """
    Configure distributed computing settings globally.

    Args:
        enabled: Enable/disable distributed computing
        num_workers: Number of parallel workers (defaults to CPU count)
        min_elements_for_parallel: Minimum tensor elements to trigger parallel execution
        use_processes: Use processes (True, bypasses GIL) or threads (False, lower overhead)
    """
    if enabled is not None:
        _DISTRIBUTED_CONFIG['enabled'] = enabled
    if num_workers is not None:
        _DISTRIBUTED_CONFIG['num_workers'] = num_workers
    if min_elements_for_parallel is not None:
        _DISTRIBUTED_CONFIG['min_elements_for_parallel'] = min_elements_for_parallel
    if use_processes is not None:
        _DISTRIBUTED_CONFIG['use_processes'] = use_processes


def get_distributed_config() -> dict:
    """Get the current distributed computing configuration."""
    return _DISTRIBUTED_CONFIG.copy()


# Worker functions must be defined at module level for pickling
def _apply_op_to_shard(args):
    """Apply an operation to a data shard (used by parallel executor)."""
    shard, op_func, other_shard = args
    if other_shard is not None:
        return op_func(shard, other_shard)
    return op_func(shard)


def _apply_unary_op_to_shard(args):
    """Apply a unary operation to a data shard."""
    shard, op_func = args
    return op_func(shard)


# Reduction worker functions (must be at module level for pickling)
def _sum_shard(args):
    """Sum a shard along specified axis."""
    shard, axis, keepdims = args
    return np.sum(shard, axis=axis, keepdims=keepdims)


def _mean_shard(args):
    """Compute mean of a shard along specified axis."""
    shard, axis, keepdims = args
    return np.mean(shard, axis=axis, keepdims=keepdims)


def _max_shard(args):
    """Compute max of a shard along specified axis."""
    shard, axis, keepdims = args
    return np.max(shard, axis=axis, keepdims=keepdims)


def _min_shard(args):
    """Compute min of a shard along specified axis."""
    shard, axis, keepdims = args
    return np.min(shard, axis=axis, keepdims=keepdims)


def _power_shard(args):
    """Compute power of a shard."""
    shard, power = args
    return np.power(shard, power)


class Tensor:
    """
    A class representing a tensor with distributed computing support.

    This class wraps numpy arrays and provides the foundation for automatic differentiation.
    When distributed mode is enabled, operations on large tensors are automatically
    parallelized across multiple CPU cores.

    Attributes:
        data (np.ndarray): The underlying numpy array
        shape (tuple): Shape of the tensor
        dtype (np.dtype): Data type of the tensor
        is_distributed (bool): Whether this tensor uses distributed operations
        shards (list): Data shards when in distributed mode
    """

    def __init__(
        self,
        data: Union[np.ndarray, float, int, List],
        dtype=np.float32,
        distributed: Optional[bool] = None
    ):
        """
        Initialize a Tensor.

        Args:
            data: The data to be stored in the tensor.
            dtype: The data type of the tensor. Defaults to np.float32.
            distributed: Override global distributed setting for this tensor.
                        If None, uses global config and size threshold.
        """
        self.data = np.array(data, dtype=dtype)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        
        # Determine if this tensor should use distributed computing
        num_elements = np.prod(self.shape)
        if distributed is not None:
            self.is_distributed = distributed
        else:
            self.is_distributed = (
                _DISTRIBUTED_CONFIG['enabled'] and
                num_elements >= _DISTRIBUTED_CONFIG['min_elements_for_parallel']
            )
        
        # Create shards if distributed
        self._shards: Optional[List[np.ndarray]] = None
        if self.is_distributed:
            self._create_shards()

    def _create_shards(self):
        """Split the tensor data into shards for parallel processing."""
        num_workers = _DISTRIBUTED_CONFIG['num_workers']
        
        if len(self.shape) == 0 or self.shape[0] < num_workers:
            # Too small to shard meaningfully
            self._shards = [self.data]
            return
        
        # Shard along the first axis
        shard_size = (self.shape[0] + num_workers - 1) // num_workers
        self._shards = []
        for i in range(num_workers):
            start = i * shard_size
            end = min((i + 1) * shard_size, self.shape[0])
            if start < self.shape[0]:
                self._shards.append(self.data[start:end])

    def _gather_shards(self):
        """Reconstruct the full tensor from shards."""
        if self._shards is None or len(self._shards) == 0:
            return
        self.data = np.concatenate(self._shards, axis=0)
        self.shape = self.data.shape

    @classmethod
    def from_array(cls, arr: Union[np.ndarray, float, int], distributed: Optional[bool] = None):
        """
        Create a Tensor from an array-like object.

        Args:
            arr: The array-like object to create the tensor from.
            distributed: Override distributed setting for this tensor.

        Returns:
            Tensor: A new Tensor instance.
        """
        return cls(arr, distributed=distributed)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...], dtype=np.float32, distributed: Optional[bool] = None):
        """Create a tensor filled with zeros."""
        return cls(np.zeros(shape, dtype=dtype), distributed=distributed)

    @classmethod
    def ones(cls, shape: Tuple[int, ...], dtype=np.float32, distributed: Optional[bool] = None):
        """Create a tensor filled with ones."""
        return cls(np.ones(shape, dtype=dtype), distributed=distributed)

    @classmethod
    def randn(cls, *shape, dtype=np.float32, distributed: Optional[bool] = None):
        """Create a tensor with random normal values."""
        return cls(np.random.randn(*shape).astype(dtype), distributed=distributed)

    def _parallel_binary_op(self, other: 'Tensor', op_func: Callable) -> np.ndarray:
        """Execute a binary operation in parallel across shards."""
        if self._shards is None:
            self._create_shards()
        
        # Handle broadcasting for scalars or smaller tensors
        if other.shape == () or other.shape == (1,):
            other_shards = [other.data] * len(self._shards)
        elif other.is_distributed and other._shards is not None:
            other_shards = other._shards
        else:
            # Create matching shards for other tensor
            other_shards = []
            idx = 0
            for shard in self._shards:
                shard_len = shard.shape[0]
                other_shards.append(other.data[idx:idx + shard_len])
                idx += shard_len
        
        # Prepare work items
        work_items = [
            (self._shards[i], op_func, other_shards[i] if i < len(other_shards) else other_shards[-1])
            for i in range(len(self._shards))
        ]
        
        # Execute in parallel
        ExecutorClass = ProcessPoolExecutor if _DISTRIBUTED_CONFIG['use_processes'] else ThreadPoolExecutor
        with ExecutorClass(max_workers=_DISTRIBUTED_CONFIG['num_workers']) as executor:
            result_shards = list(executor.map(_apply_op_to_shard, work_items))
        
        return np.concatenate(result_shards, axis=0)

    def _parallel_unary_op(self, op_func: Callable) -> np.ndarray:
        """Execute a unary operation in parallel across shards."""
        if self._shards is None:
            self._create_shards()
        
        work_items = [(shard, op_func) for shard in self._shards]
        
        ExecutorClass = ProcessPoolExecutor if _DISTRIBUTED_CONFIG['use_processes'] else ThreadPoolExecutor
        with ExecutorClass(max_workers=_DISTRIBUTED_CONFIG['num_workers']) as executor:
            result_shards = list(executor.map(_apply_unary_op_to_shard, work_items))
        
        return np.concatenate(result_shards, axis=0)

    def apply(self, op_func: Callable, other: Optional['Tensor'] = None) -> 'Tensor':
        """
        Apply an operation, automatically using parallel execution if beneficial.

        Args:
            op_func: The operation function to apply
            other: Optional other tensor for binary operations

        Returns:
            Tensor: Result of the operation
        """
        if self.is_distributed and self._shards is not None:
            if other is not None:
                result_data = self._parallel_binary_op(other, op_func)
            else:
                result_data = self._parallel_unary_op(op_func)
        else:
            if other is not None:
                result_data = op_func(self.data, other.data)
            else:
                result_data = op_func(self.data)
        
        return Tensor(result_data, distributed=self.is_distributed)

    def __add__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise addition."""
        if not isinstance(other, Tensor):
            other = Tensor(other, distributed=False)
        return self.apply(np.add, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise subtraction."""
        if not isinstance(other, Tensor):
            other = Tensor(other, distributed=False)
        return self.apply(np.subtract, other)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, distributed=False)
        return other.__sub__(self)

    def __mul__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other, distributed=False)
        return self.apply(np.multiply, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise division."""
        if not isinstance(other, Tensor):
            other = Tensor(other, distributed=False)
        return self.apply(np.divide, other)

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, distributed=False)
        return other.__truediv__(self)

    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        """Element-wise power."""
        if self.is_distributed and self._shards is not None:
            work_items = [(shard, power) for shard in self._shards]
            ExecutorClass = ProcessPoolExecutor if _DISTRIBUTED_CONFIG['use_processes'] else ThreadPoolExecutor
            with ExecutorClass(max_workers=_DISTRIBUTED_CONFIG['num_workers']) as executor:
                result_shards = list(executor.map(_power_shard, work_items))
            return Tensor(np.concatenate(result_shards, axis=0), distributed=self.is_distributed)
        return Tensor(np.power(self.data, power), distributed=self.is_distributed)

    def __neg__(self) -> 'Tensor':
        """Negation."""
        return self.apply(np.negative)

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Sum of tensor elements."""
        if self.is_distributed and self._shards is not None and axis in (None, 0):
            # Sum each shard, then combine
            work_items = [(shard, axis, keepdims) for shard in self._shards]
            ExecutorClass = ProcessPoolExecutor if _DISTRIBUTED_CONFIG['use_processes'] else ThreadPoolExecutor
            with ExecutorClass(max_workers=_DISTRIBUTED_CONFIG['num_workers']) as executor:
                partial_sums = list(executor.map(_sum_shard, work_items))
            result = sum(partial_sums)
            return Tensor(result, distributed=False)
        return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), distributed=False)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Mean of tensor elements."""
        if self.is_distributed and self._shards is not None and axis in (None, 0):
            total_sum = self.sum(axis=axis, keepdims=keepdims)
            count = np.prod(self.shape) if axis is None else self.shape[0]
            return Tensor(total_sum.data / count, distributed=False)
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), distributed=False)

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Maximum of tensor elements."""
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims), distributed=False)

    def min(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Minimum of tensor elements."""
        return Tensor(np.min(self.data, axis=axis, keepdims=keepdims), distributed=False)

    def reshape(self, *shape) -> 'Tensor':
        """Reshape the tensor."""
        # Need to gather shards before reshaping
        if self.is_distributed:
            self._gather_shards()
        return Tensor(self.data.reshape(*shape), distributed=self.is_distributed)

    def transpose(self, *axes) -> 'Tensor':
        """Transpose the tensor."""
        if self.is_distributed:
            self._gather_shards()
        if axes:
            return Tensor(np.transpose(self.data, axes), distributed=False)
        return Tensor(np.transpose(self.data), distributed=False)

    @property
    def T(self) -> 'Tensor':
        """Transpose property."""
        return self.transpose()

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        if self.is_distributed:
            self._gather_shards()
        return self.data.copy()

    def enable_distributed(self):
        """Enable distributed computing for this tensor."""
        if not self.is_distributed:
            self.is_distributed = True
            self._create_shards()

    def disable_distributed(self):
        """Disable distributed computing for this tensor."""
        if self.is_distributed:
            self._gather_shards()
            self.is_distributed = False
            self._shards = None

    def __repr__(self):
        dist_str = ", distributed=True" if self.is_distributed else ""
        if np.prod(self.shape) <= 10:
            return f"Tensor(data={self.data}{dist_str})"
        return f"Tensor(shape={self.shape}, dtype={self.dtype}{dist_str})"

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def __getitem__(self, idx):
        if self.is_distributed:
            self._gather_shards()
        return Tensor(self.data[idx], distributed=False)

class Value:
    """
    A class representing a value in the computational graph.

    This class is used to build and manipulate the computational graph for automatic differentiation.
    """

    _ops: Dict[str, Set['Value']] = {}

    def __init__(self, data, _children=(), _op='', label=None):
        """
        Initialize a Value object.

        Args:
            data: The data to be stored in the Value object.
            _children (tuple, optional): Child nodes in the computational graph. Defaults to ().
            _op (str, optional): The operation that produced this Value. Defaults to ''.
            label (str, optional): A label for the Value. Defaults to None.
        """
        self.data = Tensor(data) if not isinstance(data, Tensor) else data
        self.grad = Tensor(np.zeros_like(self.data.data))
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.is_constant = not _children
        self.label = label or self._generate_label()

    def _generate_label(self):
        """Generate a label for the Value based on its data."""
        if np.prod(self.data.shape) == 1:
            return f"Value({self.data.data.item():.4f})"
        else:
            return f"Value(shape={self.data.shape})"

    def __add__(self, other):
        """Add two Values."""
        return self._binary_op(other, '+', np.add)

    def __radd__(self, other):
        """Reverse add two Values."""
        return self + other

    def __sub__(self, other):
        """Subtract two Values."""
        return self + (-other)

    def __rsub__(self, other):
        """Reverse subtract two Values."""
        return other + (-self)

    def __mul__(self, other):
        """Multiply two Values."""
        return self._binary_op(other, '*', np.multiply)

    def __rmul__(self, other):
        """Reverse multiply two Values."""
        return self * other

    def __neg__(self):
        """Negate a Value."""
        return self * -1

    def __truediv__(self, other):
        """Divide two Values."""
        return self * other**-1

    def __rtruediv__(self, other):
        """Reverse divide two Values."""
        return other * self**-1

    def __pow__(self, other):
        """Raise a Value to a power."""
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        return self._binary_op(other, f'**{other}', lambda x, y: x ** y)

    def relu(self):
        """Apply the ReLU function to this Value."""
        return self._unary_op('ReLU', lambda x: np.maximum(0, x))

    def tanh(self):
        """Apply the tanh function to this Value."""
        return self._unary_op('tanh', np.tanh)

    def exp(self):
        """Apply the exponential function to this Value."""
        return self._unary_op('exp', np.exp)

    def log(self):
        """Apply the natural logarithm function to this Value."""
        return self._unary_op('log', np.log)

    def _binary_op(self, other, op_symbol, op_func):
        """
        Perform a binary operation.

        Args:
            other: The other Value to perform the operation with.
            op_symbol (str): A symbol representing the operation.
            op_func (function): The function to perform the operation.

        Returns:
            Value: The result of the operation.
        """
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(op_func(self.data.data, other.data.data), (self, other), op_symbol)
        
        def _backward():
            if op_symbol == '+':
                self.grad.data += out.grad.data
                other.grad.data += out.grad.data
            elif op_symbol == '*':
                self.grad.data += other.data.data * out.grad.data
                other.grad.data += self.data.data * out.grad.data
            elif op_symbol.startswith('**'):
                power = float(op_symbol[2:])
                self.grad.data += (power * self.data.data**(power-1)) * out.grad.data
        out._backward = _backward
        return out

    def _unary_op(self, op_symbol, op_func):
        """
        Perform a unary operation.

        Args:
            op_symbol (str): A symbol representing the operation.
            op_func (function): The function to perform the operation.

        Returns:
            Value: The result of the operation.
        """
        out = Value(op_func(self.data.data), (self,), op_symbol)
        
        def _backward():
            if op_symbol == 'ReLU':
                self.grad.data += (out.data.data > 0) * out.grad.data
            elif op_symbol == 'tanh':
                self.grad.data += (1 - out.data.data**2) * out.grad.data
            elif op_symbol == 'exp':
                self.grad.data += out.data.data * out.grad.data
            elif op_symbol == 'log':
                self.grad.data += (1 / self.data.data) * out.grad.data
        out._backward = _backward
        return out

    def backward(self):
        """Perform backpropagation starting from this Value."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad.data = np.ones_like(self.data.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return self.label

def optimize_graph(root: Value):
    """
    Optimize the computational graph.

    This function performs various optimizations on the graph, such as fusing ReLU and Add operations.

    Args:
        root (Value): The root node of the computational graph.

    Returns:
        Value: The root node of the optimized graph.
    """
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(root)

    for v in topo:
        if v._op == 'ReLU' and len(v._prev) == 1:
            prev = next(iter(v._prev))
            if prev._op == '+':
                fused = Value(np.maximum(0, prev.data.data), prev._prev, 'FusedReLUAdd')
                def _backward():
                    grad = (fused.data.data > 0) * fused.grad.data
                    for child in prev._prev:
                        child.grad.data += grad
                fused._backward = _backward
                v._prev = fused._prev
                v.data = fused.data
                v._op = 'FusedReLUAdd'

    return root

def visualize_graph(root: Value, filename='computational_graph'):
    """
    Visualize the computational graph.

    This function creates a visual representation of the computational graph and saves it as a PNG file.

    Args:
        root (Value): The root node of the computational graph.
        filename (str, optional): The name of the file to save the visualization. Defaults to 'computational_graph'.
    """
    # Create a 'graph' folder if it doesn't exist
    graph_root = 'graph'
    os.makedirs(graph_root, exist_ok=True)
    
    # Prepare the full file path
    file_path = os.path.join(graph_root, filename)
    
    dot = graphviz.Digraph(comment='Computational Graph')
    dot.attr(rankdir='LR')
    
    visited = set()
    
    def add_nodes(v: Value):
        if v not in visited:
            visited.add(v)
            label = v.label
            if np.prod(v.data.shape) == 1:
                label += f"\nvalue={v.data.data.item():.4f}"
                if np.prod(v.grad.shape) == 1:
                    label += f"\ngrad={v.grad.data.item():.4f}"
            dot.node(str(id(v)), label, shape='box')
            if v._op:
                dot.node(str(id(v)) + v._op, v._op, shape='ellipse')
                dot.edge(str(id(v)) + v._op, str(id(v)))
            for child in v._prev:
                add_nodes(child)
                dot.edge(str(id(child)), str(id(v)) + v._op)
    
    add_nodes(root)
    dot.render(file_path, view=True, format='png')
    print(f"Graph visualization saved as {file_path}.png")