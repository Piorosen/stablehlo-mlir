# https://flax.readthedocs.io/en/latest/mnist_tutorial.html
#%%
import jax
import numpy as np
import jax.numpy as jnp  # JAX NumPy
from flax import nnx  # The Flax NNX API.
from functools import partial
from jax import export
from transformers import FlaxResNetModel

####################################

# Document from https://openxla.org/stablehlo/tutorials/jax-export
# Note: This helper uses a JAX internal API that may break at any time, 
# but it serves no functional purpose in the tutorial aside from readability.

from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Returns prettyprint of StableHLO module without large constants
def get_stablehlo_asm(module_str):
  with jax_mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
    return stablehlo_module.operation.get_asm(large_elements_limit=20)

####################################

def convert_jax_to_stablehlo(model, inputs: np.array):
    input_shape = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]
    jit_model = jax.jit(model)
    stablehlo = export.export(jit_model)(*input_shape).mlir_module()
    return get_stablehlo_asm(stablehlo)

def convert_jax_to_stablehlo_with_weight(model, inputs: np.array):
    input_shape = [jnp.ones(input.shape, input.dtype) for input in inputs]
    lowered = jax.jit(model).lower(*input_shape)
    return lowered.as_text('stablehlo')
    
# Equal code.
# with open('mnist_ir.mlir', 'wt+') as f:
#     f.write(str(jit_mnist.compiler_ir('stablehlo')))
# with open('mnist_as.mlir', 'wt+') as f:
#     f.write(jit_mnist.as_text('stablehlo'))
    
####################################

class CNN(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    x = nnx.softmax(x)
    return x

# Instantiate the model.
resnet18 = FlaxResNetModel.from_pretrained("microsoft/resnet-18", return_dict=False)
resnet_inputs = [np.random.randn(1, 3, 224, 224)]
mnist = CNN(rngs=nnx.Rngs(0))
mnist_inputs = [np.random.randn(1, 28, 28, 1)]

seq = [[resnet18, resnet_inputs, 'resnet18'],
       [mnist, mnist_inputs, 'mnist']]

for item in seq:
    print(item[2])
    if False: # Too Massive.
        with open(f'{item[2]}_weight.mlir', 'wt+') as f:
            f.write(convert_jax_to_stablehlo(item[0], item[1]))
    with open(f'{item[2]}.mlir', 'wt+') as f:
        f.write(convert_jax_to_stablehlo(item[0], item[1]))

# %%
