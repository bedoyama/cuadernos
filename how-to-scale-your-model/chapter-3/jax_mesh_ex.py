import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
# Create our mesh! We're running on a TPU v2-8 4x2 slice with names 'X' and 'Y'.
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=("X", "Y"))


# A little utility function to help define our sharding. A PartitionSpec is our
# sharding (a mapping from axes to names).
def P(*args):
    return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))


# We shard both A and B over the non-contracting dimension and A over the contracting dim.
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P("X", "Y"))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, "Y"))

# We can perform a matmul on these sharded arrays! out_shardings tells us how we want
# the output to be sharded. JAX/XLA handles the rest of the sharding for us.
y = jax.jit(lambda A, B: jnp.einsum("BD,DF->BF", A, B), out_shardings=P("X", "Y"))(A, B)
