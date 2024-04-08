import flax
import jax
import jax.numpy as jnp

from octo.utils.typing import Sequence


@flax.struct.dataclass
class TokenGroup:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)

    Attributes:
        tokens: jax.Array of shape (..., n_tokens, token_dim)
        mask: jax.Array of shape (..., n_tokens) indicating which tokens are valid (1) vs padding (0)
    """

    ### flax.struct.dataclass是一个装饰器，用于定义一个数据类
    ### tokens和mask的顺序是不能互换的，因为flax.struct.dataclass装饰器会参考声明的顺序来自动定义__init__方法
    tokens: jax.typing.ArrayLike
    mask: jax.typing.ArrayLike

    @classmethod
    def create(
        cls, tokens: jax.typing.ArrayLike, mask: jax.typing.ArrayLike = None, **kwargs
    ):
        ### ArrayLike是指各种可以转换为JAX数组的对象，比如列表，元组，numpy数组等
        if mask is None:
            mask = jnp.ones(tokens.shape[:-1])
        assert mask.ndim == tokens.ndim - 1
        return cls(tokens, mask, **kwargs)

    @classmethod
    def concatenate(cls, group_list: Sequence["TokenGroup"], axis=-2):
        data = jnp.concatenate([t.tokens for t in group_list], axis=axis)
        mask = jnp.concatenate([t.mask for t in group_list], axis=axis + 1)
        return cls(data, mask)
