import math
from typing import List, Tuple


def shard_batch_indices(
    input_length: int, mini_batch_size: int, num_mini_batches: int
) -> Tuple[List[List[int]], List[float]]:

    batch_size = mini_batch_size * num_mini_batches
    indices = list(range(input_length))
    mini_batch_indices = []
    weight = []
    for bdx in range(math.ceil(input_length / batch_size)):
        for mdx in range(num_mini_batches):
            idx = bdx * batch_size + mdx * mini_batch_size
            jdx = bdx * batch_size + (mdx + 1) * mini_batch_size
            mini_batch = indices[idx:jdx]
            if not mini_batch:
                break
            mini_batch_indices.append(mini_batch)
            weight.append(len(mini_batch) / batch_size)

    return mini_batch_indices, weight
