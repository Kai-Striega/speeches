def _evaluate_linear(self, indices, norm_distances):
    vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))
    shift_norm_distances = [1 - yi for yi in norm_distances]
    shift_indices = [i + 1 for i in indices]

    zipped1 = zip(indices, shift_norm_distances)
    zipped2 = zip(shift_indices, norm_distances)

    hypercube = itertools.product(zipped1, zipped2)
    value = np.array([0.])
    for h in hypercube:
        edge_indices, weights = zip(*h)
        weight = np.array([1.])
        for w in weights:
            weight = weight * w
        term = np.asarray(self.values[edge_indices]) * weight[vslice]
        value = value + term
    return value

