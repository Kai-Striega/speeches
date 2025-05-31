def _find_indices(self, xi):
    indices = []
    norm_distances = []

    for x, grid in zip(xi, self.grid):
        i = np.searchsorted(grid, x) - 1
        i[i < 0] = 0
        i[i > grid.size - 2] = grid.size - 2
        indices.append(i)

        denom = grid[i + 1] - grid[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_dist = np.where(denom != 0, (x - grid[i]) / denom, 0)
        norm_distances.append(norm_dist)
    return indices, norm_dist
