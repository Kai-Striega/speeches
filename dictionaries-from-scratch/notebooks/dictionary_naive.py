class Dictionary:
    def __init__(self, size=8):
        self._data = [None] * size

    def __setitem__(self, key, value):
        h = hash(key)
        i = h % len(self._data)
        self._data[i] = value

    def __getitem__(self, key):
        h = hash(key)
        i = h % len(self._data)
        return self._data[i]