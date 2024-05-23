import weakref


class WeakRef(weakref.ref):
    def __init__(self, obj, callback=None):
        super().__init__(obj, callback)
        self._id = id(obj)

    def __hash__(self) -> int:
        return self._id


class WeakSet(weakref.WeakSet):
    def __init__(self, data=None):
        super().__init__(data)

    def add(self, item):
        if self._pending_removals:
            self._commit_removals()
        self.data.add(WeakRef(item, self._remove))

    def remove(self, item):
        if self._pending_removals:
            self._commit_removals()
        self.data.remove(WeakRef(item))

    def discard(self, item):
        if self._pending_removals:
            self._commit_removals()
        self.data.discard(WeakRef(item))

    def __contains__(self, item):
        try:
            wr = WeakRef(item)
        except TypeError:
            return False
        return wr in self.data
