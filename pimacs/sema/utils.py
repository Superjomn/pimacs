from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional


class Scoped:
    @abstractmethod
    def push_scope(self, kind: str = ""):
        pass

    @abstractmethod
    def pop_scope(self):
        pass

    @contextmanager
    def scope_guard(self, kind: str = ""):
        self.push_scope(kind)
        try:
            yield
        finally:
            self.pop_scope()
