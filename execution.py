"""Бюджеты и события одного ответа. Состояние не переносится между запросами."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class RunControl:
    max_models: int = 6
    max_searches: int = 3
    models: int = 0
    searches: int = 0
    repairs: int = 0
    observer: object = None
    statements: dict = field(default_factory=dict)
    limitation_search_done: bool = False

    def model(self, schema):
        if self.models >= self.max_models:
            raise BudgetExhausted("Исчерпан бюджет модельных проверок")
        self.models += 1
        self.emit({"stage": "searching" if schema == "Intent" else ("checking" if schema == "Verdict" else "generation"), "schema": schema})

    def search(self, purpose):
        if self.searches >= self.max_searches:
            raise BudgetExhausted("Исчерпан бюджет поиска")
        self.searches += 1
        self.emit({"stage": "searching", "purpose": purpose})

    def repair(self):
        if self.repairs:
            return False
        self.repairs += 1
        self.emit({"stage": "repairing"})
        return True

    def emit(self, event):
        if self.observer:
            try:
                self.observer(event)
            except Exception:
                # Ошибка отображения прогресса не отменяет проверенный ответ.
                pass

    def summary(self):
        return {"model_calls": self.models, "search_calls": self.searches, "repair_calls": self.repairs,
                "max_model_calls": self.max_models, "max_search_calls": self.max_searches,
                "limitations_search_done": self.limitation_search_done}


_current = ContextVar("legal_run_control", default=None)


def current_control():
    return _current.get()


@contextmanager
def controlled_run(control):
    token = _current.set(control)
    try:
        yield control
    finally:
        _current.reset(token)
