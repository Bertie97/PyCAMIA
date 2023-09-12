
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou", 
    create = "2023-09",
    fileinfo = "Useful python structures",
    requires = ""
)

__all__ = """
    struct
""".split()

class struct(dict):
    def __getattribute__(self, name):
        try: return super().__getattribute__(name)
        except AttributeError: return self[name]
