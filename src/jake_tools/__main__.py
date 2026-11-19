"""``jake-tools``: the CLI entrypoint.

Configuration is flags and the process environment only. No ``.env`` is
loaded (not from the working directory, not from any ancestor) and there is no
config file; see :mod:`jake_tools.config`.
"""

from .cli import main

if __name__ == "__main__":
    main()
