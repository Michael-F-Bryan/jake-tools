"""The jake-tools MCP server, run as ``python -m jake_tools.mcp``.

This package has its own entrypoint, separate from the ``jake-tools`` CLI,
so that nothing on this path loads ``.env``. It never imports
:mod:`jake_tools.__main__` or ``dotenv``.
"""
