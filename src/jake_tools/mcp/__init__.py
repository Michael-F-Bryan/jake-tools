"""The jake-tools MCP server, run as ``python -m jake_tools.mcp``.

This package has its own entrypoint, separate from the ``jake-tools`` CLI.
Like every ``jake_tools`` entrypoint it is configured by flags and the process
environment only: no config file and no ``.env``. It never imports ``dotenv``.
"""
