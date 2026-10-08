"""The jake-tools MCP server, run as ``python -m jake_tools.mcp``.

This package has its own entrypoint, separate from the ``jake-tools`` CLI,
so that nothing on this path loads ``.env`` or the torch loader workaround.
It never imports :mod:`jake_tools.__main__`, ``dotenv`` or anything under
:mod:`jake_tools.transcription`.
"""
