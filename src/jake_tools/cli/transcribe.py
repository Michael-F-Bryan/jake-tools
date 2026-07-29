import click

from .transcript import obsidian_recording, polish

DEPRECATION_NOTE = (
    "Deprecated: use `jake-tools transcript obsidian-recording` / "
    "`jake-tools transcript polish` instead. This group forwards to the same "
    "commands and will be removed in a future release."
)


@click.group(hidden=True, help=DEPRECATION_NOTE)
def transcribe() -> None:
    pass


# Re-registering the same Command objects `transcript` uses, rather than
# wrapping them, so this alias can never drift from the real implementation.
transcribe.add_command(obsidian_recording, name="obsidian-recording")
transcribe.add_command(polish, name="polish")
