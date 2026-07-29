import click

from .ai_watch import ai_watch
from .clockify import clockify
from .newsletter import newsletter
from .transcribe import transcribe
from .transcript import transcript


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    pass


main.add_command(ai_watch)
main.add_command(clockify)
main.add_command(newsletter)
main.add_command(transcript)
main.add_command(transcribe)
