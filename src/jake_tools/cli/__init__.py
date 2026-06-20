import click

from .newsletter import newsletter
from .transcribe import transcribe


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    pass


main.add_command(newsletter)
main.add_command(transcribe)
