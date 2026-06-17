import click

from .transcribe import transcribe


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    pass


main.add_command(transcribe)
