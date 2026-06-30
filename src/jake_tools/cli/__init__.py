import click

from .daily_report import daily_report
from .newsletter import newsletter
from .transcribe import transcribe
from .transcript import transcript


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    pass


main.add_command(daily_report)
main.add_command(newsletter)
main.add_command(transcript)
main.add_command(transcribe)
