import click

from .clockify import clockify
from .newsletter import newsletter


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    pass


main.add_command(clockify)
main.add_command(newsletter)
