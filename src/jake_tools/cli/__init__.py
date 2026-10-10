import click

from .clockify import clockify
from .codex_usage import codex_usage_alert
from .newsletter import newsletter


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    pass


main.add_command(clockify)
main.add_command(codex_usage_alert)
main.add_command(newsletter)
