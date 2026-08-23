from typing import ClassVar

import pytest
from jinja2 import UndefinedError

from jake_tools.prompting import Prompt


def test_template_referencing_unknown_variable_is_rejected_on_definition() -> None:
    with pytest.raises(TypeError, match="not declared as fields"):

        class _Bad(Prompt):
            template: ClassVar[str] = "Hello {{ name }}"


def test_field_unused_by_template_is_rejected_on_definition() -> None:
    with pytest.raises(TypeError, match="never uses"):

        class _Bad(Prompt):
            template: ClassVar[str] = "Hello there"

            name: str


def test_strict_undefined_catches_mistyped_nested_access_at_render() -> None:
    class Risky(Prompt):
        template: ClassVar[str] = "Name: {{ person.naem }}"

        person: dict[str, str]

    with pytest.raises(UndefinedError):
        Risky(person={"name": "Ada"}).render()
