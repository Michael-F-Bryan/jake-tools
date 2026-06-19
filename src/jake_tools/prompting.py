"""Strongly-typed prompt templates.

A prompt is a Pydantic model: its template variables are typed fields, so the
caller gets autocomplete and type checking instead of stringly-typed render
calls. Each prompt lives next to the code that uses it, not in a global prompt
folder.

Two guards keep a prompt's fields in sync with its Jinja template:

* at class-definition time we reject a template that references a variable the
  model does not declare, and a field the template never uses;
* at render time ``StrictUndefined`` turns any missing (e.g. mistyped nested)
  variable into an error rather than silently emitting nothing.

The output side stays in sync automatically: ``Hermes`` injects the response
model's JSON schema, so the type is the single source of truth and prompts do
not hand-write a schema.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, ClassVar, Generic, TypeVar

from jinja2 import Environment, StrictUndefined, Template, meta
from pydantic import BaseModel

TResponse = TypeVar("TResponse", bound=BaseModel)


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


_ENV = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)
_ENV.filters["json"] = _to_json


class Prompt(BaseModel):
    """Base class for a prompt. Subclasses declare a ``template`` and its fields."""

    template: ClassVar[str]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        template = cls.__dict__.get("template")
        if template is None:
            # An intermediate/abstract base (e.g. StructuredPrompt) without its
            # own template. Concrete prompts always declare one.
            return

        used = meta.find_undeclared_variables(_ENV.parse(template))
        fields = set(cls.model_fields)

        unknown = used - fields
        if unknown:
            raise TypeError(
                f"{cls.__name__} template references variables not declared as fields: "
                f"{sorted(unknown)}"
            )

        unused = fields - used
        if unused:
            raise TypeError(
                f"{cls.__name__} declares fields its template never uses: {sorted(unused)}"
            )

    def render(self) -> str:
        return _compile(type(self)).render(**self.model_dump(mode="json"))


@lru_cache(maxsize=None)
def _compile(cls: type[Prompt]) -> Template:
    return _ENV.from_string(cls.template.strip())


class StructuredPrompt(Prompt, Generic[TResponse]):
    """A prompt whose reply parses into ``response_model``.

    The response type is given twice on purpose: as the generic argument (so
    ``Hermes.run_structured`` infers the return type) and as the ``response_model``
    class variable (the runtime value passed to the parser). Pydantic strips the
    parametrized base from ``__orig_bases__``, so the value can't be derived from
    the generic argument automatically.
    """

    response_model: ClassVar[type[BaseModel]]
