"""Schema contract checks (S1, S5): the pydantic models stay in field-order
lockstep with the hand-drafted schemas, and every worked example validates.

Fast and pure: no ffmpeg, no neural stack.
"""

from __future__ import annotations

import json
import types
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from mrw import models

SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"

DOCUMENTS = [
    (models.SourceDocument, "source.schema.json", "source.json"),
    (models.AudioFeaturesDocument, "audio_features.schema.json", "audio_features.json"),
    (models.LyricsDocument, "lyrics.schema.json", "lyrics.json"),
    (models.VideoDocument, "video.schema.json", "video.json"),
    (models.Manifest, "manifest.schema.json", "manifest.json"),
]


def _load(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text())


def _resolve(node: dict, current: str) -> tuple[dict, str]:
    """Follow $ref (and single-schema allOf) to the node that has properties."""
    while True:
        if "$ref" in node:
            ref = node["$ref"]
            file, _, pointer = ref.partition("#")
            file = file or current
            target = _load(file)
            for part in [p for p in pointer.split("/") if p]:
                target = target[part]
            extra = {k: v for k, v in node.items() if k != "$ref"}
            node, current = {**target, **extra}, file
            continue
        if "allOf" in node and "properties" not in node:
            merged: dict = {}
            for sub in node["allOf"]:
                sub, current = _resolve(sub, current)
                for key in ("properties", "required"):
                    if key in sub:
                        merged.setdefault(key, {} if key == "properties" else [])
                        if key == "properties":
                            merged[key].update(sub[key])
                        else:
                            merged[key] += sub[key]
            node = {**{k: v for k, v in node.items() if k != "allOf"}, **merged}
            continue
        return node, current


def _nested_model(annotation):
    """The BaseModel inside X, Optional[X], list[X] or Optional[list[X]]."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if arg is not type(None):
                return _nested_model(arg)
        return None
    if origin is list:
        inner = _nested_model(typing.get_args(annotation)[0])
        return ("list", inner[1]) if inner else None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return ("one", annotation)
    return None


def _walk(model: type[BaseModel], node: dict, current: str, path: str, problems: list):
    node, current = _resolve(node, current)
    props = node.get("properties", {})
    model_keys = list(model.model_fields)
    if list(props) != model_keys:
        problems.append(f"{path}: schema {list(props)} != model {model_keys}")
        return
    for name in node.get("required", []):
        field = model.model_fields[name]
        # Absent-not-null: only a None default can drop the key from output.
        if not field.is_required() and field.default is None:
            problems.append(f"{path}.{name}: required by schema, may be absent in output")
    for name, field in model.model_fields.items():
        nested = _nested_model(field.annotation)
        if nested is None:
            continue
        kind, sub_model = nested
        sub_node, sub_current = _resolve(props[name], current)
        if kind == "list":
            sub_node, sub_current = _resolve(sub_node.get("items", {}), sub_current)
        _walk(sub_model, sub_node, sub_current, f"{path}.{name}", problems)


@pytest.mark.parametrize("model,schema,example", DOCUMENTS, ids=lambda x: getattr(x, "__name__", x))
def test_model_schema_lockstep(model, schema, example):
    problems: list[str] = []
    _walk(model, _load(schema), schema, model.__name__, problems)
    assert not problems, "\n".join(problems)


def _validator(schema_name: str):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    registry = Registry()
    for f in SCHEMAS.glob("*.schema.json"):
        registry = registry.with_resource(
            f.name, Resource.from_contents(json.loads(f.read_text()))
        )
    return Draft202012Validator(_load(schema_name), registry=registry)


@pytest.mark.parametrize("name", sorted(p.name for p in SCHEMAS.glob("*.schema.json")))
def test_schemas_are_valid_2020_12(name):
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(_load(name))


@pytest.mark.parametrize("example", sorted(p.name for p in (SCHEMAS / "examples").glob("*.json")))
def test_examples_validate(example):
    schema = example.replace(".json", ".schema.json")
    document = json.loads((SCHEMAS / "examples" / example).read_text())
    errors = list(_validator(schema).iter_errors(document))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


@pytest.mark.parametrize("model,schema,example", DOCUMENTS, ids=lambda x: getattr(x, "__name__", x))
def test_examples_parse_as_models(model, schema, example):
    # The worked examples must also be producible by the models.
    model.model_validate_json((SCHEMAS / "examples" / example).read_text())
