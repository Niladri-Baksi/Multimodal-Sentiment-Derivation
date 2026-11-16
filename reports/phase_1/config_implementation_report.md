# Phase 1 - Step 2: Central Configuration Utility

## Functions implemented

Implemented in `src/sentiment_derivation/utils/config.py`:

- `load_config(config_path: str | pathlib.Path) -> dict[str, Any]`
- `validate_required_sections(config: Mapping[str, Any], required_sections: Sequence[str]) -> None`

`load_config` uses `pathlib.Path` and `yaml.safe_load`. It requires an existing file, a YAML mapping at the document root, and string keys for every top-level mapping entry. `validate_required_sections` checks only for required top-level key presence; it does not define or enforce section contents.

## Contracts and error handling

| Function | Input | Output | Errors |
|---|---|---|---|
| `load_config` | String or `Path` to a YAML file | A Python dictionary containing the parsed mapping | `FileNotFoundError` for absent paths; `IsADirectoryError` for directories; `ValueError` with path/context for YAML syntax errors, empty/null YAML, a non-mapping root, or non-string top-level keys. |
| `validate_required_sections` | Mapping and sequence of required top-level section names | `None` when all named keys exist | `TypeError` if the configuration is not a mapping; `ValueError` listing all missing section names. |

Present-but-null sections count as present. No assumptions are made about each section's internal schema. I/O and decoding errors other than the explicitly described cases propagate to the caller.

## Tests created

`tests/test_config.py` contains 11 tests, using temporary files where file input is needed, for:

1. Valid YAML mapping loaded from a string path.
2. Valid YAML mapping loaded from a `pathlib.Path`.
3. Missing-file exception.
4. Directory path raises `IsADirectoryError`.
5. Empty YAML rejection.
6. Invalid YAML syntax and contextual error.
7. List root rejection.
8. Non-string top-level YAML key rejection.
9. Required sections present.
10. Missing required sections named in the error.
11. Non-mapping input to required-section validation raises `TypeError`.

The tests do not use project data, model weights, or pretrained modules.

## Test execution results

Command attempted:

```text
python -m pytest tests/test_config.py -q
```

| Result | Count / detail |
|---|---|
| Passed | 0 |
| Failed assertions | 0 |
| Not executed | 11 test cases; pytest could not start. |
| Execution error | `No module named pytest` |

No packages were installed. The test outcomes are therefore unverified; the command failure is an unavailable test-runner condition, not a test assertion failure.

## Remaining limitations

- `scripts/prepare_data.py` still uses its local loader; integration with the new utility is intentionally deferred.
- Required-section validation checks key presence only. Callers remain responsible for validating each section's value types and deeper schema.
- Test execution must be repeated in an environment with the repository's declared pytest and PyYAML dependencies available.
