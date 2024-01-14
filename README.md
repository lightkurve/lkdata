# Lightkurve

[![Coverage badge](https://github.com/tessgi/secret/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/tessgi/secret/tree/python-coverage-comment-action-data)

Must add to developer instructions;

- deprecation steps

```
git clone ...
pip install poetry --upgrade
make install
```

## Lightkurve; now more easy to develop

I'm going to to try to make it so that it is easier to develop for lightkurve. This includes some new things;

- pre-commit hooks. This is going to stop anyone from commiting anything to their branch that breaks our standards. That means linted, formatted code, well written markdown files, checking with mypy etc. This means it's going to be harder (but not impossible) for you to open a PR against lightkurve that doesn't have these things fixed. This is much more strict than V2.
- docs that comiple better. We're going to try to make documentation that is easier to compile and upload by using a sphinx gallery.
