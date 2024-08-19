# STICKERMAN
Generates a sticker-like contour for a PNG logo. The logo has to be a PNG file with the alpha channel present as the algorithm extracts the alpha channel rather to identify the edges.

## Features
- Generate a contour along the logo edges that looks like a sticker
- Generate a vector path data of the identified sticker area

## Installation
- [Poetry](https://python-poetry.org/docs/)

```sh
poetry install
```

## Running
```sh
poetry run python sticker.py <FILE_PATH_TO_PNG_LOGO>
```

## TODO
- Code refactor
- Documentation
- Test