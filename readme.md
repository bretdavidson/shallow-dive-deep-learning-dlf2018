# Shallow Dive into Deep Learning

Using [backslide](https://github.com/sinedied/backslide).

Install backslide: `npm install -g backslide`

Serve slides in development: `bs s`

Export to _mostly_ self-contained file: `bs e`. Look for the results in `dist`.

_Currently backslide does not inline video._

Create PDF: `bs p`. Look for the result in `pdf`.

## Embedding Media

If you need to embed video (and poster image) or audio you can use this fork of backslide:
https://github.com/jronallo/backslide

`bs e --media`
