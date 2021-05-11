# Gaussian noise FITS image generator

This is a script for generating arbitrarily large FITS images filled with Gaussian noise, suitable for testing. The method for generating images larger than the available memory is taken from [the AstroPy documentation](https://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html).

Various additional options are provided, for example for adding different distributions of NAN and INF pixels. Not all of these options are fully compatible with the option to restrict memory usage.

## Requirements

Python 3, numpy, astropy.

## Installation

This is a single-file script. Copy or symlink it into your path.

## Usage

Type `make_image.py -h` for a list of options.
