#!/usr/bin/env python3

# Script for generating synthetic FITS files
# Recipe for writing files too large to fit in memory taken from:
# https://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html

import argparse
import itertools
import sys
import numpy as np
from astropy.io import fits

def make_image(args):
    dims = tuple(args.dimensions)

    # create header

    dummy_dims = tuple(1 for d in dims)
    dummy_data = np.zeros(dummy_dims, dtype=np.float32)
    hdu = fits.PrimaryHDU(data=dummy_data)

    header = hdu.header
    for i, dim in enumerate(dims, 1):
        header["NAXIS%d" % i] = dim

    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"

    if len(dims) >= 3:
        header["CTYPE3"] = "FREQ"

    if len(dims) >= 4:
        header["CTYPE4"] = "STOKES"

    header.tofile(args.output, overwrite=True)

    # create full-sized zero image

    header_size = len(header.tostring()) # Probably 2880. We don't pad the header any more; it's just the bare minumum
    data_size = (np.product(dims) * np.dtype(np.float32).itemsize)
    # This is not documented in the example, but appears to be Astropy's default behaviour
    # Pad the total file size to a multiple of the header block size
    block_size = 2880
    data_size = block_size * ((data_size//block_size) + 1)

    with open(args.output, "rb+") as f:
        f.seek(header_size + data_size - 1)
        f.write(b"\0")

    # write random data

    hdul = fits.open(args.output, "update", memmap=True)
    data = hdul[0].data

    if not args.max_bytes:
        strip_size = np.product(data.shape[-2:])
    else:
        strip_size = args.max_bytes // np.dtype(np.float32).itemsize

    total_size = np.product(dims)
    rounded_size = strip_size * (total_size // strip_size)
    remainder = total_size - rounded_size
    contiguous_data = data.ravel()

    for i in range(0, rounded_size, strip_size):
        contiguous_data[i:i+strip_size] = np.random.default_rng().normal(size=strip_size).astype(np.float32)

    contiguous_data[rounded_size:total_size] = np.random.default_rng().normal(size=remainder).astype(np.float32)

    hdul.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic FITS file.")
    parser.add_argument("dimensions", metavar="N", type=int, nargs="+", help="The dimensions of the file, in order (XY, XYZ or XYZW), separated by spaces.")
    parser.add_argument("-m", "--max-bytes", type=int, help="The maximum size of image data (in bytes) to create in memory at once. Default is the channel size.")
    parser.add_argument("-o", "--output", help="The output file name.")

    args = parser.parse_args()

    if not (2 <= len(args.dimensions) <= 4):
        sys.exit("At least two dimensions required. At most 4 dimensions allowed.")

    if not args.output:
        args.output = "image-%s.fits" % "-".join(str(d) for d in args.dimensions)

    make_image(args)
