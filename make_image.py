#!/usr/bin/env python3

# Script for generating synthetic FITS files
# Recipe for writing files too large to fit in memory taken from:
# https://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html

import argparse
import itertools
import sys
import numpy as np
from astropy.io import fits

def control_image(args):
    dims = tuple(reversed(args.dimensions))
    hdu = fits.PrimaryHDU(data=np.random.default_rng().normal(size=dims).astype(np.float32))
    hdu.writeto('control.fits', overwrite=True)

def make_zero_image(args):
    dims = tuple(args.dimensions)
    dummy_dims = tuple(1 for d in dims)
    dummy_data = np.zeros(dummy_dims, dtype=np.float32)

    hdu = fits.PrimaryHDU(data=dummy_data)

    header = hdu.header
    for i, dim in enumerate(dims, 1):
        header['NAXIS%d' % i] = dim
    header.tofile('test.fits', overwrite=True)
    
    header_size = len(header.tostring()) # Probably 2880. We don't pad the header any more; it's just the bare minumum
    data_size = (np.product(dims) * np.abs(header['BITPIX']//8))
    # This is not documented in the example, but appears to be Astropy's default behaviour
    # Pad the total file size to a multiple of the header block size
    block_size = 2880
    data_size = block_size * ((data_size//block_size) + 1)

    with open('test.fits', 'rb+') as f:
        f.seek(header_size + data_size - 1)
        f.write(b'\0')

def write_random_data(args):
    hdul = fits.open('test.fits', 'update', memmap=True)
    data = hdul[0].data
    channel_shape = data.shape[-2:]
    other_dims = data.shape[:-2]
    for channel_index in itertools.product(*(range(d) for d in other_dims)):
        # TODO split further into max chunks
        data[channel_index][:] = np.random.default_rng().normal(size=channel_shape).astype(np.float32)
    hdul.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a synthetic FITS file.')
    parser.add_argument('dimensions', metavar='N', type=int, nargs='+',
                    help='The dimensions of the file, in order (XY, XYZ or XYZW), separated by spaces.')

    args = parser.parse_args()
    
    if len(args.dimensions) < 2:
        sys.exit("At least two dimensions required.")
    
    make_zero_image(args)
    write_random_data(args)
    control_image(args)
