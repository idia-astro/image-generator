#!/usr/bin/env python3

# Script for generating synthetic FITS files
# Recipe for writing files too large to fit in memory taken from:
# https://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html

import argparse
import itertools
import sys
import numpy as np
from astropy.io import fits

NAN_OPTIONS = ("pixel", "row", "column", "channel", "stokes", "image")
INF_OPTIONS = ("positive", "negative")

def make_image(args):
    dims = tuple(args.dimensions)
    N = len(dims)

    # create header

    dummy_dims = tuple(1 for d in dims)
    dummy_data = np.zeros(dummy_dims, dtype=np.float32)
    hdu = fits.PrimaryHDU(data=dummy_data)

    header = hdu.header
    for i, dim in enumerate(dims, 1):
        header["NAXIS%d" % i] = dim
    
    if args.header:
        extraheader = fits.Header.fromstring(args.header, sep="\n")
        for card in extraheader.cards:
            header.append(card)

    header.tofile(args.output, overwrite=True)

    # create full-sized zero image

    header_size = len(header.tostring()) # Probably 2880. We don't pad the header any more; it's just the bare minumum
    data_size = (np.product(dims) * np.dtype(np.float32).itemsize)
    # This is not documented in the example, but appears to be Astropy's default behaviour
    # Pad the total file size to a multiple of the header block size
    block_size = 2880
    data_size = block_size * (((data_size - 1)//block_size) + 1)

    with open(args.output, "rb+") as f:
        f.seek(header_size + data_size - 1)
        f.write(b"\0")

    # write random data
    
    # optionally seed the random number generator
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()

    hdul = fits.open(args.output, "update", memmap=True)
    data = hdul[0].data

    if not args.max_bytes:
        strip_size = np.product(data.shape[-2:])
    else:
        strip_size = args.max_bytes // np.dtype(np.float32).itemsize

    total_size = np.product(dims)
    remainder = total_size % strip_size
    rounded_size = total_size - remainder
    contiguous_data = data.ravel()
    
    width, height = dims[0:2]
    channel_size = width * height
    depth = dims[2] if N > 2 else 1
    stokes = dims[3] if N > 3 else 1
    shaped_data = data.reshape(stokes, depth, height, width)
    
    def nan_sample(start, end):
        nan_sample_size = int(np.round((args.nan_density/100) * (end - start)))
        return rng.choice(range(start, end), nan_sample_size, False).astype(int)
    
    def inf_sample(start, end):
        inf_sample_size = int(np.round((args.inf_density/100) * (end - start)))
        return rng.choice(range(start, end), inf_sample_size, False).astype(int)
    
    inf_choice = [];
    if "positive" in args.infs:
        inf_choice.append(np.inf)
    if "negative" in args.infs:
        inf_choice.append(-np.inf)

    if "image" in args.nans:
        for i in range(0, rounded_size, strip_size):
            contiguous_data[i:i+strip_size] = np.nan
        contiguous_data[rounded_size:total_size] = np.nan
    elif args.checkerboard:
        ch = args.checkerboard
        for i in np.arange(0, width, ch * 2):
            for j in np.arange(0, height, ch * 2):
                shaped_data[:, :, i:i+ch, j:j+ch] = 0.75
                shaped_data[:, :, i+ch:i+ch*2, j+ch:j+ch*2] = 0.75
                shaped_data[:, :, i:i+ch, j+ch:j+ch*2] = 0.5
                shaped_data[:, :, i+ch:i+ch*2, j:j+ch] = 0.5
    else:
        for i in range(0, rounded_size, strip_size):
            contiguous_data[i:i+strip_size] = rng.normal(size=strip_size).astype(np.float32)
            if "pixel" in args.nans:
                contiguous_data[nan_sample(i, i + strip_size)] = np.nan
            if args.infs:
                sample = inf_sample(i, i + strip_size)
                contiguous_data[sample] = rng.choice(inf_choice, sample.size)

        if remainder:
            contiguous_data[rounded_size:total_size] = rng.normal(size=remainder).astype(np.float32)
            if "pixel" in args.nans:
                contiguous_data[nan_sample(rounded_size, total_size)] = np.nan
            if args.infs:
                sample = inf_sample(rounded_size, total_size)
                contiguous_data[sample] = rng.choice(inf_choice, sample.size)
    
        # add row, column, channel and stokes nans here in separate passes
        # For now an implementation which ignores max_bytes
        
        if "stokes" in args.nans:
            nan_stokes = nan_sample(0, stokes)
            for s in nan_stokes:
                shaped_data[s] = np.nan
        
        if "channel" in args.nans:
            nan_channels = nan_sample(0, depth * stokes)
            for channel in nan_channels:
                s, c = divmod(channel, depth)
                shaped_data[s,c] = np.nan
                
        if "row" in args.nans:
            nan_rows = nan_sample(0, height * depth * stokes)
            for row in nan_rows:
                s, row = divmod(row, height * depth)
                c, y = divmod(row, height)
                shaped_data[s,c,y] = np.nan
                
        if "column" in args.nans:
            nan_columns = nan_sample(0, width * depth * stokes)
            for column in nan_columns:
                s, column = divmod(column, width * depth)
                c, x = divmod(column, width)
                shaped_data[s,c,:,x] = np.nan

    hdul.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic FITS file.")
    parser.add_argument("dimensions", metavar="N", type=int, nargs="+", help="The dimensions of the file, in order (XY, XYZ or XYZW), separated by spaces.")
    parser.add_argument("-m", "--max-bytes", type=int, help="The maximum size of image data (in bytes) to create in memory at once. Default is the channel size.")
    
    parser.add_argument("-n", "--nans", nargs="+", help="Options for inserting random NaNs, which are cumulative, separated by spaces. Any combination of %s. By default no NaNs are inserted. 'image' overrides all other options and creates an image full of NaNs. Channel, row and column algorithm currently ignores the --max-bytes restriction." % ", ".join(repr(o) for o in NAN_OPTIONS), default=[])
    parser.add_argument("-d", "--nan-density", type=float, help="The density of NaNs to insert, as a percentage. Default: 25. Ignored if -n/--nans is unset.", default=25.0)
    
    parser.add_argument("-i", "--infs", nargs="+", help="Options for inserting random INF pixels, which are cumulative, separated by spaces. Any combination of %s. By default no INFs are inserted." % ", ".join(repr(o) for o in INF_OPTIONS), default=[])
    parser.add_argument("--inf-density", type=float, help="The density of INFs to insert, as a percentage. Default: 1. Ignored if -i/--infs is unset.", default=1.0)
    
    parser.add_argument("-c", "--checkerboard", type=int, help="If this is set, the image is filled with a checkerboard pattern with each square the specified number of pixels. If it is unset, Gaussian noise is used. The checkerboard algorithm currently ignores the --max-bytes restriction. This option can be used together with the NaN options.")
    
    parser.add_argument("-o", "--output", help="The output file name.")
    
    parser.add_argument("-s", "--seed", help="Seed for the random number generator.", type=int)
    
    parser.add_argument("-H", "--header", help="Additional entries to append to the header, as a literal string with cards separated by newlines. Must be parseable by astropy.")

    args = parser.parse_args()

    if not (2 <= len(args.dimensions) <= 4):
        sys.exit("At least two dimensions required. At most 4 dimensions allowed.")

    if not args.output:
        args.output = "image-%s.fits" % "-".join(str(d) for d in args.dimensions)

    make_image(args)
