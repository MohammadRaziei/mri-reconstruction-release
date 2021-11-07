import os, glob, sys, io
from pathlib import Path

from pprint import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import scipy
import scipy.signal
import scipy.interpolate
import cupy as cp

import ismrmrd
import ismrmrd.xsd

from tqdm.auto import tqdm, trange


# from IPython.core.display import display, HTML



parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default="data", help='Choose path where downloaded on')
parser.add_argument('--outdir', type=str, default="data-npz", help='Choose path where to be converted on')
args = parser.parse_args()

data_dir = args.datadir 
out_dir = args.outdir 
tables = pd.read_csv('mridata.csv')
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

filenames = glob.glob(data_dir+"/*.h5")

def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions

    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    return img



def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    return k


def convert_ismrmrd_to_cupy_array(filename):
    # Load file
    if not os.path.isfile(filename):
        print("%s is not a valid file" % filename)
        raise SystemExit
    dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)

    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    rNx = enc.reconSpace.matrixSize.x
    rNy = enc.reconSpace.matrixSize.y
    rNz = enc.reconSpace.matrixSize.z

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z
    rFOVx = enc.reconSpace.fieldOfView_mm.x
    rFOVy = enc.reconSpace.fieldOfView_mm.y
    rFOVz = enc.reconSpace.fieldOfView_mm.z

    lNx = rNx
    lNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
    lNz = enc.encodingLimits.kspace_encoding_step_2.maximum + 1

    # Number of Slices, Reps, Contrasts, etc.
    ncoils = header.acquisitionSystemInformation.receiverChannels
    if enc.encodingLimits.slice != None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1


    if enc.encodingLimits.repetition != None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1

    if enc.encodingLimits.contrast != None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1

    firstacq=0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        else:
            firstacq = acqnum
            break




    remove_oversampling_x = True

    # Initialiaze a storage array
    all_data = cp.zeros((nreps, ncontrasts, nslices, ncoils, lNz, lNy, rNx if remove_oversampling_x else eNx), 
                        dtype=cp.complex64)

    # Loop through the rest of the acquisitions and stuff
    for acqnum in trange(firstacq,dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        data = cp.asarray(acq.data)

        # Remove oversampling if needed
        if remove_oversampling_x and eNx != rNx:
            xline = transform_kspace_to_image(data, [1])
            x0 = (eNx - rNx) // 2
            x1 = (eNx - rNx) // 2 + rNx
            xline = xline[:,x0:x1]
            acq.resize(rNx,acq.active_channels,acq.trajectory_dimensions)
            acq.center_sample = rNx//2
            # need to use the [:] notation here to fill the data
            data = transform_image_to_kspace(xline, [1])

        # Stuff into the buffer
        rep = acq.idx.repetition
        contrast = acq.idx.contrast
        slice = acq.idx.slice
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        all_data[rep, contrast, slice, :, z, y, :] = data

    info = dict(lNx=lNx, lNy=lNy, lNz=lNz,
                eNx=eNx, eNy=eNy, eNz=eNz,
                rNx=rNx, rNy=rNy, rNz=rNz,
                eFOVx=eFOVx, eFOVy=eFOVy, eFOVz=eFOVz,
                rFOVx=rFOVx, rFOVy=rFOVy, rFOVz=rFOVz,
                ncoils=ncoils, nslices=nslices, nreps=nreps, ncontrasts=ncontrasts)
    return all_data, info


for i, filename in enumerate(filenames): 
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()

    uuid = Path(filename).stem
    outfile=os.path.join(out_dir, "%s.npz"%uuid)
    print("\n[{:03.0f}%]>> process on '{}'".format(100*(i+1)/len(filenames),uuid))
    if not os.path.exists(outfile):
        try:
            all_data, info = convert_ismrmrd_to_cupy_array(filename)
            np.savez(outfile, kspace=all_data.get(), info = info, allow_pickle=True)
        except:
            print("ERROR: '{}'".format(outfile))

