import numpy as np

rad2arcsec = np.degrees(3600)
arcsec2rad = np.radians(1/3600)

def crop_image(img, cen, wid):
    if cen is None:
        cen = np.array(img.shape)[-2:].astype(int)//2

    sub_img = img[..., max(0, int(cen[1] - wid)) : min(img.shape[-2], int(cen[1] + wid)), \
                       max(0, int(cen[0] - wid)) : min(img.shape[-1], int(cen[0] + wid))].copy()

    return sub_img

def mask_image(img, cen, wid=5, value=0.):
    img[..., max(0, int(cen[1] - wid)) : min(img.shape[-2], int(cen[1] + wid)), \
             max(0, int(cen[0] - wid)) : min(img.shape[-1], int(cen[0] + wid))] = value
    return img

def pad_array(inarr, num_pad=1, NN=None):
    if NN is None:
        NN = int(inarr.shape[-1]*num_pad)
    return np.pad(inarr, (NN - inarr.shape[-1])//2)

def round_aperture(img, dr=0):
    #Build radius values
    rhoi = get_image_radii(img.shape[-2:])
    rtest = rhoi >= (min(img.shape[-2:]) - 1.)/2. - dr

    #Set electric field outside of aperture to zero (make aperture circular through rtest)
    img[...,rtest] = 0.

    #cleanup
    del rhoi, rtest

    return img

def get_image_radii(img_shp, cen=None):
    yind, xind = np.indices(img_shp)
    if cen is None:
        cen = [(img_shp[-2] - 1)/2, (img_shp[-1] - 1)/2]
    return np.hypot(xind - cen[0], yind - cen[1])
