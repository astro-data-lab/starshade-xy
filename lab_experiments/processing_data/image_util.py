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

##############################################
###		Centroiding	###
##############################################

def centroid_image(img, noise_mult=5.):
    """Calculate center of light"""
    xTop, yTop, DN = 0., 0., 0.
    #Calculate noise threshold from mean of edges. ignore all pixels below threshold
    noiseThreshold = np.concatenate((img[0],img[-1],img[:,0],img[:,-1])).mean()*noise_mult
    #lower threshold if not enough pixels are above threshold
    cntr = 0
    while img[img > noiseThreshold].size < len(img)/4.  and cntr < 20:
        noiseThreshold *= 0.9
        cntr += 1
    #Centroid by Center of Light
    for i in range(len(img)):
        tt = (img[i][img[i] > noiseThreshold]-noiseThreshold).sum()
        yTop += i*tt
        DN += tt
    for j in range(len(img[0])):
        xTop += j*(img[:,j][img[:,j] > noiseThreshold]-noiseThreshold).sum()

    #Normalize
    if xTop != 0.:
        xRet = xTop/DN
    else:
        xRet = len(img)/2.
    if yTop != 0.:
        yRet = yTop/DN
    else:
        yRet = len(img)/2.

    return xRet, yRet

def get_centroid_pos(image, cen, wid, noise_mult=5):
    """Find the central location of a point source in the given image"""
    #Crop image
    newImg = crop_image(image, cen, wid)
    #Centroid the sub image
    cenx,ceny = centroid_image(newImg, noise_mult=noise_mult)
    #Add offsets to convert from sub image coordinates to image coordinates
    xcen = int(max(0, cen[0] - wid)) + cenx
    ycen = int(max(0, cen[1] - wid)) + ceny
    return xcen, ycen

def get_max_index(image):
    return np.unravel_index(np.argmax(image), image.shape)[::-1]

##############################################
##############################################
