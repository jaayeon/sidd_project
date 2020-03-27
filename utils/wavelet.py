import pywt
import numpy as np

def normalize_coeffs(coeffs, ch_min, ch_max):
    assert coeffs.shape[1] == len(ch_max)
    assert coeffs.shape[1] == len(ch_min)

    out_coeffs = coeffs.clone()
    for i in range(coeffs.shape[1]):
        out_coeffs[:,i:i+1,:,:] = (coeffs[:,i:i+1,:,:] - ch_min[i]) / (ch_max[i] - ch_min[i])
    return out_coeffs

def unnormalize_coeffs(coeffs, ch_min, ch_max):
    assert coeffs.shape[1] == len(ch_min)
    assert coeffs.shape[1] == len(ch_max)

    out_coeffs = coeffs.clone()
    for i in range(coeffs.shape[1]):
        out_coeffs[:,i:i+1,:,:] = coeffs[:,i:i+1,:,:] * (ch_max[i] - ch_min[i]) +  ch_min[i]
    return out_coeffs

def standarize_coeffs(coeffs, ch_mean, ch_std):
    if not ch_mean or not ch_std:
        raise ValueError("ch_mean or ch_std is empty")
    out_coeffs = coeffs.clone()
    for i, (mean, std) in enumerate(zip(ch_mean, ch_std)):
        out_coeffs[:,i:i+1,:,:] = (coeffs[:,i:i+1,:,:] - mean) / std
    return out_coeffs

def unstandarize_coeffs(coeffs, ch_mean, ch_std):
    if not ch_mean or not ch_std:
        raise ValueError("ch_mean or ch_std is empty")
    out_coeffs = coeffs.clone()
    for i, (mean, std) in enumerate(zip(ch_mean, ch_std)):
        out_coeffs[:,i:i+1,:,:] = coeffs[:,i:i+1,:,:]  * std + mean
    return out_coeffs

def preprocess_coeffs(coeffs, ch_min, ch_max, ch_mean, ch_std):
    coeffs = normalize_coeffs(coeffs, ch_min=ch_min, ch_max=ch_max)
    coeffs = standarize_coeffs(coeffs, ch_mean=ch_mean, ch_std=ch_std)
    return coeffs

def postprocess_coeffs(coeffs, ch_min, ch_max, ch_mean, ch_std):
    coeffs = unstandarize_coeffs(coeffs, ch_mean=ch_mean, ch_std=ch_std)
    coeffs = unnormalize_coeffs(coeffs, ch_min=ch_min, ch_max=ch_max)
    return coeffs

def swt2d_c(img, wavelet='bior2.2', level=2):
    coeffs = pywt.swt2(img, wavelet, level=level)

    approx_list = []
    detail_list = []
    detail_list.append(coeffs[0][0])
    for a, d in coeffs:
        approx_list.append(a)
        detail_list.extend(d)

    return approx_list, detail_list

def iswt2d(approxs, details, wavelet='bior2.2'):
    coeffs = []
    for i in range(len(approxs)):
        if i == 0:
            coeff = (details[0], details[1:4])
        else:
            coeff = (approxs[i], details[1+3*i:1+3*(i+1)])
        
        coeffs.append(coeff)

    img = pywt.iswt2(coeffs, wavelet=wavelet)
    return img

def swt2d_rgb(img, wavelet='bior2.2', level=2):
    coeffs_b = pywt.swt2(img[:, :, 0], wavelet, level=level)
    coeffs_g = pywt.swt2(img[:, :, 1], wavelet, level=level)
    coeffs_r = pywt.swt2(img[:, :, 2], wavelet, level=level)

    approx_list = []
    detail_list = []
    detail_list.append(coeffs_b[0][0])
    detail_list.append(coeffs_g[0][0])
    detail_list.append(coeffs_r[0][0])
    for (a_b, d_b), (a_g, d_g), (a_r, d_r) in zip(coeffs_b, coeffs_g, coeffs_r):
        approx_list.append(a_b)
        approx_list.append(a_g)
        approx_list.append(a_r)
        # detail_list.extend(d_b)
        # detail_list.extend(d_g)
        # detail_list.extend(d_r)
        for (b, g, r) in zip (d_b, d_g, d_r):
            detail_list.append(b)
            detail_list.append(g)
            detail_list.append(r)
    return approx_list, detail_list

def swt2d(img, wavelet='bior2.2', level=2):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    nc = img.shape[2]
    coeffs = [pywt.swt2(img[:,:,c], wavelet=wavelet, level=level) for c in range(nc)]
    # print("len(coeffs):", len(coeffs))

    approx_list = []
    detail_list = []

    for c in range(nc):
        detail_list.append(coeffs[c][0][0])

    for coeff in zip(*coeffs):
        # print("len(coeff):", len(coeff))
        
        for c in range(nc):
            # print("len(coeff[{}]: {}".format(c, len(coeff[c])))
            approx_list.append(coeff[c][0])

        for d in zip(*[coeff[i][1] for i in range(nc)]):
            for di in range(nc):
                detail_list.append(d[di])
    
    # print("len(approx_list):", len(approx_list))
    # print("len(detail_list):", len(detail_list))
    return approx_list, detail_list

def iswt2d_rgb(approxs, details, wavelet='bior2.2'):
    approxs_b = []
    approxs_g = []
    approxs_r = []

    details_b = []
    details_g = []
    details_r = []

    coeffs_b = []
    coeffs_g = []
    coeffs_r = []

    for i in range(0, len(approxs), 3):
        approxs_b.append(approxs[i])
        approxs_g.append(approxs[i + 1])
        approxs_r.append(approxs[i + 2])

    for i in range(0, len(details), 3):
        details_b.append(details[i])
        details_g.append(details[i+1])
        details_r.append(details[i+2])

    for i in range(0, len(approxs_b)):
        if i == 0:
            coeff_b = (details_b[0], details_b[1:4])
            coeff_g = (details_g[0], details_g[1:4])
            coeff_r = (details_r[0], details_r[1:4])
        else:
            coeff_b = (approxs_b[i], details_b[1+3*i:1+3*(i+1)])
            coeff_g = (approxs_g[i], details_g[1+3*i:1+3*(i+1)])
            coeff_r = (approxs_r[i], details_r[1+3*i:1+3*(i+1)])
        
        coeffs_b.append(coeff_b)
        coeffs_g.append(coeff_g)
        coeffs_r.append(coeff_r)

    ch_b = pywt.iswt2(coeffs_b, wavelet=wavelet)
    ch_g = pywt.iswt2(coeffs_g, wavelet=wavelet)
    ch_r = pywt.iswt2(coeffs_r, wavelet=wavelet)

    # ch_b[ch_b > 255] = 255
    # ch_b[ch_b < 0] = 0
    # ch_g[ch_g > 255] = 255
    # ch_g[ch_g < 0] = 0
    # ch_r[ch_r > 255] = 255
    # ch_r[ch_r < 0] = 0

    # ch_b = ch_b.astype(np.uint8)
    # ch_g = ch_g.astype(np.uint8)
    # ch_r = ch_r.astype(np.uint8)

    img = np.dstack((ch_b, ch_g, ch_r))
    return img

def swt_lv2_dict(img, wavelet='bior2.2'):
    """
    2019.11.29
    It seems that trim_approx option behaves differently from the document.
    If trim_approx is False, approximation is shown in every level.
    Approximation should be included to recover original image from wavelet transform coefficients

    For 1D stationary wavelet transform, the document is writtne correctly
    """
    coeffs = pywt.swt2(img, wavelet=wavelet, level=2, start_level=0, axes=(-2,-1), trim_approx=True)
    LL2, LL, HH = coeffs
    LH2, HL2, HH2 = LL
    LH1, HL1, HH1 = HH

    coeffs_dict = {}
    coeffs_dict['LL2'] = LL2
    coeffs_dict['LH2'] = LH2
    coeffs_dict['HL2'] = HL2
    coeffs_dict['HH2'] = HH2
    coeffs_dict['LH1'] = LH1
    coeffs_dict['HL1'] = HL1
    coeffs_dict['HH1'] = HH1

    return coeffs_dict

def swt_lv2(img, wavelet='bior2.2'):
    coeffs = pywt.swt2(img, wavelet=wavelet, level=2, start_level=0, axes=(-2,-1), trim_approx=True)
    LL2, LL, HH = coeffs
    LH2, HL2, HH2 = LL
    LH1, HL1, HH1 = HH

    return (LL2, LH2, HL2, HH2, LH1, HL1, HH1)

    # Inverse swt 
    #https://groups.google.com/forum/embed/#!topic/pywavelets/1ZPcIL_MEg8

"""
Implement stationary wavelet transform with pytorch
Refer https://github.com/PyWavelets/pywt/blob/master/pywt/_swt.py
"""
