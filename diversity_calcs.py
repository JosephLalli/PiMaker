import numpy as np

def performPiCalc(read_cts):
    piCalcs = np.zeros(shape=(read_cts.shape[0:2] + (7,)), dtype=np.int32)
    piCalcs[:, :, 0] = np.sum(read_cts, axis=2)
    piCalcs[:, :, 1] = read_cts[:, :, 0] * read_cts[:, :, 1]
    piCalcs[:, :, 2] = read_cts[:, :, 0] * read_cts[:, :, 2]
    piCalcs[:, :, 3] = read_cts[:, :, 0] * read_cts[:, :, 3]
    piCalcs[:, :, 4] = read_cts[:, :, 1] * read_cts[:, :, 2]
    piCalcs[:, :, 5] = read_cts[:, :, 1] * read_cts[:, :, 3]
    piCalcs[:, :, 6] = read_cts[:, :, 2] * read_cts[:, :, 3]
    return piCalcs


def calcPerSitePi(piCalcs):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sum(piCalcs[:, :, 1:], axis=2) / ((piCalcs[:, :, 0]**2 - piCalcs[:, :, 0]) / 2)
    return np.nan_to_num(result)


def calcPerSamplePi(sitePi, length=None, printit=None):
    '''given numpy array of pi values, returns average per sample pi'''
    if length is None:
        length = sitePi.shape[1]
    return np.nansum(sitePi, axis=1) / length


def determine_read_frame(chrm_ref_idx, start, end):
    '''read frame notation: 0,1,2 = fwd1, fwd2, fwd3
                            3,4,5 = rev1, rev2, rev3'''
    fwd_rev_adjust = 0
    if np.sign(end - start) < 0:
        fwd_rev_adjust = 3
    return (start - chrm_ref_idx) % 3 + fwd_rev_adjust
