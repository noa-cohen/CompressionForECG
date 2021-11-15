import torch

#Calculate evalutation methods like in paper "An efficient compression of ECG signals using deep convolutional autoencoders"

def calc_MSE(orig,reconstruct):
    diff = orig - reconstruct
    diff = torch.pow(diff, 2)
    sum = torch.sum(diff, dim = 1) #sum ecg window and stay with batch size
    return sum


def calc_RMS(orig, reconstruct):

    assert orig.shape == reconstruct.shape, f'orig and reconstructed signal are not the same shape orig.shape={orig.shape} reconstract.shape={reconstruct.shape} '
    sum = calc_MSE(orig,reconstruct)
    sum = sum / orig.shape[1] #TOOD check this is ok in paper the value is D-1
    rms = torch.sqrt(sum)
    return rms

def calc_PRD(orig, reconstruct):
    assert orig.shape == reconstruct.shape, f'orig and reconstructed signal are not the same shape orig.shape={orig.shape} reconstract.shape={reconstruct.shape} '
    nominator = calc_MSE(orig,reconstruct)
    denominator = torch.pow(orig, 2)
    denominator = torch.sum(denominator, dim = 1)
    fraction = nominator/denominator
    prd = torch.sqrt(fraction)
    prd = prd*100
    return prd

def calc_PRDN(orig, reconstruct):
    assert orig.shape == reconstruct.shape, f'orig and reconstructed signal are not the same shape orig.shape={orig.shape} reconstract.shape={reconstruct.shape} '
    nominator = calc_MSE(orig, reconstruct)
    mean = torch.mean(orig, dim = 1)
    mean = mean.unsqueeze(dim = 1)
    mean = mean.repeat(1, orig.shape[1])
    denominator = calc_MSE(orig,mean)
    fraction = nominator/denominator
    prdn = torch.sqrt(fraction)
    prdn = prdn*100
    return prdn

def calc_SNR(orig, reconstruct):
    assert orig.shape == reconstruct.shape, f'orig and reconstructed signal are not the same shape orig.shape={orig.shape} reconstract.shape={reconstruct.shape} '
    mean = torch.mean(orig, dim = 1)
    mean = mean.unsqueeze(dim = 1)
    mean = mean.repeat(1, orig.shape[1])
    nominator = calc_MSE(orig,mean)
    denominator = calc_MSE(orig, reconstruct)
    fraction = nominator/denominator
    snr = 10*torch.log10(fraction)
    return snr

def calc_CR_DEEP():
    return 2000/62

def calc_QS(orig,reconstruct,cr):
    prd = calc_PRD(orig,reconstruct)
    return cr/prd


if __name__ == '__main__':
    orig = torch.ones(5)
    recon = orig + 1
    rms = calc_RMS(orig,recon)
    prd = calc_PRD(orig,recon)
    print(prd)

    cr = calc_CR_DEEP()
    print(f'cr ={cr}')
    prdn = calc_PRDN(orig,recon)
    print(f'prdn = {prdn}')
    print('Done')