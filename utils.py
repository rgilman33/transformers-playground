import numpy as np

def make_sine(wavelength_scale=4, length=1e6, add_noise=False):
    """ Generate a single sine wave """
    ix = np.random.randint(-4*wavelength_scale, 4*wavelength_scale)
    x = np.arange(ix, ix+length)
    w = np.sin(x / 1.0 / wavelength_scale).astype('float64')
    if add_noise: w+= np.random.uniform(-0.1, 0.1, length)
    return w

def invert_sine_wave(w):
    """ Invert sine wave every fourth wave. """
    seq = [w[0]]
    invert = False
    counter = 0
    for i in range(1, len(w)):
        d = w[i]; d_prev = w[i-1]
        #if d < 0 and d_prev > 0 and np.random.random() > .30: 
        if d < 0 and d_prev > 0: 
            counter += 1
            if counter==4:
                invert = True
                counter = 0
        elif d > 0 and d_prev < 0: 
            invert = False

        if invert: 
            seq.append(w[i]*-1)
        else: 
            seq.append(w[i])
            
    return np.array(seq)

def create_diablo_sequence(seq_len):
    """ Quick hack to generate a complex sequence of obs size 4 """
    m = make_sine(4, seq_len);
    mm = make_sine(40, seq_len);
    mmm = make_sine(400, seq_len);
    seq = invert_sine_wave(m)
    seq2 = m * mm +.2
    seq3 = (seq * -1)
    seq4 = (seq2 * -1)
    seq += mm; seq2 += mm; seq3 += mm; seq4 += mm
    seq += mmm; seq2 += mmm; seq3 += mmm; seq4 += mmm
    seq = np.stack([seq,seq2,seq3,seq4]).T
    return seq


def batchify(long_seq, bs=35):
    """ Split up a single long sequence into bs shorter ones. Return bs first like for other seq models (e.g. LMs) """
    TOTAL_SEQ_LEN = len(long_seq) // bs; 
    batched_seqs = []; ix = 0
    while ix < len(long_seq):
        batched_seqs.append(long_seq[ix:ix+TOTAL_SEQ_LEN,None,:]); 
        ix += TOTAL_SEQ_LEN
    batched_seqs = np.concatenate([s for s in batched_seqs if len(s)==TOTAL_SEQ_LEN], axis=1); 
    return batched_seqs