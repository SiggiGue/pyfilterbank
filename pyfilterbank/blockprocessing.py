from numpy import arange, array, concatenate, floor, zeros

def block_generator(data, sample_rate, block_len, overlap,
                    num_blocks_zeropad=None):
    """Returns a generator that slices an input array into overlapping blocks.

    Parameters
    ----------
    data : ndarray
        Input data is a scliceable object (e.g. ndarray)
    sample_rate : int
        The sampling rate of the signal
    block_len : int
        The length of one block in seconds.
    overlap : scalar
        The percentage of overlap between blocks.
    num_blocks_zeropad : int
        The number of blocks prepended and appended to the signal
        for block process.

    Returns
    -------
    block : ndarray
        Data block with len of block_len.
    index : int
        Position of the current block in samples.

    Example
    -------
    Example two with zeropadding
    ..code-block:: python

    data = ones(36)
    for block, index in block_generator(data, 6, 1, 0.5, 0.5):
        print(block, index)

    produces:
    | [ 0.  0.  0.  1.  1.  1.] 0
    | [ 1.  1.  1.  1.  1.  1.] 3
    | [ 1.  1.  1.  1.  1.  1.] 6
    | [ 1.  1.  1.  1.  1.  1.] 9
    | [ 1.  1.  1.  1.  1.  1.] 12
    | [ 1.  1.  1.  1.  1.  1.] 15
    | [ 1.  1.  1.  1.  1.  1.] 18
    | [ 1.  1.  1.  1.  1.  1.] 21
    | [ 1.  1.  1.  1.  1.  1.] 24
    | [ 1.  1.  1.  1.  1.  1.] 27
    | [ 1.  1.  1.  1.  1.  1.] 30
    | [ 1.  1.  1.  1.  1.  1.] 33
    | [ 1.  1.  1.  0.  0.  0.] 36

    """

    block_samples = round(block_len * sample_rate)
    num_samples = len(data)

    if num_blocks_zeropad:
        num_samples_zeropad = num_blocks_zeropad * block_samples
        num_samples += num_samples_zeropad
        if data.ndim==1:
            zeropad = zeros(num_samples_zeropad)
        else:
            zeropad = zeros((num_samples_zeropad, data.shape[1:]))
        data = concatenate((zeropad, data, zeropad))

    overlap_samples = round(block_samples * overlap)
    shift_samples = block_samples - overlap_samples
    num_blocks = floor((len(data)-overlap_samples) / shift_samples)

    for i in arange(0, num_blocks):
        samples = data[i*shift_samples:i*shift_samples + block_samples]
        yield array(samples, copy=True), int(i*shift_samples)
