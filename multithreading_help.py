from threading import Thread

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def split(l, n):
    """Splits l into n almost-equally sized chunks."""
    return chunks(l, -(-len(l)/n))

def map_async(fn, arr, pool_size = 8):
    """Like a multithreaded map(), dividing the task among pool_size seperate
    threads."""
    from multiprocessing import Pool
    return Pool(processes=pool_size).map(fn, arr)
