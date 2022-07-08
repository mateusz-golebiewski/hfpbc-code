import mumax3PP.ovf as ovf
import mumax3PP.parameters as parameters
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import multiprocessing as mp
import matplotlib.colors as colors
import cProfile

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()

    return profiled_func

# @do_cprofile
def FFT_across_xyzm(A, times):
    shape = A.shape
    
    dt = (times[-1]-times[0])/times.shape[0]
    freqs = np.fft.rfftfreq(shape[0], dt)
    print(A.shape)
    pool = mp.Pool(processes=int(mp.cpu_count()-1))
    A = A.reshape(shape[0], np.prod(shape[1:])).T
    A = np.array(pool.map(np.fft.rfft, A))
    pool.close()
    pool.join()
    
    A = (A.T).reshape( int(shape[0]/2+1), shape[1], shape[2], shape[3], shape[4])
    print(shape, "vs", A.shape)
    
    
    return A, freqs


import threading

# @do_cprofile
def FFT_across_xyzm_threads(A, times):
    shape = A.shape
    print(A.shape)
    
    dt = (times[-1]-times[0])/times.shape[0]
    freqs = np.fft.rfftfreq(shape[0], dt)
    
#     pool = mp.Pool(processes=int(mp.cpu_count()-1))
    nThreads = 28
    A = A.reshape(shape[0], np.prod(shape[1:]))
    B = (np.zeros( (freqs.shape[0], A.shape[1]) )).astype(np.complex64)
    print("-->", A.shape)
    class my_thread (threading.Thread):
        def __init__(self, numbOfThreads, threadID):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.numbOfThreads = numbOfThreads
        def run(self):
            coords1 = [i*self.numbOfThreads+self.threadID for i in range(int(A.shape[1]/self.numbOfThreads)+1 )]
            coords = []
            for el in coords1:
                if el<A.shape[1]: coords.append(el)  
            for i in coords:
                B[:freqs.shape[0],i] = np.fft.rfft(A[:,i])
    
    threads_li = []
    for idxOfThread in range(nThreads):
        threads_li.append( my_thread( nThreads, idxOfThread )) # "Thread-"+str(j), j) )
    for j in range(nThreads):
        threads_li[j].start()

    threadLock = threading.Lock()    
    for t in threads_li:
        t.join()
    print("Exiting Main Thread")
    del(A)
    A = B.reshape( int(shape[0]/2+1), shape[1], shape[2], shape[3], shape[4])
    print(shape, "vs", A.shape)
    
    return A, freqs