"Software for Systematic Long Range Phasing by Message Passing"
# Dependence on python 2.5
from __future__ import with_statement

# Indentation
#Xpylint: disable-msg=W0311
#pylint: "indent-string=   "
# Long lines
#pylint: disable-msg=C0301

# Naming conventions
#pylint: disable-msg=C0103

# Disable all conventions
#pylint: disable-msg-cat=C



import gc
import sys
import random
import time
import pdb
import threading
import signal
#gc.set_debug(gc.DEBUG_LEAK)

import handythread
from SLRPlib import __version__

try:
   from meminfo import resident
except ImportError:
   def resident(since=0.0):
      return 0



#sys.path.append("/nfs/users/nfs_k/kp5/Kuusamo/scripts/")


import SLRPlib.tools as tools
import SLRPlib.scanTools
from SLRPlib.tools import printerr

def handler(signum, frame):
   raise KeyboardInterrupt("Job is being terminated")

# Set the signal handler and a 5-second alarm
signal.signal(signal.SIGTERM, handler)

TAG_CHUNK_SIZE = 1
TAG_CHUNK = 0
TAG_NON_ROOT_GATHER = 2
TAG_SLICE_COORDS = 3
TAG_SLICE_DATA = 4

def info(err_type, value, tb):
   if hasattr(sys, 'ps1') or not sys.stderr.isatty():
      # we are in interactive mode or we don't have a tty-like
      # device, so we call the default hook
      sys.__excepthook__(err_type, value, tb)
   else:
      import traceback
      # we are NOT in interactive mode, print the exception...
      traceback.print_exception(err_type, value, tb)
      print
      # ...then start the debugger in post-mortem mode.
      pdb.pm()
   try:
      if MPI.Is_initialized() and not MPI.Is_finalized():
         printerr(MPI.Get_processor_name(),MPI.COMM_WORLD.Get_rank())
         MPI.Finalize()
   except NameError :
      pass

sys.excepthook = info

try:

#   import multiprocessing as mp
   import Queue
   canMultiproc=True

   class IterableQ:
      def __init__(self,q,block=False,stopper="StopIteration"):
         self.q = q
         self.__block=block
         self.stopper = stopper
         
         for a in dir(self.q):
            if a[0] != "_":
               printerr("Setting",a,"to",getattr(self.q,a))
               setattr(self,a,getattr(self.q,a))
               
      def __iter__(self):
         return self

      def next(self):
         try:
            r = self.get(block = self.__block)
         except Queue.Empty:
            printerr("Q empty")
            raise StopIteration()
         if r == self.stopper:
            raise StopIteration()
         return r
         
         
except ImportError ,e:
   canMultiproc=False
   








import optparse

#from simuPop_utils import *



from bwt import runlength_enc

import os


import itertools as it

try:
   from itertools import combinations
except ImportError:
   # For users with python < 2.6
   def combinations(iterable, r):
       # combinations('ABCD', 2) --> AB AC AD BC BD CD
       # combinations(range(4), 3) --> 012 013 023 123
       pool = tuple(iterable)
       n = len(pool)
       if r > n:
           return
       indices = range(r)
       yield tuple(pool[i] for i in indices)
       while True:
           for i in reversed(range(r)):
               if indices[i] != i + n - r:
                   break
           else:
               return
           indices[i] += 1
           for j in range(i+1, r):
               indices[j] = indices[j-1] + 1
           yield tuple(pool[i] for i in indices)

def writeOptions(opts,outstrm=sys.stdout,outFmt="%(optName)s: %(optValue)s\n"):
   "Write set option values"
   classattrs = dir(optparse.Values)
   for name in sorted((_ for _ in dir(opts) if _ not in classattrs)):
      outstrm.write(outFmt%{"optName":name,"optValue":str(getattr(opts, name))})

try:
   import numpy
except ImportError:
   printerr("Can not find the required numpy library!")
   raise

# Improve the perfomance of loading huge text files.
try:
   pass
   #import loadtxtHuge
   #numpy.loadtxt = loadtxtHuge.loadtxt
except ImportError,e:
   printerr("Couldn't find loadtxtHuge module. Loading huge text matrices will take time!")


class ThreadTagOutput(object):

   def __init__(self, f):
      self.decor = "[%s %s %.3gGB]"
      self.f = f
      self.l = threading.Lock()

   def __call__(self,*x):
      self.l.acquire()
      sys.stderr.write(self.decor%(threading.currentThread().name,time.asctime(),resident()*(2.0**-30)))
      self.f(*x)
      self.l.release()



try:
   import mpi4py.rc
   mpi4py.rc.initialize = False
   from mpi4py import MPI
   import time
   hasMPI = True

   class MPItagOutput(object):

      def __init__(self, f):

         mpi_size = MPI.COMM_WORLD.Get_size()
         mpi_rank = MPI.COMM_WORLD.Get_rank()
         mpi_name = MPI.Get_processor_name()

         self.decor="[%s:%d/%d %%s] "%(mpi_name, mpi_rank, mpi_size)
         self.f = f
         
      def __call__(self,*x):
         sys.stderr.write(self.decor%(time.asctime()))
         self.f(*x)
            

   
   TypeMap = dict(b=MPI.SIGNED_CHAR,
                  h=MPI.SHORT,
                  i=MPI.INT,
                  l=MPI.LONG,
                  #q=MPI.LONG_LONG,
                  f=MPI.FLOAT,
                  d=MPI.DOUBLE)



   
   class mpi_ndarrayRMA:
      def __win__getitem(self,coords):
         "Fragile! Does not check that the requested data is contiguous"
         xs = numpy.zeros((2,len(self.shape)), dtype = numpy.int).T
         #print MPI.COMM_WORLD.Get_rank(),coords, xs
         for firstCoord,x in enumerate(coords):
            try:
               xs[firstCoord,:] = x.indices(self.shape[firstCoord])[:2]
               break
            except AttributeError :
               x = int(x)
               #print xs[firstCoord,:],(x,x)
               xs[firstCoord,:] = (x,x)
         for i,x in enumerate(coords[firstCoord+1:]):
            assert x.indices(self.shape[i]) == (0,self.shape[i],1), "Bad RMA slicing! Only supporting a single continuous slice."


         offsets = numpy.dot(xs.T,self.strides)


         new_shape = [ xs[firstCoord,1]-xs[firstCoord,0] ] + list(self.shape[firstCoord+1:])
         
         #print new_shape,offsets,offsets[1]-offsets[0]
         ret = numpy.empty(new_shape, dtype = self.dataType)

         

         #self.win.Lock(MPI.LOCK_SHARED, rank = 0 , assertion = MPI.MODE_NOCHECK )
         self.win.Lock(MPI.LOCK_SHARED, rank = 0  )
         # Hope this works:
         self.win.Get(ret,  0, target = offsets[0] )
         self.win.Unlock(rank = 0)

         return ret
         
      def Finished(self):
         pass

      def Serve(self):
         pass
         
      def ADD(self,coords,other):
         "Fragile! Does not check that the requested data is contiguous"


         xs = numpy.zeros((2,len(self.shape)), dtype = numpy.int).T

         for firstCoord,x in enumerate(coords):
            try:
               xs[firstCoord,:] = x.indices(self.shape[firstCoord])[:2]
               break
            except AttributeError :
               x = int(x)
               xs[firstCoord,:] = (x,x)
         for i,x in enumerate(coords[firstCoord+1:]):
            assert x.indices(self.shape[i]) == (0,self.shape[i],1), "Bad RMA slicing! Only supporting a single continuous slice."


         offsets = numpy.dot(xs.T,self.strides)

         #if self.myRank == 1:
         #   print other
         #printerr("Asking for addition lock")

         #self.win.Lock(MPI.LOCK_SHARED, rank = 0 ,assertion = MPI.MODE_NOCHECK )
         self.win.Lock(MPI.LOCK_SHARED, rank = 0 )

         #printerr("Got MPI addition lock",self.win)
         #self.win.Lock(MPI.LOCK_SHARED, rank = 0  )
         # Hope this works:
         self.win.Accumulate(other,  0, target = offsets[0], op = MPI.SUM )
         #printerr("Accumulated")
         self.win.Unlock(rank = 0)
         #printerr("Released MPI addition lock")




      def __win__getslice(self,first,last):
         return self[first:last,]
         
      def __init__(self,data = None, copyData = True):
         self.data = data
         self.rank = MPI.COMM_WORLD.Get_rank()



         self.useRMA = not copyData

         self.shape = None
         self.strides = None

         if self.data is not None :
            self.dataType = self.dtype
            self.shape = self.data.shape
            self.strides = self.data.strides

         if self.useRMA:
            if self.rank == 0:
               #print self.data.flags
               self.win = MPI.Win.Create(self.data, comm = MPI.COMM_WORLD)
            else:
               self.win = MPI.Win.Create(None, comm = MPI.COMM_WORLD)
               
            self.__getitem__ = self.__win__getitem
            self.__getslice__ = self.__win__getslice
            self.__del__ = self.win.Free
            
            
      def Broadcast(self,root = 0 ):
         "Distribute data or it's stride information to all participants"
         
         comm = MPI.COMM_WORLD
         
         # 1st tell the size of the data

         if self.rank == root:
            comm.bcast( (self.shape,self.strides, self.dataType ) , root = root)
            if not self.useRMA:
               comm.Bcast(self.data, root = root)
         else:
            #printerr( MPI.Get_processor_name(),self.rank,self.shape,"Before")

            self.shape, self.strides, self.dataType = comm.bcast( None , root = root)
            self.shape = numpy.array(self.shape, dtype = numpy.int)
            self.strides = numpy.array(self.strides, dtype = numpy.int)
            
            #printerr(MPI.Get_processor_name(), self.rank,self.shape,"After")
            if not self.useRMA:
               self.data = numpy.empty(self.shape,dtype = self.dataType)
               #printerr(MPI.Get_processor_name(), self.rank,"Getting data")
               #self.data =
               comm.Bcast(self.data, root = root)
               #printerr(MPI.Get_processor_name(), self.rank,"Got:",self[1])

            
      def __getattr__(self,name):
         return getattr(self.data, name)
  

   class mpi_ndarrayMsg(mpi_ndarrayRMA):
      def __init__(self,data = None, copyData = True):
         self.data = data
         self.rank = MPI.COMM_WORLD.Get_rank()

         self.useRMA = not copyData

         self.shape = None
         self.strides = None

         if self.data is not None :
            self.dataType = self.dtype
            self.shape = self.data.shape
            self.strides = self.data.strides

         if self.useRMA:
            self.__getitem__ = self.__win__getitem
            

      def __win__getitem(self,coords):
         xs = numpy.zeros((2,len(self.shape)), dtype = numpy.int).T
         #print MPI.COMM_WORLD.Get_rank(),coords, xs
         for firstCoord,x in enumerate(coords):
            try:
               xs[firstCoord,:] = x.indices(self.shape[firstCoord])[:2]
               break
            except AttributeError :
               x = int(x)
               #print xs[firstCoord,:],(x,x)
               xs[firstCoord,:] = (x,x)
         for i,x in enumerate(coords[firstCoord+1:]):
            assert x.indices(self.shape[i]) == (0,self.shape[i],1), "Bad RMA slicing! Only supporting a single continuous slice."


         offsets = numpy.dot(xs.T,self.strides)


         new_shape = [ xs[firstCoord,1]-xs[firstCoord,0] ] + list(self.shape[firstCoord+1:])
         
         #print new_shape,offsets,offsets[1]-offsets[0]
         ret = numpy.empty(new_shape, dtype = self.dataType)

         comm = MPI.COMM_WORLD

         idx = numpy.array( [ offsets[0], ret.nbytes ], dtype = numpy.int )
         send_req = comm.Isend(idx, dest = 0, tag = TAG_SLICE_COORDS)

         recv_req = comm.Irecv(ret, source = 0, tag = TAG_SLICE_DATA)

         MPI.Request.Waitall([send_req,recv_req])
            #ret.shape = new_shape
         #send_req.Free()
         #recv_req.Free()
         #printerr(idx,ret)
         return ret

      def Finished(self):
         "Report that this worker has finished"
         comm = MPI.COMM_WORLD
         final = numpy.array([-1,0], dtype = numpy.int)
         comm.Isend(final, dest = 0, tag = TAG_SLICE_COORDS)

         
      def Serve(self):
         "Serve read requests to the workers. This is a hack but much more efficient than RMA"

         printerr("Starting to serve likelihoods")
         
         comm = MPI.COMM_WORLD
         worldSize = comm.Get_size()
         
         self.reqBuffers = [ numpy.empty(2, dtype = numpy.int)  for _ in range(1,worldSize) ]
         self.requests = [ comm.Irecv( rb, source = i+1, tag = TAG_SLICE_COORDS )
                           for i,rb in enumerate(self.reqBuffers) ]

         sendReqs = []
         sendBufs = []
         sendWorkers = []
         sendReplyTags = []
         sendCoords = []
         recvCoords = []
         recvBufs = []
         recvReqs = []
         workersGoing = range(1,worldSize)
         while len(self.requests) > 0 :
            # This should loadbalance the workers
            recv_count,recv_indices = MPI.Request.Waitsome(self.requests)
            #printerr("Sending data to workers %s"%(", ".join(str(workersGoing[x]) for x in recv_indices)))
            for i in  recv_indices:
               req = self.requests[i]
               offsetSize = self.reqBuffers[i]
               worker = workersGoing[i]

               #req.Free()
               
               if offsetSize[0] == -1:
                  del self.requests[i]
                  del self.reqBuffers[i]
                  del workersGoing[i]
                  printerr("Done on %d : %d"%(worker,len(self.requests)))
               else:
                  sendCoord = ( offsetSize[0] / self.data.itemsize ) , ( (offsetSize[0] + offsetSize[1] )/ self.data.itemsize )
                  sendBuf = self.flat[ sendCoord[0]:sendCoord[1] ].copy()
                  
                  sendReq = comm.Isend(sendBuf, dest = worker, tag = TAG_SLICE_DATA )
                  sendReqs.append(sendReq)
                  sendBufs.append(sendBuf)
                  sendCoords.append(sendCoord)
                  sendWorkers.append(worker)
                  sendReplyTags.append(offsetSize[0]^offsetSize[1]+10)
                  
                  self.requests[i] = comm.Irecv(offsetSize, source = worker, tag = TAG_SLICE_COORDS)
                  
            #printerr("Waiting for",sendReqs)   
            req_count,req_indices = MPI.Request.Waitsome(sendReqs)
            #printerr("Ready",req_indices)
            for i in sorted(req_indices, reverse = True):
               #sendReqs[i].Free()
               recvCoords.append(sendCoords[i])
               recvBufs.append(sendBufs[i])
               #printerr("Waiting for",sendReplyTags[i])
               recvReqs.append(comm.Irecv(recvBufs[-1], source = sendWorkers[i], tag = sendReplyTags[i]))
               del sendReqs[i]
               del sendBufs[i]
               del sendWorkers[i]
               del sendCoords[i]
               del sendReplyTags[i]

            # React to add requests
            req_count,req_indices = MPI.Request.Testsome(recvReqs)
            for i in sorted(req_indices, reverse = True):
               #printerr("Processing",i)
               self.flat[recvCoords[i][0] : recvCoords[i][1]] += recvBufs[i]
               del recvReqs[i]
               del recvBufs[i]
               del recvCoords[i]



      def __testAdd(self):
         if len(self.sendReqs) > 0:
            #printerr("Reqs:",len(self.sendReqs))
            req_count,req_indices = MPI.Request.Testsome(self.sendReqs)
            for i in sorted(req_indices, reverse = True):
               del self.sendReqs[i]
               del self.sendBufs[i]

         
      def ADD(self,coords,other):
         "Fragile! Does not check that the requested data is contiguous"

#         printerr("ADD",coords)
         xs = numpy.zeros((2,len(self.shape)), dtype = numpy.int).T

         for firstCoord,x in enumerate(coords):
            try:
               xs[firstCoord,:] = x.indices(self.shape[firstCoord])[:2]
               break
            except AttributeError :
               x = int(x)
               xs[firstCoord,:] = (x,x)
         for i,x in enumerate(coords[firstCoord+1:]):
            assert x.indices(self.shape[i]) == (0,self.shape[i],1), "Bad RMA slicing! Only supporting a single continuous slice."


         offsets = numpy.dot(xs.T,self.strides)
         tag = offsets[0] ^ other.nbytes + 10

         comm = MPI.COMM_WORLD

         req = comm.Isend(other, dest = 0, tag = tag)
         #printerr("Sending",tag)

         try:
            self.sendReqs.append(req)
            self.sendBufs.append(other)
         except AttributeError:
            self.sendReqs=[req]
            self.sendBufs=[other]

               
         self.__testAdd()
               

    
      
except ImportError :
   hasMPI = False
   pass



class ndarray_ADD(numpy.ndarray):
   "Unify the interface with the MPI implementation"
   def ADD(self,coords,val):
      self[coords] += val

from math import exp

    


try:
    product([1],[2])
except NameError:
    def product(*args, **kwds):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = map(tuple, args) * kwds.get('repeat', 1)
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)










def open(filename,mode="r",bufsize=-1):
    "Open a file, which is possibly gzipped."
    import gzip
    import os
    
    from __builtin__ import open as bopen
    if filename[-3:] == ".gz" or "r" in mode:
        f = gzip.open(filename,mode,bufsize)

        if "r" in mode:
            try:
                b = f.read(1)
                f._unread(b)
            except IOError,e:
                if e.message == 'Not a gzipped file':
                    f = bopen(filename,mode,bufsize)
                else:
                    raise
    else:
        f = bopen(filename,mode,bufsize)
        

    return f










def nlogsumnexp(p,q):
    "Return -log(exp(-p)+exp(-q)) calculated accurately"
    p,q=max(-p,-q),min(-p,-q)
    r=p+numpy.log1p(exp(q-p))
    return -r



try:
   from scipy import weave
   from scipy.weave import converters
   import scipy.linalg
except ImportError:
   printerr("Can not find the required scipy library!")
   raise



class ThreadBarrier:
   def __init__(self,n = 0):
      "Implement barrier synchronization for n threads. If n < 1, make barrier for all currently active threads"
      if n <= 0:
         self.n = threading.activeCount()
      else:
         self.n = n

      self.orig_n = self.n
      
      self.cv = threading.Condition()

   def Barrier(self):
      "Block for n-1 calls, release all threads at n:th"
      #printerr("Trying to acquire")
      self.cv.acquire()
      self.n -= 1
      printerr("N =",self.n+1,"=>",self.n)
      if self.n <= 0:
         #printerr("Notifying all")
         self.cv.notifyAll()
         self.n = self.orig_n
      else:
         #printerr("Waiting")
         self.cv.wait()
      #printerr("Progressing")
      self.cv.release()
   

class longRangePhase:
   __mean_recomb_rate = 1.5e-6 # Mean rcombination rate 1.5cM/Mbp
   
   def __init__(self, dampF=0.5, dType = numpy.float64):
      self.hLike = None
      self.CAj = None
      self.CPj = None
      self.firstCP2P = None 
      self.chrom = None
      self.__pylab_lock = threading.Lock()

      # Full probability of IBD between two random chromosomes.
      self.ibd_prob = 0.001

      # Default for non MPI
      self.myRank = -1
      self.worldSize = 0
      
      self.setDamping(dampF)

      # Data for threading
      self.poolSize = 1
      # iterateOverIBD synchronization variables
      self.__ioIBD_sync = ThreadBarrier(self.poolSize)
      self.__assist = threading.local()
      self.__inlining_lock = threading.Lock()

      self.ibd_regionsLoader = None 


      self.dataType = dType
      self.ibd_regions = None
      self.fad = None

      self.set_min_ibd_length()
      
      self.firstPhase=True
      #self.ibdSegments=set()
      self.ibdSegments_lock = threading.Lock()
      

      if self.dataType==numpy.float64:
         self.cDataType="double"
      elif self.dataType==numpy.float32:
         self.cDataType="float"
      else:
         assert ValueError()

      self.c_ext = SLRPlib.scanTools.c_ext(self.dataType)
      self.set_mutation_rate()
      self.set_ibd_transition_limit()


   def setChromosome(self,chrom):
      "Set the chromosome to be used"
      self.chrom = chrom


   def set_min_ibd_length(self,lim = 0):
      "Set the hard lower limit on IBD segment lengths. Cuts off problematic, very short IBD segments."
      self.minIBDlength = lim

   def set_ibd_transition_limit(self,lim = 0.1):
      "Set the upper limit for probability of transition from non IBD to IBD state"
      assert 0 < lim, "Non IBD to IBD transition probability limit must be greater than zero."
      self.IBDtransProbLimit = lim
      

   def setIBDprob(self,ibd_prob):
      "Set the prior IBD probability"
      assert 0 < ibd_prob < 0.25
      self.ibd_prob = ibd_prob

   def set_mutation_rate(self,mut_rate = 2 * 2.5e-8 *10000 ):
      """Set the population scaled per site mutation rate \Theta = 2 * N_e * u.
      For humans, u, the site specific mutation rate is about 2.5e-8 (Nachman, Crowell, Genetics 2000),
      and the effective population size, N_e, is about 1e4. For isolated population
      the effective size is probably smaller e.g. 1e3"""

      assert mut_rate>=0, "The mutation rate must be non negative!"
      self.scaled_mutation_rate = mut_rate
      #self.per_site_match_prob = 1.0/(1.0 + self.scaled_mutation_rate)
      self.log_per_site_match_prob = -numpy.log1p(self.scaled_mutation_rate)



      
   def setDamping(self,dampF):
      "Set damping factor for the message passing"
      self.dampF = dampF

      
   def setErrP(self,errP):
      "Set prior error probability for array genotypes"
      self.errP = errP

      logG0err=-2*numpy.log(1-errP)
      logG1err=-numpy.log(errP)-numpy.log(1-errP)
      logG2err=-2*numpy.log(errP)

      logHetErr=-numpy.log((1-2*numpy.exp(-logG1err))/2.0)

      self.logG={0:[logG0err,logG1err,logG1err,logG2err],
                 1:[logG2err,logG1err,logG1err,logG0err],
                 2:[logG1err,logHetErr,logHetErr,logG1err]}


      gZ = [numpy.log(sum(numpy.exp(-x) for x in self.logG[g])) for g in range(3)]

      #print gZ
      self.logG = dict( (g,tuple(x+gZ[g] for x in self.logG[g])) for g in range(3) )

      # For partially phased input
      self.logG_part={(0,0):[logG0err,logG1err,logG1err,logG2err],
                      (1,1):[logG2err,logG1err,logG1err,logG0err],
                      (2,2):[logG1err,logHetErr,logHetErr,logG1err],
                      (1,0):[logG1err,logG0err,logG1err,logG1err],
                      (0,1):[logG1err,logG1err,logG0err,logG1err]}


      gZ = dict((g, numpy.log(sum(numpy.exp(-x) for x in self.logG_part[g])) ) for g in self.logG_part.keys() )

      #print gZ
      self.logG_part = dict( (g,tuple(x+gZ[g] for x in self.logG_part[g])) for g in self.logG_part.keys() )



      # Figure out the limits on the white noise to add to break the ties
      altValues = set( it.chain( *self.logG.values() ) )
      self.__minDif = min(x-y for x in altValues for y in altValues if x>y)


   def loadGeno(self, fadFileName = None, tpedFileName = None, vcfFileName = None):
      "Load input genotypes in any format"
      self.indivs = 0
      self.markers = 0
      self.snpPos = numpy.empty((self.markers,), dtype = int)
      self.snpCMpos = numpy.empty((self.markers,), dtype = float )
      self.haplos = numpy.empty((self.markers,self.indivs,2),  dtype = numpy.int8)
#      self.zeroAF = numpy.empty((self.markers,), dtype = float )
      self.famInfo = []
      
      if fadFileName is not None:
         printerr("Loading %s"%(fadFileName))
         self.loadFAD(fadFileName)

      if tpedFileName is not None:
         printerr("Loading %s"%(tpedFileName))
         self.loadTPED(tpedFileName)

      printerr("Setting prior likelihoods")
      self.__setLikelihoods()

      if vcfFileName is not None:
         self.loadVCF(vcfFileName)

      self.hLike = numpy.ascontiguousarray(self.hLike)
      self.hLike = self.hLike.view(ndarray_ADD)

      self.snpCMpos =  self.snpPos * self.__mean_recomb_rate
      if not hasattr(self,"zeroAF"):
         printerr("Prior allele frequency distribution is Beta(%g,%g)."%(1.0,1.0))
         self.zeroAF = self.estimateZeroAlleleFrequencies()


   def loadVCF(self, vcfFileName):
      "Load VCF file with genotype likelihoods. Assume single chromosome with biallelic snps. VCF file must have PL filed with List of Phred-scaled genotype likelihoods, number of values is (#ALT+1)*(#ALT+2)/2"

      vcfFile = open(vcfFileName)
      magicLine = vcfFile.readline().strip()
      if magicLine != "##fileformat=VCFv4.1":
         printerr("Your VCF file has %s, which might cause trouble. I'm only prepared for VCFv4.1 (ish)."%(magicLine))

      vcfHeader = list(tools.takewhile2(lambda x:x[:6]!="#CHROM",vcfFile))
      #vcfFile = it.dropwhile(lambda x:x[:6]!="#CHROM",vcfFile)
      CHROMline = vcfHeader[-1].strip().split()
      indivs = CHROMline[9:]
      n_indivs = len(indivs)

      if self.chrom is None:
         VCFlines = (x.split() for x in vcfFile if x[0]!="#")
      else:
         __chr = self.chrom
         VCFlines = (x.split() for x in vcfFile if x[0]==__chr)
         
      def parseVCFsite(vcf_rec):
         PLidx = vcf_rec[8].split(":").index("PL")
         
         #r = [int(vcf_rec[1])]
         PL=[]
         for s_rec in vcf_rec[9:]:
            PL.append(tuple(map(int,s_rec.split(":")[PLidx].split(","))))
         return (int(vcf_rec[1]),vcf_rec[3],vcf_rec[4], tuple(PL))

      #VCFsites = it.chain(*(parseVCFsite(x)    for x in VCFlines if x[6]=="PASS"))
      VCFsites = (parseVCFsite(x)    for x in VCFlines if x[6]=="PASS")

      VCFdata = numpy.fromiter(VCFsites,dtype = [ ("bpPos",">i8"), ("zeroAllele","|S1"), ("oneAllele","|S1"),("geno_like",">i8",(n_indivs,3))] )
      #VCFdata = numpy.fromiter(VCFsites,dtype = [ ("bpPos",">i8"), ("geno_like",">i8",(n_indivs,3))] )
      #VCFdata = numpy.fromiter(VCFsites,dtype = [ ("bpPos",">i8"), ("zeroAllele","|S1"), ("oneAllele","|S1")] )
      #VCFdata = numpy.fromiter(VCFsites,dtype =">i8" )
      #VCFdata = VCFdata.view([ ("bpPos",">i8"), ("geno_like",">i8",(n_indivs,3))] )

      n_markers = len(VCFdata)
      vcfLike = numpy.empty((n_indivs,n_markers,4),dtype = self.dataType)
      vcfLike[:,:,0] = 0.1*VCFdata["geno_like"][:,:,0].T/numpy.log(10.0)
      vcfLike[:,:,1] = 0.1*VCFdata["geno_like"][:,:,1].T/numpy.log(10.0) - numpy.log(0.5)  # Split the het likelihood to the two phases
      vcfLike[:,:,2] = vcfLike[:,:,1] 
      vcfLike[:,:,3] = 0.1*VCFdata["geno_like"][:,:,2].T/numpy.log(10.0)

      genoHapTempl = numpy.array([[0,0],[2,2],[1,1]], dtype=numpy.int8)
      vcfHaplos = genoHapTempl[VCFdata["geno_like"].argmin(axis=2)]



      allPos = numpy.concatenate((VCFdata["bpPos"], self.snpPos))
      uniqPos,uniqInd,uniqInv = numpy.unique( allPos, return_index = True, return_inverse = True)
      # uniqPos is sorted
      
      new2vcf = uniqInv[:n_markers]
      new2old = uniqInv[n_markers:]

      tot_markers = len(uniqPos)
      tot_indivs = n_indivs + self.indivs


      new_like = numpy.empty((tot_indivs, tot_markers, 4), dtype = self.dataType, order = "C")
      new_like[:self.indivs, new2old,:] = self.hLike
      new_like[self.indivs:, new2vcf,:] = vcfLike

      new_haplos = numpy.empty((tot_markers, tot_indivs, 2), dtype = int )
      new_haplos[:] = 3
      new_haplos[new2old, :self.indivs, : ] = self.haplos
      new_haplos[new2vcf, self.indivs:, : ] = vcfHaplos
      

      # Check, and possibly flip alleles
      vcf_alleles = VCFdata[["zeroAllele","oneAllele"]].view("|S1").reshape((-1,2))
#       all_alleles = numpy.concatenate((self.alleles, vcf_alleles))
#       allele_order = uniqInv.argsort()
#       allele_grps = numpy.hstack((uniqInv[allele_order].reshape((-1,1)), all_alleles[allele_order]))
#       grps = it.groupby(allele_grps, lambda x:x[0])
      
      
      new_alleles = numpy.zeros((tot_markers,2), dtype = "|S1")
      new_alleles[new2old] = self.alleles

      (new_alleles[new2vcf] == vcf_alleles).prod(axis=1)


      # Watson-Crick base pairing
      WC = { "A":"T",
             "C":"G",
             "G":"C",
             "T":"A"}
      bad_snps = set()

      for i,a in enumerate(vcf_alleles):
         if (a == new_alleles[new2vcf[i]] ).all():
            continue
         elif (new_alleles[new2vcf[i]] == ["",""] ).all() or \
                  a[0] == WC[new_alleles[new2vcf[i]][0]] and a[1] == WC[new_alleles[new2vcf[i]][1]]:
            # The reported strand is different, but the genotypes are OK, or this is a new site
            new_alleles[new2vcf[i]] = a
            continue
         elif (a[::-1] == new_alleles[new2vcf[i]] ).all() \
                  or ( a[1] == WC[new_alleles[new2vcf[i]][0]] and a[0] == WC[new_alleles[new2vcf[i]][1]] ):
            if ( a[1] == WC[new_alleles[new2vcf[i]][0]] and a[0] == WC[new_alleles[new2vcf[i]][1]] ):
               new_alleles[new2vcf[i],0] = WC[a[1]]
               new_alleles[new2vcf[i],1] = WC[a[0]]
               
            # Flip haplotypes
            h = new_haplos[new2vcf[i],:self.indivs,:]
            h[h<2] = 1 - h[h<2]
            new_haplos[new2vcf[i],:self.indivs,:] = h
            
            new_like[:self.indivs,new2vcf[i],:] = new_like[:self.indivs,new2vcf[i],::-1]
         else:
            printerr("Bad alleles for %dth snp at %d: VCF:%s <-> prior:%s"%(i, uniqPos[new2vcf[i]], str(a), str(new_alleles[new2vcf[i]])))
            bad_snps.add(new2vcf[i])
            
      printerr("Found %d discordant allele snps"%(len(bad_snps)))

      bad_mask = numpy.ones(len(uniqPos))==1
      for i in bad_snps:
         bad_mask[i] = False

      self.hLike = new_like[:,bad_mask,:]
      self.haplos = new_haplos[bad_mask,:,:]
      self.markers, self.indivs = self.haplos.shape[:2]
      self.snpPos = uniqPos[bad_mask]
      self.alleles = new_alleles[bad_mask]
      self.famInfo.extend(indivs)
      



   def loadTPED(self, tpedFileName):
      "Load genotypes in tped format, with allelic codes"
      # TODO: fail if cant find
      assert self.indivs == 0, "Can't append individuals yet"
      import re
      tfamFileName = re.sub(r"[.]tped([.]gz)?$",".tfam",tpedFileName)

      #tfamFile = open(tpedFileName+".tfam")
      tfamFile = open(tfamFileName)

      tfam = [ ":".join(x.strip().split()[:2]) for x in tfamFile ]
      self.famInfo.extend(tfam)
      self.indivs = len(self.famInfo)

      tpedIndivs = len(tfam)
      #tpedFile = open(tpedFileName)
      self.tped =  numpy.loadtxt(tpedFileName,dtype = [ ("chrom",">i1"),("rsID","|S20"),("cmPos","f8"),("bpPos",">i8"), ("haplos","|S1",(tpedIndivs*2,) )] )

      if self.chrom is not None:
         # PLINK uses following mapping:
         #X    X chromosome                    -> 23
         #Y    Y chromosome                    -> 24
         #XY   Pseudo-autosomal region of X    -> 25
         #MT   Mitochondrial                   -> 26
         chrom2plink = {"X":23,
                        "Y":24,
                        "XY":25,
                        "MT":26}
         try:
            __chr = int(self.chrom)
         except ValueError:
            __chr = chrom2plink[self.chrom]
         self.tped = self.tped[ self.tped["chrom"] == __chr ]

      else:
         self.chrom = str(self.tped["chrom"][0])
         
      assert self.tped["chrom"].var() == 0.0, "Tried to give SNPs on more than one chromosome: "+str( numpy.unique(self.tped["chrom"]) )
      
      tpedMarkers = self.tped.shape[0]

      tpedAlleles = numpy.fromiter((numpy.intersect1d(x,"ACGT") for x in self.tped["haplos"]),dtype=[('zero',"|S1"),('one',"|S1")])
      tpedAlleles[tpedAlleles["one"]==""] = "."
      self.alleles = tpedAlleles.view("|S1")
      self.alleles.shape = (tpedMarkers, 2)
      numAlleles = tpedAlleles.view("int8")
      numAlleles.shape = (tpedMarkers,2)


      hapView = self.tped["haplos"].view("int8")
      hapView.shape = (tpedMarkers,self.indivs,2)

      self.haplos = numpy.empty(hapView.shape,dtype="int8")
      self.haplos[:] = 3
      hets = hapView[:,:,0]!=hapView[:,:,1]
      self.haplos[hets] = 2
      notHets = hets==False
      for i in range(self.haplos.shape[1]):
         self.haplos[numpy.logical_and(notHets[:,i],numAlleles[:,0] == hapView[:,i,0]),i,:] = 0
         self.haplos[numpy.logical_and(notHets[:,i],numAlleles[:,1] == hapView[:,i,0]),i,:] = 1

      assert self.haplos.var(axis=2).max()==0.0

      self.markers = tpedMarkers
      self.snpPos = self.tped["bpPos"]
      if self.tped["cmPos"].var() == 0:
         self.snpCMpos =  self.snpPos * self.__mean_recomb_rate
      else:
         self.snpCMpos = self.tped["cmPos"]

      assert 0<= self.haplos.min() < self.haplos.max() <= 3, "Funny allele values for haplotypes"
      
            
   def loadFAD(self, fadFileName):
      "Load the input file in FAD format"
      def fadFormatter(fad_line):
         fad_line = fad_line.strip().split()
         return ( int(fad_line[0]), [int(x) for x in fad_line[1]])

      fadFile = open(fadFileName)
      

      fline = fadFile.readline().strip().split()
      self.indivs = len(fline[1]) / 2
      self.famInfo.extend("%s_ind%d"%(fadFileName,i) for i in range(self.indivs))
      

      assert self.snpPos.shape[0] == 0, "FAD must be the first file format to read"
      
      self.fad = numpy.loadtxt(fadFileName,dtype = [ ("pos",">i8"), ("haplos","|S%d"%(self.indivs*2) )] )
      self.snpPos = self.fad["pos"]
      self.snpCMpos =  self.snpPos * self.__mean_recomb_rate
                                                     
      #self.fad = [ fadFormatter(x) for x in fadFile ]
      

      self.indivs,self.markers = len(self.fad[0][1])/2, len(self.fad)
      self.haplos = numpy.fromiter(it.chain(*self.fad["haplos"]),dtype=numpy.int8,count=self.markers*self.indivs*2)
      self.haplos.shape = (self.markers, self.indivs, 2)
      #self.snpPos = [ p for p,_ in self.fad ]
      #self.snpCMpos=[ float(p) / 1e6 for p in self.snpPos ]
      self.set_slice_len()
      self.alleles = numpy.fromiter( it.cycle("01"),dtype="|S1", count = 2*self.markers)
      self.alleles.shape = (self.markers,2)
      # Regularize the allele frequency estimate with symmetric beta prior

      



   def breakPhaseSymmetry(self, trimLength=0):
      "Break symmetries on individuals that don't have any phase information from FAD file. Set one of the heterozygotes after beginning of the ibdSegments to phased state."
      assert self.hLike is not None, "Likelihood table must be set up before breaking symmetries. Use initMessages()"

      indivsWithoutPhase = numpy.where(self.haplos.var(axis=2).max(axis=0) == 0.0)[0]
      if len(indivsWithoutPhase) > 0 :
         breakableSNPs = numpy.where(self.haplos[trimLength:,:,0].transpose()==2)
         break_marker_indiv = numpy.array([hets.next() for (i,hets) in it.groupby(it.izip(*breakableSNPs),lambda x:x[0])],dtype=int)
         break_marker_indiv[:,1]+=trimLength

         mask = numpy.in1d(break_marker_indiv[:,0], indivsWithoutPhase)

         self.hLike[break_marker_indiv[mask,0], break_marker_indiv[mask,1]] = self.logG_part[(0,1)]
         return sum(mask)
      else:
         return 0


   def estimateZeroAlleleFrequencies(self, prior_beta_param = 1.0): # By default, one "fake" heterozygote individual.
      "Estimate the frequency of the zero allele"
      #self.zeroAF = numpy.zeros(self.markers, dtype = float)

      hapView = self.haplos.view()
      hapView.shape = (self.haplos.shape[0],-1)
      g0 = (hapView==0).sum(axis=1)
      g1 = (hapView==1).sum(axis=1)
      gHet = (hapView==2).sum(axis=1)
      totalAlleles = g0 + g1 + gHet + 2 * prior_beta_param
      zeroAF = (g0 + 0.5*gHet + prior_beta_param)/totalAlleles

      return zeroAF
      
      
   def __setLikelihoods(self):
      "Initialize haplotype likelihood array"
      # This makes hLike[x:y,i,:] as C_contigious. Should be good for MPI and other things.
      self.hLike = numpy.empty((self.indivs, self.markers, 4),dtype = self.dataType,
                               order="C")

      tic = time.clock()
      printerr("Memory for likelihoods %gMB"%(self.hLike.nbytes*1.0/(2.0**20)))

      logGp=numpy.zeros((4,4,4),dtype=self.dataType)
      for k,v in self.logG_part.iteritems():
         logGp[k[0],k[1],:]=v
      self.hLike = logGp[self.haplos[:,:,0],self.haplos[:,:,1],:].transpose((1,0,2))


      toc = time.clock()
      printerr("Set likelihoods in %gs."%(toc-tic))
      self.hLike = self.hLike.view(ndarray_ADD)


      self.maxNoise = self.__minDif /  4 #* self.markers * self.indivs) 

      # Break symmetries by adding noise

      hetNoiseLevel = 1e-3

      if False:
         self.maxNoise = numpy.log(1.0 + hetNoiseLevel) - numpy.log(1.0 - hetNoiseLevel)
         printerr("Breaking symmetries randomly with additive +-%g noise in log likelihood space!"%(self.maxNoise) )

         state01 = self.hLike[:,:,1] # No copy, just a view
         noiseFactor = numpy.random.uniform(low = 1.0 - hetNoiseLevel, high = 1.0 + hetNoiseLevel, size = state01.shape)
         noiseFactor[state01 != self.logG_part[(2,2)][1]] = 1.0

         self.hLike[:, :, 1] += numpy.log(noiseFactor)
         self.hLike[:, :, 2] += numpy.log(2-noiseFactor)
      






   def initMessages(self, expect_IBS = 1.5, expect_IBD = 15.0):
      self.expect_IBS = expect_IBS
      self.expect_IBD = expect_IBD
      printerr("Setting phase probability tables")
      self.__set_CPT()

      #self.__setCPj(gainRate, lossRate)
      #printerr("Setting allele constraints")
      #self.__setCAj()

   def visualiseP(self, ind1, ind2, beginM, endM,fname = None):
      "Visualise the phase probabilities from beginM marker to endM marker"
      ca2pP=self.__assist.cp2pP
      ca2p=self.__assist.ca2p

      def normalize_beliefs(x):
         x = numpy.exp(-x)
         return x/x.sum(axis=0)

      len_ij = endM - beginM +1
      max_belief = ca2p[:,1:len_ij+1] + ca2pP[:,0:len_ij]
      
      norm_belief = normalize_beliefs(max_belief)

      with self.__pylab_lock:
         if fname is not None:
            printerr("Using matplotlib backend AGG.")
            import matplotlib
            matplotlib.use("AGG")
         import pylab
         pylab.figure()

         pylab.plot(self.snpCMpos[beginM:endM + 1], norm_belief[0,:],
                    self.snpCMpos[beginM:endM + 1], norm_belief[1,:],
                    self.snpCMpos[beginM:endM + 1], norm_belief[2,:],
                    self.snpCMpos[beginM:endM + 1], norm_belief[3,:],
                    self.snpCMpos[beginM:endM + 1], norm_belief[4,:])
         pylab.legend(("0-0", "1-0", "0-1", "1-1","X-X"))
         pylab.xlabel("Position (cM)")
         pylab.ylabel("Probability")
         pylab.title("Max Conditional IBD probability between individuals %d and %d\nEibs=%gcM Eibd=%gcM"%(ind1,ind2, self.expect_IBS, self.expect_IBD))

         pylab.axis([self.snpCMpos[beginM], self.snpCMpos[endM],0,1.0])
         #pylab.axis([0,10.0,1e-5,1.0])

         if fname is not None:
            pylab.savefig(fname)
         else:
            pylab.show()
  
   def visualiseCPT(self, g = None, h = None, fname = None, beginM=0):
      __snpCMpos = self.snpCMpos

      
      #self.snpCMpos = numpy.arange(len(self.snpCMpos), dtype = float )/100.0
      if g is not None and h is not None:
         self.__set_CPT(g,h)

      with self.__pylab_lock:
         if fname is not None:
            printerr("Using matplotlib backend AGG.")
            import matplotlib
            try:
               matplotlib.use("AGG")
            except UserWarning,e:
               pass

         import pylab
         pylab.figure()
         # hom het genotypes
         # step from no ibd to ibd
         legends = []
         for h0,h1 in ((0,0), (3,3) ):
            p4to0 = -self.CPT[beginM+1,4,0,h0,h1]
            p0to0 = -self.CPT[beginM+2:,0,0,h0,h1]

            pIn0 = numpy.concatenate(([p4to0],  p0to0)).cumsum()


            pIn4 = -self.CPT[beginM+1:,4,4,0,2].cumsum()
            pylab.semilogy(self.snpCMpos[beginM+1:], numpy.exp(pIn0 ))
            pylab.semilogy(self.snpCMpos[beginM+1:], numpy.exp(pIn4))
            legends.append(str((h0,h1,"IBD")))
            legends.append(str((h0,h1,"noIBD")))
         #

         #pylab.legend(("Prob. IBD","Prob. no IBD"))
         pylab.legend(legends)
         pylab.xlabel("Position (cM)")
         pylab.ylabel("Probability")
         pylab.axis([self.snpCMpos[beginM],self.snpCMpos[beginM]+10,1e-5,1.0])

         if fname is not None:
            pylab.savefig(fname)
         else:
            pylab.show()
      self.snpCMpos = __snpCMpos      
      

   def __set_CPT(self, g = None, h = None ):
      "Calculate conditional probability tables for hidden phase variables P(p_j | h0_j, h1_j, p_j-1)"

      if g is None: # Gain rate
         g = 1.0 / (4.0 * self.expect_IBS)

      if h is None: # Loss rate
         h = 1.0 / float(self.expect_IBD)
         
      rateMat5=numpy.array([[0,0,0,0,h],
                            [0,0,0,0,h],
                            [0,0,0,0,h],
                            [0,0,0,0,h],
                            [g,g,g,g,0]])


      assert (rateMat5 >= 0).all()
      
      rateMat5+=numpy.diag(-rateMat5.sum(axis=1))



      printerr(rateMat5)

      if False:
         snpDists=[0.0] + [nextPos-prevPos for prevPos,nextPos in it.izip(self.snpCMpos,
                                                                          self.snpCMpos[1:])]



         snpDists = numpy.ones(len(self.snpCMpos)) * mean_cm_dist

         printerr("Assuming equidistant markers spaced every %g cM! Length of the chromosome is %g cM"%(mean_cm_dist,mean_cm_dist * self.markers))
      else:
         snpDists = numpy.concatenate( ([0.0], numpy.diff(self.snpCMpos)))

      mean_cm_dist = self.snpCMpos[-1]/len(self.snpCMpos)

      expected_nonIBD_snps = -1.0/(rateMat5[4,4]*mean_cm_dist)
      expectedTimeToCoalescence = 50.0 / (-1.0/rateMat5[0,0])
      expectedTimeToNonIBDcoalescence = 50.0 / (-1.0/rateMat5[4,4])
      printerr("Expected length of nonIBD IBS stretches: %g cM (~ %g SNPs). Coalescence expected %g generations back."%(-1.0/rateMat5[4,4],
                                                                                                                        -1.0/(rateMat5[4,4]*mean_cm_dist),
                                                                                                                        expectedTimeToNonIBDcoalescence))
      printerr("Expected length of IBD stretches:        %g cM (~ %g SNPs). Coalescence expected %g generations back."%(-1.0/rateMat5[0,0],
                                                                                                                    -1.0/(rateMat5[0,0]*mean_cm_dist),
                                                                                                                    expectedTimeToCoalescence) )

      if mean_cm_dist < 1e-5:
         printerr("The SNPs are spaced on average %gcM from each other, which is very close. Are you absolutely sure that everything is right?"%(mean_cm_dist))

      #snpDists=numpy.ones(len(self.snpCMpos))*10
      # CPT[marker,p_j-1,p_j,hi,hi']

      # CPT[marker,p_marker-1,p_marker,hi,hi'] = P(p_marker|p_marker-1)
      self.CPT = numpy.dstack( scipy.linalg.expm(snpDist*rateMat5) for snpDist in snpDists )

      

      # TODO: NEEDS TESTING  Tue Jun 08 10:13:21 BST 2010 


      limFactor = self.CPT[4,4,:] / (1.0 - self.IBDtransProbLimit)

      limMarkerCount = (limFactor<1.0).sum()
      printerr("Limiting non IBD to IBD transition probability to %g for %g%% (%d) of markers."%(self.IBDtransProbLimit, limMarkerCount*100.0 / self.CPT.shape[2], limMarkerCount))
      self.CPT[4,4,limFactor < 1.0 ] = 1.0 - self.IBDtransProbLimit
      self.CPT[4,0:4,limFactor < 1.0] = self.IBDtransProbLimit / 4.0
      

      # END TODO

      self.CPT = numpy.array( self.CPT.transpose((2,0,1)), dtype = self.dataType )
      assert numpy.allclose(self.CPT.sum(axis=2),1.0)




      #self.zeroAF[0]=0.25
      def TESTBIT(x,b):
          return (x&(1<<b))
      
      def GETBIT(x,b):
          return (x&(1<<b))>>b
       


       
      def CAgivenGeno2(j,h0,h1,phase):
          "P(h0,h1|phase,f)"

          f = self.zeroAF[j]
          #FAKE
          #f=0.25
          if phase==4: # No shared haplotype 
              # rHW is P(h0,h1|f)

              r=rHW=1.0
              rHW*=TESTBIT(h0,0) and 1-f or f
              rHW*=TESTBIT(h0,1) and 1-f or f
              rHW*=TESTBIT(h1,0) and 1-f or f
              rHW*=TESTBIT(h1,1) and 1-f or f
              return rHW
           
          elif GETBIT(h0,GETBIT(phase,0)) == GETBIT(h1,GETBIT(phase,1)):  # Shared haplotype and concordant alleles
              r=1.0


              # r is P(h0,h1|phase,f)
              r*= TESTBIT(h0,GETBIT(phase,0)) and 1-f or f
              #
              r*= TESTBIT(h0,GETBIT(~phase,0)) and 1-f or f
              r*= TESTBIT(h1,GETBIT(~phase,1)) and 1-f or f
              return r
          else: # discordant alleles
              return 0.0

          assert(False)

     
      assert numpy.allclose(self.CPT.sum(axis=2),1.0)

      alt_CAj=numpy.fromiter( (CAgivenGeno2(j,h1,h2,p) for j,p,h1,h2 in product( xrange(self.markers),
                                                                                      xrange(5),xrange(4),
                                                                                      xrange(4))),
                                   dtype = self.dataType)
      alt_CAj.shape = (self.markers,1,5,4,4)
      #self.L_pop_table = alt_CAj[:,0,4]
      #self.L_ibd_table = alt_CAj[:,0,0]


      # Following is a bad idea about controlling the unobserved IBS
#       snpBPdists=numpy.array([0] + [nextPos-prevPos for prevPos,nextPos in it.izip(self.snpPos,
#                                                                                    self.snpPos[1:])])
#       prob_IBS_prior = numpy.exp(-numpy.array(snpBPdists)*1e-6)
#       prob_IBS_prior.shape = (-1,1,1)
#       tmp_CPT = self.CPT[:,:,:4]*numpy.tile(prob_IBS_prior,(1,5,4))
#       lastCol = 1 - tmp_CPT.sum(axis=2)
      
#       self.CPT = numpy.dstack((tmp_CPT,lastCol))


      alt_CAj = numpy.tile(alt_CAj,(1,5,1,1,1))
      assert numpy.allclose(alt_CAj.sum(axis=4).sum(axis=3),1.0)

      
      self.CPT.shape = (-1,5,5,1,1)
      self.CPT = numpy.tile(self.CPT,(1,1,1,4,4))

      
      assert numpy.allclose(self.CPT.sum(axis=2),1.0)

      assert self.CPT.shape == alt_CAj.shape


      self.firstCP2P = numpy.array([self.ibd_prob, self.ibd_prob, self.ibd_prob, self.ibd_prob, 1.0-4.0*self.ibd_prob],dtype=self.dataType)


      self.firstCP2P = numpy.array(-numpy.log(self.firstCP2P),dtype=self.dataType)
      printerr("firstcp2p : ",numpy.exp(-self.firstCP2P))


      self.CPT[0,:,0:4,:] = self.ibd_prob
      self.CPT[0,:,4,:] = 1-4*self.ibd_prob


      

      # self.CPT[j,p_j-1,p_j,hj0,hj1] = log( P(p_j|p_j-1) * P(hj0,hj1|p_j) )
      self.CPT = (self.CPT * alt_CAj)

      ## Marginalise properly 
      assert numpy.allclose( self.CPT.sum(axis=4).sum(axis=3).sum(axis=2), 1.0)

      # cptS[j,p_j-1,h0,h1] = \sum_q P(q|p_j-1) * P(hj0,hj1|q)
      cptS = self.CPT.sum(axis=2)
      cptS.shape = (-1,5,1,4,4)
      cptS = numpy.tile(cptS,(1,1,5,1,1))

      # Marginalization proper
      cptS = numpy.log(self.CPT) - numpy.log(cptS)
      cptS[numpy.isnan(cptS)] = -numpy.Inf



      # self.CPT[marker, p_j-1, p_j, hj0, hj1] = -log( P(pj | p_j-1, hj0, hj1) )
      #assert numpy.allclose(numpy.array(list(set(float(x) for x in lopulta.sum(axis=2).flat)-set([0]))),1.0)

      self.CPT = numpy.empty(cptS.shape,dtype=self.dataType).transpose((0,2,1,3,4))
      
      self.CPT[:] = -cptS


      p3ibd = numpy.exp(-self.CPT[:,4,:,0,0])
      likelyIBD = p3ibd[:,0]>p3ibd[:,4]
      printerr("Total %d sites are IBD just by IBS due to allele frequency and genetic distance to neighbours\n"%(likelyIBD.sum()))



      assert numpy.allclose(numpy.exp(-self.CPT[1:,]).sum(axis=2),1.0)

      self.CPT = numpy.ascontiguousarray(self.CPT)

      




   def loadGeneticMap(self, geneticMap):
      "Load genetic map in hapmap format"

      #       printerr("Reading genetic map %s"%(options.geneticMap))

      gmFile = open(geneticMap)
      header = gmFile.readline()
      header = header.strip().split()
      if len(header) == 4:
         gmap = numpy.loadtxt(gmFile, dtype = [ ("Chrom","|S10"), ("bpPos",numpy.int), ("rate",numpy.float), ("cmPos", numpy.float) ] )
      elif len(header) == 3:
         gmap = numpy.loadtxt(gmFile, dtype = [ ("bpPos",numpy.int), ("rate",numpy.float), ("cmPos", numpy.float) ] )
      else:
         raise Exception("Malformatted genetic map file %s. Need 3-4 columns: chromosome(optional) bp_position recomb_rate centiMorgan_position"%(geneticMap))


      numpy.diff(gmap["cmPos"])
      map_offset = gmap["rate"][0]*gmap["bpPos"][0]*1e-6
      way_past_cm = gmap["cmPos"][-1] + map_offset + \
                    gmap["rate"][-1]*(self.snpPos[-1] - gmap["bpPos"][-1])*1e-6


      gmap = gmap[ numpy.diff(gmap["cmPos"]) > 1e-5 ]
      self.snpCMpos = numpy.interp(self.snpPos,
                                   numpy.concatenate(([0], gmap["bpPos"], [ gmap["bpPos"][-1]*2 ])),
                                   numpy.concatenate(([0.0],
                                                      gmap["cmPos"] + map_offset,
                                                      [ way_past_cm ])))
                                     
      if self.snpPos.min() < gmap["bpPos"].min() or self.snpPos.max() > gmap["bpPos"].max():
         gmap_cover_prop = (min(gmap["bpPos"][-1],self.snpPos[-1])-max(gmap["bpPos"][0],self.snpPos[0]))*1.0/(self.snpPos[-1]-self.snpPos[0])
         printerr("Extrapolating genetic map from region %d-%dbp (%g cM) to %d-%dbp (%g cM)"%(gmap["bpPos"].min(), gmap["bpPos"].max(),
                                                                                              gmap["cmPos"].max()-gmap["cmPos"].min(),
                                                                                              self.snpPos.min(), self.snpPos.max(),
                                                                                              self.snpCMpos[-1] - self.snpCMpos[0]))
         printerr("Genetic map covers %g%% of the marker span."%(100.0*gmap_cover_prop))
         
      printerr("Length of chromosome: %d bp, %g cM"%(self.snpPos[-1] - self.snpPos[0]+1,
                                                     self.snpCMpos[-1] - self.snpCMpos[0]))
      assert self.snpPos[-1] - self.snpPos[0]+1 > 0, "Non positive physical chromosome length"
      assert self.snpCMpos[-1]- self.snpCMpos[0] > 0, "Non positive genetic chromosome length"
      

      

   ibd_regions_dtype = { 'names' : ( "ind1", "ind2",  "score",  "beginM", "endM", "beginBP",  "endBP" ),
                   'formats' : (numpy.int32, numpy.int32, numpy.float, numpy.int32, numpy.int32, numpy.int32, numpy.int32) }

   ibd_regions_dtype_int = { 'names' : ( "ind1", "ind2",  "score",  "beginM", "endM", "beginBP",  "endBP" ),
                       'formats' : (numpy.int32, numpy.int32, numpy.int32, numpy.int32, numpy.int32, numpy.int32, numpy.int32) }

   def writeIBD(self,foutName):
      self.ibd_regionsSegmentsToIBD()
      if self.ibd_regions is not None and len(self.ibd_regions) > 0:
         numpy.savetxt(foutName,
                       self.ibd_regions,
                       fmt = "%d\t%d\t%g\t%d\t%d\t%d\t%d")
      #open(foutName,"w").writelines("%d\t%d\t%g\t%d\t%d\t%d\t%d\n" % (
      #   x[0], x[1], self.snpCMpos[x[3]] - self.snpCMpos[x[2]],
      #   x[2],x[3], self.snpPos[x[2]], self.snpPos[x[3]] ) for x in self.ibdSegments)


   def __waitIBD(self):
      "Wait for IBD file to load. Must be called before accessing ibd_regions"
      
      if self.ibd_regionsLoader is not None:
         printerr("Waiting IBD loading to finish!")
         self.ibd_regionsLoader.join()
         self.ibd_regionsLoader = None
         printerr("IBD loading finished!")

         
   def __loadIBDthread(self,fname):
      self.ibd_regions = numpy.loadtxt( fname, dtype = self.ibd_regions_dtype )
      assert ( self.ibd_regions["endM"] < self.markers ).all()

      
   def loadIBD(self, ibdFileName):
      "Load precalculated IBD file (in separate thread)"

      
      #self.ibd_regionsLoader = threading.Thread(target = self.__loadIBDthread,
#                                          name = "IBD load",
#                                          args = (ibdFileName,))

      #self.ibd_regionsLoader.start()
      
      self.ibd_regions = numpy.loadtxt( ibdFileName, dtype = self.ibd_regions_dtype )


      assert ( self.ibd_regions["endM"] < self.markers ).all()


   def haveIBD(self):
      return self.ibd_regions is not None

   def ibd_regionsSegmentsToIBD(self):
      "Convert the parsimonious 4 value ibd segment information to IBD file format"
      if len(self.ibdSegments) > 0:  # DANGER!!! over writes the input ibd data
         printerr("Setting ibd_regions")
         printerr("Starting:",time.asctime())
#          self.ibd_regions = numpy.array( [(x[0], x[1], self.snpCMpos[x[3]] - self.snpCMpos[x[2]],
#                                      x[2],x[3], self.snpPos[x[2]], self.snpPos[x[3]] ) for x in self.ibdSegments],
#                                    dtype = self.ibd_regions_dtype)
         self.ibd_regions = numpy.fromiter( ((x[0], x[1], self.snpCMpos[x[3]] - self.snpCMpos[x[2]],
                                        x[2],x[3], self.snpPos[x[2]], self.snpPos[x[3]] ) for x in self.ibdSegments),
                                      dtype = self.ibd_regions_dtype,
                                      count = len(self.ibdSegments))
         printerr("Done:",time.asctime())
      else:
         # Should do something intelligent, e.g. warn that couldn't find any IBD segments.
         pass
      
   def ibdCoverCounts(self,scoreLimit = 0.0 ):
      "Calculate the coverage of the called IBD relations for each site x haplotype"
      ibdCoverCounts=numpy.zeros( (self.markers, 2*self.indivs),
                                 dtype=numpy.uint32)

      self.ibd_regionsSegmentsToIBD()
      self.__waitIBD()

      for ind1,ind2,score,beginM,endM,beginBP,endBP in self.ibd_regions:
        if score >= scoreLimit:
            ibdCoverCounts[beginM:endM+1,ind1] += 1
            ibdCoverCounts[beginM:endM+1,ind2] += 1
      return ibdCoverCounts
   def ibdCoverCounts2(self,scoreLimit = 0.0 ):
      "Calculate the coverage of the called IBD relations for each site x haplotype"
      ibdCoverCounts=numpy.zeros( (self.markers, self.indivs,2),
                                 dtype=numpy.uint32)

      self.ibd_regionsSegmentsToIBD()
      self.__waitIBD()

      for ind1,ind2,score,beginM,endM,beginBP,endBP in self.ibd_regions:
        if score >= scoreLimit:
            ibdCoverCounts[beginM:endM+1,ind1/2,ind1%2] += 1
            ibdCoverCounts[beginM:endM+1,ind2/2,ind2%2] += 1
      return ibdCoverCounts



   def computeGenos(self):
      """Compute the genotypes from self.haplos. This is needed e.g. for the initial likelihood
      scan over genotypes"""

      # hom 0, hom 1, het 2, missing 3

      if not hasattr(self,"geno"):
         self.geno = self.haplos.sum(axis=2)
         # Phased hets
         self.geno[self.geno == 1 ] = 4
         # One missing
         self.geno[ (self.haplos==3).any(axis=2) ] = 6

         self.geno = self.geno.transpose()
         self.geno /= 2
         #self.geno = numpy.cast[numpy.int8](self.geno)
         self.geno = numpy.ascontiguousarray(self.geno, dtype=numpy.int8)


   def computeLogLikeTable(self):
      """Compute the log likelihood table to be used in preprocessing"""
      # LLtable[marker,genotype1, genotype2]
      # genotypes hom 0, hom 1, het 2

      #printerr("Calculating log likelihoods from the IBD process. VERY PRELIMINARY. Doesn't work very well. ")
      # Non IBD likelihoods with complete data
      G00,Ghet,G11 = 0,2,1

      L_pop_table = numpy.zeros( (self.markers,4,4) )
      L_pop_table[:,0:3,0:3] = self.CPT[numpy.ix_(range(self.markers),[4],[4],[0,3,1],[0,3,1])].squeeze()

      
      
      # IBD likelihoods with complete data
      L_ibd_table = numpy.zeros( (self.markers,4,4) )
      
      
      L_ibd_table[:,G00, G00 ] = self.CPT[:,0,0,0,0]
      L_ibd_table[:,G00, Ghet] = self.CPT[:,0,0,0,2]
      L_ibd_table[:,G00, G11 ] = numpy.inf
      L_ibd_table[:,Ghet,G00 ] = self.CPT[:,0,0,2,0]
      L_ibd_table[:,Ghet,Ghet] = -numpy.log( self.zeroAF * numpy.exp(-self.CPT[:,0,0,1,1]) + (1-self.zeroAF) * numpy.exp(-self.CPT[:,0,0,2,2]))
      L_ibd_table[:,Ghet,G11 ] = self.CPT[:,0,0,1,3]
      L_ibd_table[:,G11, G00 ] = numpy.inf
      L_ibd_table[:,G11, Ghet] = self.CPT[:,0,0,3,1]
      L_ibd_table[:,G11, G11 ] = self.CPT[:,0,0,3,3]



      L_ibd_table[numpy.isinf(L_ibd_table)]=-numpy.log(self.errP)

      self.LLtable = L_pop_table - L_ibd_table
      
      self.LLtable = numpy.cast[self.dataType](self.LLtable)

   def computeLogLikeTable_directLogLike(self):
      """Compute the log likelihood table to be used in preprocessing"""
      # LLtable[marker,genotype1, genotype2]
      # genotypes hom 0, hom 1, het 2

      raise "Don't come here"
      # Non IBD likelihoods with complete data
      G00,Ghet,G11 = 0,2,1

      L_pop_table = numpy.zeros( (self.markers,4,4) )

      L_pop_table[:,G00 ,Ghet] = 2 * self.zeroAF**3 * (1.0-self.zeroAF)
      L_pop_table[:,Ghet,G11 ] = 2 * self.zeroAF * (1.0-self.zeroAF)**3
      L_pop_table[:,G00 ,G00 ] = self.zeroAF**4
      L_pop_table[:,G11 ,Ghet] = 2 * self.zeroAF * (1.0-self.zeroAF)**3
      L_pop_table[:,Ghet,Ghet] = 4 * self.zeroAF**2 * (1.0-self.zeroAF)**2
      L_pop_table[:,G00 ,G11 ] = self.zeroAF**2 * (1.0-self.zeroAF)**2
      L_pop_table[:,G11 ,G00 ] = L_pop_table[:,G00,G11]
      L_pop_table[:,G11 ,G11 ] = (1.0-self.zeroAF)**4
      L_pop_table[:,Ghet,G00 ] =  2 * self.zeroAF **3 * (1.0-self.zeroAF)

      L_pop_table *= (1.0 - self.errP)
      L_pop_table += self.errP/9.0
      
      assert numpy.allclose(L_pop_table[:,:3,:3].sum(axis=1).sum(axis=1),1.0) 


      # IBD likelihoods with complete data
      L_ibd_table = numpy.zeros( (self.markers,4,4) )


      L_ibd_table[:,G00, G00 ] = self.zeroAF**3
      L_ibd_table[:,G00, Ghet] =  self.zeroAF**2 * (1.0-self.zeroAF)
      L_ibd_table[:,G00, G11 ] = 0.0
      L_ibd_table[:,Ghet,G00 ] =  self.zeroAF **2 * (1.0-self.zeroAF)
      L_ibd_table[:,Ghet,Ghet] =  self.zeroAF**2 * (1.0-self.zeroAF)  + self.zeroAF * ((1.0-self.zeroAF) **2)
      L_ibd_table[:,Ghet,G11 ] =  self.zeroAF * (1.0-self.zeroAF)**2
      L_ibd_table[:,G11, G00 ] = 0.0
      L_ibd_table[:,G11, Ghet] =  self.zeroAF * (1.0-self.zeroAF)**2
      L_ibd_table[:,G11, G11 ] = (1.0-self.zeroAF)**3


      L_ibd_table *= (1.0 - self.errP)
      L_ibd_table += self.errP/9.0

      assert numpy.allclose(L_ibd_table[:,:3,:3].sum(axis=1).sum(axis=1),1.0) 


      self.LLtable = numpy.log(L_ibd_table) - numpy.log(L_pop_table)
      del(L_ibd_table)
      del(L_pop_table)
      self.LLtable = numpy.cast[self.dataType](self.LLtable)
      

   def computeOneIBDnoUpdate_LLscan(self,indPairs):
      """Calculate putative IBD segments with a log likelihood approach. The likelihoods are
      taken from the HMM transition/emission table and depend on the genetic distances
      between the markers."""
      
      


      
      assert self.geno.flags.c_contiguous, "Genotype array must be C-contiguous"
      assert self.geno.strides[1] == 1, "Geno array must be indexable by first coordinate."
      assert self.LLtable.shape == (self.markers,4,4), "Badly shaped LL table"
      assert self.LLtable.flags.c_contiguous, "LLtable array must be C-contiguous"
      peakThreshold = 4.0;
      dipThreshold = 12.0
      debug = True

      ibd_regions = self.c_ext.LLscan(indPairs,self.geno,self.LLtable,peakThreshold,dipThreshold)

      return ibd_regions

   def loadAlleleFrequencies(self,freqFile):
      "Load allele frequency information from the named file"
      posFreq = numpy.loadtxt(freqFile, dtype = [('pos', int), ('freq', float), ("A0","|S1"), ("A1","|S1")] )
      assert len(posFreq) == self.markers, "Frequency file format error. There has to be frequencies for exactly the given markers"
      assert ((posFreq["pos"] - self.snpPos)==0).all(), "Frequency file format error. There has to be frequencies for exactly the given markers"

      load_A = posFreq[["A0","A1"]].view("|S1").reshape((-1,2))
      assert (self.alleles == load_A).all(), "Some allele mappings do not match between the genotype data and the allele frequency file"

      self.zeroAF = posFreq["freq"].copy()

   def writeAlleleFrequencies(self,freqFile):
      "Write allele frequency information to the named file"
      frq = numpy.fromiter(it.izip(self.snpPos, self.zeroAF, self.alleles[:,0],self.alleles[:,1]),dtype= [('pos', int), ('freq', float), ("A0","|S1"), ("A1","|S1")] )
      numpy.savetxt(freqFile,frq,fmt="%d\t%g\t%s\t%s")
      pass
   
   def computeOneIBDnoUpdate(self,indPairs):
      "Compute one forward message passing over pair of individuals on restricted region. Dont update likelihoods."
      printerr('Running child process with id: ', os.getpid())
      #       cp2pN=self.cp2pN
      #       ca2p=self.ca2p
      #       p2ca=self.p2ca
      #       p2cpN=self.p2cpN
      #       caj=self.CAj
      #       cpj=self.CPj
      #       firstCP2P=self.firstCP2P
      #       bt=self.backtrack

      self.____assist_arrays()


      ca2p=self.__assist.ca2p
      CPT=self.CPT

      firstCP2P=self.firstCP2P
      bt=self.__assist.backtrack
      
      #printerr("Running __computeOneIBDnoUpdate")

      
      #TODO: Move more stuff to C
         
      ibdRegions=[]



            

      # This should make a view with ndarray and a copy if using MPI
      hLike = self.hLike[:,:,:]
      hLike = hLike.view(numpy.ndarray)
      
      endM=self.markers-1

      # Compile:  Make a dummy pass with no real input to compile the code if necessary.
#       __indPairs=indPairs
#       indPairs=[]
#       self.__lock_inlining()
#       weave.inline(codeNormBT%{"cType":self.cDataType}, \
#                    ["bt",'firstCP2P',"hLike","indPairs","ca2p","endM","CPT","ibdRegions"], \
#                    type_converters=converters.blitz,compiler="gcc",verbose=2,force=False,
#                    extra_compile_args=["-O3"],support_code="#include<time.h>\n#include<cstdlib>")
#       indPairs=__indPairs
#       self.__unlock_inlining()
      # end compile
      
      indPairs=list(indPairs)
      self.c_ext.scan_IBD_hmm(bt,firstCP2P,hLike,indPairs,ca2p,endM,CPT,ibdRegions)
      
#       weave.inline(codeNormBT%{"cType":self.cDataType}, \
#                    ["bt",'firstCP2P',"hLike","indPairs","ca2p","endM","CPT","ibdRegions"], \
#                    type_converters=converters.blitz,compiler="gcc",verbose=2,force=False,
#                    extra_compile_args=["-O3"],support_code="#include<time.h>\n#include<cstdlib>")
# #                   type_converters=converters.blitz,compiler="gcc",verbose=2,force=self.firstPhase,extra_compile_args=["-msse2","-ftree-vectorize","-ftree-vectorizer-verbose=0","-g"])
      self.firstPhase=False

      ibdRegions = numpy.array(ibdRegions,dtype=int)
      return ibdRegions



   def set_workers(self, poolSize):
      "Set workers for parallel processing"
      self.poolSize = poolSize
      self.__ioIBD_sync = ThreadBarrier(self.poolSize)
      if self.poolSize > 1:
         global printerr
         printerr = ThreadTagOutput(printerr)


   def pmap(self, func, iterable):
      "Parallel map with shared state"
      # Handythread from numpy cookbook
      
      r = handythread.parallel_map(func, iterable, threads = self.poolSize)
      return r
   


   def __lock_inlining(self):
      printerr("Serializing")
      # Thread lock
      self.__inlining_lock.acquire()

      # MPI lock
      if self.worldSize > 1:
         self.__inlineLock.Lock( MPI.LOCK_EXCLUSIVE,  0)
         time.sleep(2*self.myRank)
      

      printerr("Got lock for inline compilation")



   def __unlock_inlining(self):
      # MPI lock
      if self.worldSize > 1:
         self.__inlineLock.Unlock(0)

      # Thread lock
      self.__inlining_lock.release()
      printerr("Released the inline lock")

            
      

   def __computeOneIBD(self,ind1,ind2,beginM,endM,ca2h1,ca2h2,doUpdate=True, serializedPass = False):
      "Compute one forward-backward message passing over pair of individuals on restricted region. Return the mean squared error for messages to the haplotype nodes."
      #TODO: Improve speed. Takes 15% of time, excluding the C part
      self.____assist_arrays()


      #ca2pN=self.__assist.cp2pN
      ca2pP=self.__assist.cp2pP
      ca2p=self.__assist.ca2p
      #p2ca=self.__assist.p2ca
      #p2cpN=self.__assist.p2cpN
      #p2cpP=self.__assist.p2cpP
      CPT=self.CPT
      #cpj=self.CPj
      dampF=self.dampF
      firstCP2P=self.firstCP2P
      #firstCP2P = numpy.array(-numpy.log([0.0, 0.0, 0.0, 0.0, 1.0]),dtype=self.dataType)
      bt=self.__assist.backtrack

      #firstCP2P=numpy.zeros(5)
      #print "firstCP2P:",firstCP2P
      #print endM,type(endM)

      #TODO: Move more stuff to C
      hLike1 = self.hLike[ind1/2,beginM:endM+1,:].view(numpy.ndarray)
      hLike2 = self.hLike[ind2/2,beginM:endM+1,:].view(numpy.ndarray)

      if doUpdate:
         h12ca = hLike1 - ca2h1[:endM-beginM+1,:]
         h22ca = hLike2 - ca2h2[:endM-beginM+1,:]

         ca2h1_prior = ca2h1.copy()
         ca2h2_prior = ca2h2.copy()

      else:
         h12ca=hLike1
         h22ca=hLike2

      # Get rid of bliz converters. They take 67% of runtime.
      codeNormBT="""
#line 1664 "longRangePhaseMP.py"

            int ij,j;
            int h0,h1,p,p1;
            %(cdType)s tot_min_ca2p;
            int num_markers = ca2p_array->dimensions[1]-1;
            %(cdType)s old_ca2h[2][4];
            double sum_sq_err = 0.0;

            Py_BEGIN_ALLOW_THREADS;
            
            // Forward


            //std::cout << "nd: " << bt_array->nd<<std::endl;
            //std::cout << "dim: " << bt_array->dimensions[0] <<","<< bt_array->dimensions[1] <<std::endl;

            // First step

            ij = 0;
            for(p=0;p<5;p++) {
               if(beginM == 0 ) {
                   //ca2p(p,0) = firstCP2P(p); //0.0;
                   ca2p(p,0) = 0.0;
               } else {
                  if(p == 4) {
                     ca2p(p,0) = 0.0;
                  } else {
                     ca2p(p,0) = -log(0.0);
                  }
               }
               //std::cerr<<"ca2p("<<p<<", "<<ij<<") = "<<ca2p(p, ij)<<std::endl;
               //ca2p(p,0) = 0.0;
              
               bt(p,beginM) = -1;

            }




            // Rest of the steps
            for(ij=0,j = beginM ; j <= endM;j++,ij++) {
                 tot_min_ca2p = 1e308;

                 // ca_j -> p_j
                 // TODO: Don't minimize exhaustively, but try to be smart with structure of CPT etc.

                 for(p=0;p<5;p++) {
                    %(cdType)s  ca2p_V = 1e307, min_ca2p = 1e308;
                    
                    for(int p0=0; p0<5; p0++) {
                       const %(cdType)s ca2p0_V =  ca2p(p0, ij);
                       //if(ca2p(p0,ij)<0 ) { std::cerr<<"ca2p("<<p0<<", "<<ij<<") = "<<ca2p(p0, ij)<<std::endl;exit(1);}
                       for(h0=0;h0<4;h0++) {
                         const %(cdType)s h12ca_V =  h12ca(ij,h0);
                         for(h1=0;h1<4;h1++) {
                            //std::cerr<<ca2p_V<<"->";
                            ca2p_V = std::min(ca2p_V, CPT(j, p0 ,p,h0,h1) + h12ca_V + h22ca(ij,h1) + ca2p0_V );
                            //std::cerr<<ca2p_V<<"="<<CPT(j, p0 ,p,h0,h1) <<"+"<< h12ca_V <<"+"<< h22ca(ij,h1) <<"+"<< ca2p0_V<<std::endl;
                         }
                       }
                       if(min_ca2p > ca2p_V ) {
                          ca2p(p,ij+1) = ca2p_V;
                          //std::cerr<<"ca2p("<<p0<<", "<<ij+1<<") <- "<<ca2p(p0, ij+1)<<" = "<<ca2p_V<<std::endl;
                          // bt(p,j+1) == p0  iff the most probable state at marker j is p0, given that state at marker j+1 is p
                          // bt(p,j) == p0  iff the most probable state at marker j-1 is p0, given that state at marker j is p
                          bt(p,j+1)=p0;
                          min_ca2p = ca2p_V;
                       }

                    }
                    tot_min_ca2p = std::min(tot_min_ca2p,min_ca2p);
                 }
                 // Normalize
                 for(p=0;p<5;p++) {
                    ca2p(p,ij+1) -= tot_min_ca2p;
                 }

                 //std::cout<<ij<<": tot_min_ca2p="<< tot_min_ca2p<<std::endl;
            }
            """
      codeNormBkwd="""
#line 1591 "longRangePhaseMP.py"



            // Backward
            for(p=0;p<5;p++) { //Initialization
               if(endM == (num_markers - 1) ) { // No information past the end of chromosome
                  //ca2pP(p,endM-beginM+1)=(%(cdType)s)firstCP2P(p); // THIS IS UNDER TESTING #  Fri Sep 03 16:35:32 BST 2010 
                  ca2pP(p,endM-beginM+1)=0.0;
                  //std::cerr<<"End of chromosome@"<<endM<<": ca2pP <- " << ca2pP(p,endM-beginM+1) <<std::endl;
               } else { // Within the chromosome, IBD region ends in non IBD state
                  if(p == 4) {
                     ca2pP(p,endM-beginM+1) = 0.0;
                  } else {
                     ca2pP(p,endM-beginM+1) = (%(cdType)s)-log(0.0);
                  }
                  //std::cerr<<"ca2pP <- " << ca2pP(p,endM-beginM+1) <<std::endl;
               } 
            }
            for(ij=endM-beginM,j=endM; j>=beginM; j--,ij--) {
                // Backwards from P(p_j| h0,h1,p_j-1) (to h0, h1, and p_j-1]
                %(cdType)s min_ca2pP[5] = {1e308,1e308,1e308,1e308,1e308};
                %(cdType)s tmp_ca2h1[4] = {1e308,1e308,1e308,1e308};
                %(cdType)s tmp_ca2h2[4] = {1e308,1e308,1e308,1e308};

                for(p=0;p<5;p++) {
                    const %(cdType)s ca2pP_V = ca2pP(p,ij+1);
                    for(p1=0;p1<5;p1++) {
                       const %(cdType)s ca2p_V = ca2p(p1,ij+1);
                       for(h0=0;h0<4;h0++) {
                           const %(cdType)s h12ca_V = h12ca(ij,h0);
                           for(h1=0;h1<4;h1++) {
                               const %(cdType)s CPT_V =  CPT(j,p1,p,h0,h1); //Speedup
                               const %(cdType)s h22ca_V = h22ca(ij,h1);
                               
                               min_ca2pP[p1] = std::min(min_ca2pP[p1], CPT_V +          ca2pP_V + h12ca_V + h22ca_V );
                               tmp_ca2h1[h0] = std::min(tmp_ca2h1[h0], CPT_V + ca2p_V + ca2pP_V +           h22ca_V );
                               tmp_ca2h2[h1] = std::min(tmp_ca2h2[h1], CPT_V + ca2p_V + ca2pP_V + h12ca_V );
                           }
                       }
                   }
                }
                // Normalize and set:
                %(cdType)s minVal=1e308;
                for(p=0;p<5;p++) {
                     minVal=std::min(minVal, min_ca2pP[p]);
                }
                //std::cout<<ij<<": minVal="<< minVal<<std::endl;
                for(p=0;p<5;p++) {
                     ca2pP(p,ij) = min_ca2pP[p] - minVal;
                }

                

                minVal = 1e308;
                %(cdType)s minVal2 = 1e308; 
                for(h0=0;h0<4;h0++) {
                    old_ca2h[0][h0] = ca2h1(ij,h0);
                    old_ca2h[1][h0] = ca2h2(ij,h0);

                    ca2h1(ij,h0) = dampF * ca2h1(ij,h0) + (1-dampF)*tmp_ca2h1[h0];
                    ca2h2(ij,h0) = dampF * ca2h2(ij,h0) + (1-dampF)*tmp_ca2h2[h0];


                    minVal=std::min(minVal, ca2h1(ij,h0));
                    minVal2=std::min(minVal2, ca2h2(ij,h0));
                }
                for(h0=0;h0<4;h0++) {
                    ca2h1(ij,h0) -= minVal;
                    ca2h2(ij,h0) -= minVal2;

                    // Count error
                    double err_value ;
                    err_value = (old_ca2h[0][h0] - ca2h1(ij,h0));
                    sum_sq_err += err_value * err_value;
                    err_value = (old_ca2h[1][h0] - ca2h2(ij,h0));
                    sum_sq_err += err_value * err_value;
                }


            } 
            Py_END_ALLOW_THREADS;
            
            return_val = sum_sq_err / ( endM - beginM + 1) ;


            """

      # Release GIL (Global Interpreter Lock) 
      codeNormBT=codeNormBT+codeNormBkwd
            



      
            

      if serializedPass:
         self.__lock_inlining()
      #printerr("X")   

      # Nasty formatting 
      endM = int(endM)
      beginM = int(beginM)
      ind1 = int(ind1)
      ind2 = int(ind2)

      sum_sqr_E = weave.inline(codeNormBT%({"cdType":self.cDataType}), ["bt","ca2h1","ca2h2","ca2pP","ca2p","beginM","endM","CPT","h12ca","h22ca","dampF","firstCP2P"], \
                               type_converters=converters.blitz,compiler="gcc",verbose=9,force=False,
                               #                   extra_compile_args=["-msse2","-ftree-vectorize","-ftree-vectorizer-verbose=0","-g"])
                               extra_compile_args=["-ftree-vectorize","-ftree-vectorizer-verbose=0","-O3"])

      assert numpy.isfinite(ca2h1).all()
      assert numpy.isfinite(ca2h2).all()
      # should be proportional to
      if False: #True:
         def checkit(tmp_j=endM):
            tmp_ij=tmp_j-beginM
            ca2p_tmp = ca2p[:,tmp_ij]
            ca2p_tmp.shape=(5,1,1,1)


            apu = (numpy.tile(  numpy.tile(h12ca[tmp_ij,:],(4,1))+numpy.tile(h12ca[tmp_ij,:],(4,1)).transpose()      ,(5,5,1,1))+CPT[tmp_j,] +numpy.tile(ca2p_tmp,(1,5,4,4)) ).min(axis=3).min(axis=2).min(axis=0)

            #print numpy.exp(-firstCP2P)
            print(self.hLike[ind1/2,tmp_j],self.hLike[ind2/2,tmp_j])
            print ca2p[:,tmp_j+1],apu-apu.min()
            print ind1,ind2,bt[4,beginM+1:endM+2].min()

            
         checkit()



      #if bt[4,beginM+1:endM+2].min()<4:
      #   pass
      if serializedPass:
         self.__unlock_inlining()
         printerr("Returned from locking")
         

      self.firstPhase=False

      if False and ( ind1,ind2) in ((118,156),(0,52) ):
         def normalize_beliefs(x):
            x = numpy.exp(-x)
            return x/x.sum(axis=0)
         
         #print ind1,ind2,beginM,endM
         #pProbs=(self.ca2p+self.cp2pN+self.cp2pP[:,:markers])
         pCalls=list(self.callCurStates(beginM,endM))#pProbs.argmin(axis=0)
         assert(len(pCalls)==self.markers)
#          printerr(list(runlength_enc(pCalls[:5322])))
#          printerr(list(runlength_enc(pCalls[5322:5611])))
#          printerr(list(runlength_enc(pCalls[5611:7071])))
#          printerr(list(runlength_enc(pCalls[7071:7504])))
#          printerr(list(runlength_enc(pCalls[7504:])))
         printerr("calls:",list((ind1+h1,ind2+h2,stopM-startM+1,startM,stopM) \
                       for  h1,h2,startM,stopM in self.callCurIBD(beginM,endM)))

         rle=list(runlength_enc(pCalls))
         printerr(rle)
         genos =numpy.vstack((self.haplos[:endM,ind1/2,0],self.haplos[:endM,ind2/2,0]) ).transpose()
         IBS1p = numpy.logical_or((genos.max(axis=1)==2), (genos[:,1]==genos[:,0]))
         #self.visualiseCPT(fname="frombeginM.png",beginM = beginM)
         try:
            self.__iter_counter[(ind1,ind2,beginM,endM)] = self.__iter_counter.get((ind1,ind2,beginM,endM),-1) + 1
         except AttributeError,e:
            self.__iter_counter = dict()
            self.__iter_counter[(ind1,ind2,beginM,endM)] = 0

            
            
         self.visualiseP(ind1,ind2,beginM,endM,fname = "ind%d-%d-marker-%d-%d-i%d.png"%(ind1,ind2,beginM,endM,self.__iter_counter[(ind1,ind2,beginM,endM)]))



      #if endM >= 7504 and "2" in (self.fad[endM][1][ind1], self.fad[endM][1][ind2]):
      # Testing ends

      meanSqrD=0.0
      if doUpdate:

         # TODO: improve this. Bliz might help.
         #meanSqrD=(numpy.abs(ca2h1_prior-ca2h1)+numpy.abs(ca2h2_prior-ca2h2)).sum()/(2.0*4*(endM-beginM+1))
         #meanSqrD=(numpy.abs(ca2h1_prior-ca2h1)+numpy.abs(ca2h2_prior-ca2h2)).max()
         #meanSqrD=(((ca2h1_prior-ca2h1)*(ca2h1_prior-ca2h1)).sum() +((ca2h2_prior-ca2h2)*(ca2h2_prior-ca2h2)).sum() ) / (2*4 * (endM-beginM+1))
         #print meanSqrD,sum_sqr_E
         #h12ca += ca2h1
         #h22ca += ca2h2

         #ca2h1_delta = h12ca - hLike1
         #ca2h2_delta = h22ca - hLike2
         ca2h1_delta = ca2h1 - ca2h1_prior
         ca2h2_delta = ca2h2 - ca2h2_prior
         
         #self.hLike[ind1/2,beginM:endM+1,:]+=ca2h1[:,:]
         #self.hLike[ind2/2,beginM:endM+1,:]+=ca2h2[:,:]
            
         #printerr("ADDING")
         self.hLike.ADD( (ind1/2, slice(beginM, endM+1), slice(None) ), ca2h1_delta[:endM-beginM+1])
         self.hLike.ADD( (ind2/2, slice(beginM, endM+1), slice(None) ), ca2h2_delta[:endM-beginM+1])
         #printerr("ADDED")

         #if self.myRank == 2:
         #   hLike1_new =  self.hLike[ind1/2,beginM:endM+1,:]
         #   diff = hLike1_new - (hLike1 + ca2h1_delta)

            

      return sum_sqr_E  #meanSqrD

   def callCurStates(self,beginM=0,endM=None):
      "Very, very silly way of calling MAP phase states. Only used for regression testing.!"
      if endM is None:
         endM=self.markers-1

      
      firstP=4

      segLast=endM-beginM
      #pProbs=(self.__assist.ca2p[:,segLast]+self.__assist.cp2pN[:,segLast]+self.CPj[endM,:,4])
      pProbs = self.__assist.ca2p[:,segLast] + self.firstCP2P
      try:
         self.cCount+=1
      except AttributeError :
         self.cCount=1

      #pProbs=(self.ca2p[:,endM]+self.cp2pN[:,endM]+self.cp2pP[:,endM])

      if pProbs.max()==0.0:
         firstP=4
      else:
         firstP=int(pProbs.argmin())
      if firstP!=4:
         try:
            self.fCount+=1
         except AttributeError :
            self.fCount=1
         #sys.stderr.write("Need to test this!! Should usually give 4, possibly sometimes something else! Now %d. (%dth case out of %d)\n"%(firstP,self.fCount,self.cCount))



      def _backt(i,p=4):
         yield p
         p=self.__assist.backtrack[p,i]
         while p>=0:
            yield p               
            i-=1
            p=self.__assist.backtrack[p,i]
         assert(p==-1)


      rPcalls=list(_backt(endM,firstP))

      assert len(rPcalls)==(endM-beginM+1)
      pCalls=reversed(rPcalls)

      if rPcalls[0]!=4 and rPcalls[1]==4: # Only one non 4 state:
         printerr("Nasty one marker IBD segment at %d"%(beginM))


      return it.chain(it.repeat(4,beginM),pCalls,it.repeat(4,self.markers-1-endM))


   #TODO: Make this faster. Seems to take 15-20% of time
   def callCurIBD(self,beginM=0,endM=None):
      "Slightly faster way of computing IBD regions!"
      
      if endM is None:
         endM=self.markers-1


      firstP=4
      segLast=endM-beginM
      # Probabilities of the last state
      #pProbs=(self.__assist.ca2p[:,segLast]+self.__assist.cp2pN[:,segLast]+self.CPj[endM,:,4])
      pProbs = self.__assist.ca2p[:,segLast+1] #+ self.__assist.cp2pP[:,segLast]
      try:
         self.cCount+=1
      except AttributeError :
         self.cCount=1


      if pProbs.max()==0.0:
         firstP=4
      else:
         firstP=int(pProbs.argmin())
      if firstP!=4:
         try:
            self.fCount+=1
         except AttributeError :
            self.fCount=1
         #sys.stderr.write("Need to test this!! Should usually give 4, possibly sometimes something else! Now %d. (%dth case out of %d)\n"%(firstP,self.fCount,self.cCount))



      h1=[0,1,0,1]
      h2=[0,0,1,1]
      #h1={0:0, 1:1, 2:0, 3:1}
      #h2={0:0, 1:0, 2:1, 3:1}
      r=[]
      i=endM
      p=firstP
      #print pProbs,"=>",firstP
      while i>beginM and p>=0:
         endM=i
         curState=p
         
         while curState==p:
            p=self.__assist.backtrack[p,i]
            i-=1
         if curState<4:
            r.append((h1[curState],h2[curState],i+1,endM))
            
      return r

  
   def processAllocedIBD(self,allocedIBD,MAPestimate=True):
      "Run one iteration of message passing over messages, preallocated in allocedIBD"
      startTime = time.time()
      totMeanSqrE = self.c_ext.processAllocedIBD(allocedIBD,self.hLike, self.CPT,self.dampF,self.firstCP2P,MAPestimate)
      _unreach =gc.collect()
      if _unreach > 0: printerr("Unreachable objects: %d"%(_unreach))
      doneTime = time.time()
      printerr("Repeat took %g minutes"%((doneTime-startTime)/60.0))
      printerr("#secs #segments #sites %d %d %d"%((doneTime-startTime), len(allocedIBD), (allocedIBD["lastMarkerFilled"] - allocedIBD["beginMarker"]).sum()))

      return totMeanSqrE

   def iterateOverIBD(self,allocedIBD,iterations,intermedFAD=None,intermedIBD=None,use_max_product=True):
      "Choose the correct method to do the iteration proper"
      goldStandard = False

      #allocedIBD = allocedIBD[allocedIBD["beginMarker"]<10]
      if use_max_product:
         self.iterateOverIBDmax(allocedIBD,iterations,intermedFAD,intermedIBD,goldStandard)
      else:
         self.iterateOverIBDsum(allocedIBD,iterations,intermedFAD,intermedIBD)


      pass

   def iterateOverIBDsum(self,allocedIBD,iterations,intermedFAD=None,intermedIBD=None):
      "Do the phasing proper, with the sum-product algorithm, trying to find the phase marginals"


      raise "Don't come here"
      self.hLike = self.normalizedLike()
      #allocedIBD = allocedIBD[allocedIBD['ind1firstHaplo'] == 0]
      self.CPT = numpy.exp(-self.CPT)

      for i in allocedIBD:
         i["p2h1"][:]=1.0
         i["p2h2"][:]=1.0
         
      assert numpy.isfinite(self.CPT).all()
      assert numpy.isfinite(self.hLike).all()
      prevMeanSqrE = 1.0
      repeatCount=it.count(0)
      repeat=repeatCount.next()
      
      while  repeat<iterations:
         start_repeat_time=time.time()
         printerr("Repeat %d"%(repeat))



         totMeanSqrE = self.processAllocedIBD(allocedIBD,MAPestimate=False)
         totMeanSqrE /= len(allocedIBD)
         
         if intermedFAD is not None:
            try:
               foutName=intermedFAD%(repeat)
            except TypeError:
               foutName=intermedFAD
            self.writeFAD(foutName)

         if isinstance(intermedIBD,"".__class__):
            try:
               foutName=intermedIBD%(repeat)
            except TypeError:
               foutName=intermedIBD
            self.writeIBD(foutName)

         errImpro = prevMeanSqrE - totMeanSqrE
         try:
            improProp = errImpro * 100.0 / prevMeanSqrE
         except ZeroDivisionError:
            break

         printerr("Mean squared error for %dth repeat: %g (improvement %g,  %g%%)"%(repeat,totMeanSqrE, errImpro, improProp ))
         prevMeanSqrE = totMeanSqrE





         repeat=repeatCount.next()

         # Synchronizing all threads and MPI processes
         if self.worldSize > 1 and self.__ioIBD_sync.n == 1:
            self.Barrier()
            pass

         self.__ioIBD_sync.Barrier()

      self.hLike = -numpy.log(self.hLike)
      #allocedIBD = allocedIBD[allocedIBD['ind1firstHaplo'] == 0]
      self.CPT = -numpy.log(self.CPT)

      
      
   def iterateOverIBDmax(self,allocedIBD,iterations,intermedFAD=None,intermedIBD=None,use_gold_standard=False):
      "Do the phasing proper, with the min-sum algorithm, trying to find most likely phase"

      totMeanSqrE=prevBestMeanSqrE=2e9

      repeatCount=it.count(0)
      repeat=repeatCount.next()
      #for repeat in xrange(iterations):
      startTime=time.time()
      newTime=startTime
      aboutPercent=max(1000,int(len(allocedIBD)/20)) * max(1, self.worldSize)

      firstIteration = True
      firstMeanSqrE, prevMeanSqrE = None, 1.0

      from array import array

      my_ibdSegments = array("l") # long

      __dampF = self.dampF
      #self.setDamping(0.0)
      __dampF = 0.0

      sOrder = allocedIBD.argsort(order=["endMarker"])[::-1]
      allocedIBD = allocedIBD[sOrder]
      sOrder = allocedIBD.argsort(order=["beginMarker"])
      allocedIBD = allocedIBD[sOrder]

      while  repeat<iterations:

         start_repeat_time=time.time()
         #allocedIBD = allocedIBD[numpy.random.permutation(len(allocedIBD))]



         #self.ibd_regions = None
         printerr("Repeat %d"%(repeat))
         totMeanSqrE = 0.0        
         

         if use_gold_standard:
            for ibdIdx,(ind1,ind2,beginM,endM,ca2h1,ca2h2,meanSqrD,_) in enumerate(allocedIBD):
               if ibdIdx%aboutPercent == 0:
                  oldTime,newTime=newTime,time.time()
                  pDone = ibdIdx*100.0/len(allocedIBD)
                  try:
                     expTime = (newTime-start_repeat_time)*100.0/pDone/60.0
                  except ZeroDivisionError :
                     expTime=0.0
                  printerr("Doing %d %d %d-%d (%d/%d = %g%% done. Expected time for rep %g min)"%(ind1, ind2, beginM, endM, ibdIdx, len(allocedIBD), pDone, expTime))



               meanSqrD[0]=self.__computeOneIBD(ind1,ind2,beginM,endM,ca2h1,ca2h2, doUpdate = True, serializedPass = firstIteration )


               firstIteration = False

               totMeanSqrE += meanSqrD[0]/len(allocedIBD)

         else:
            totMeanSqrE = self.processAllocedIBD(allocedIBD,MAPestimate=True)
            totMeanSqrE /= len(allocedIBD)


         if intermedFAD is not None:
            try:
               foutName=intermedFAD%(repeat)
            except TypeError:
               foutName=intermedFAD
            self.writeFAD(foutName)

         if isinstance(intermedIBD,"".__class__):
            try:
               foutName=intermedIBD%(repeat)
            except TypeError:
               foutName=intermedIBD
            self.writeIBD(foutName)

         errImpro = prevMeanSqrE - totMeanSqrE
         try:
            improProp = errImpro * 100.0 / prevMeanSqrE
         except ZeroDivisionError:
            break

         printerr("Mean squared error for %dth repeat: %g (improvement %g,  %g%%)"%(repeat,totMeanSqrE, errImpro, improProp ))
         prevMeanSqrE = totMeanSqrE





         repeat=repeatCount.next()

         # Synchronizing all threads and MPI processes
         if self.worldSize > 1 and self.__ioIBD_sync.n == 1:
            self.Barrier()
            pass

         self.__ioIBD_sync.Barrier()

      printerr("Running last iteration and making IBD calls")
      # Making ibd calls
      start_repeat_time=time.time()
      totMeanSqrE=0.0
      for ibdIdx,(ind1,ind2,beginM,endM,ca2h1,ca2h2,meanSqrD,_) in enumerate(allocedIBD):
         if ibdIdx%aboutPercent == 0:
            oldTime,newTime=newTime,time.time()
            pDone = ibdIdx*100.0/len(allocedIBD)
            try:
               expTime = (newTime-start_repeat_time)*100.0/pDone/60.0
            except ZeroDivisionError :
               expTime=0.0
            printerr("Doing %d %d %d-%d (%d/%d = %g%% done. Expected time for rep %g min)"%(ind1, ind2, beginM, endM, ibdIdx, len(allocedIBD), pDone, expTime))



            #meanSqrD[0]=self.__computeOneIBD(ind1,ind2,beginM,endM,ca2h1,ca2h2, doUpdate = True, serializedPass = firstIteration )
         meanSqrD[0]=self.__computeOneIBD(ind1,ind2,beginM,endM,ca2h1,ca2h2, doUpdate = True, serializedPass = firstIteration )

         firstIteration = False


         if (ca2h1.shape[0] == (endM-beginM+1) ):
            ind1, ind2 = int(ind1), int(ind2)
            my_ibdSegments.extend( it.chain.from_iterable( (ind1+h1,ind2+h2,startM,stopM) for  h1,h2,startM,stopM in self.callCurIBD(beginM,endM) ) )
            #self.ibdSegments.update((ind1+h1,ind2+h2,startM,stopM) \
            #                        for  h1,h2,startM,stopM in self.callCurIBD(beginM,endM))

            
         
         totMeanSqrE+=meanSqrD[0] / len(allocedIBD)
      printerr("Final error: %g"%(totMeanSqrE))
      if len(my_ibdSegments) > 0 :
         my_ibdSegments = numpy.array(my_ibdSegments,dtype=int).reshape((-1,4) )
         self.ibdSegments_lock.acquire()
         self.ibdSegments = numpy.vstack((self.ibdSegments,
                                          my_ibdSegments) )
         self.ibdSegments_lock.release()



      return totMeanSqrE


      

   def writeLike(self,fname):
      "Write likelihoods of 00, 10, 01, 11 haplotypes for each marker(rows) and individual(4 columns)"
      numpy.savetxt(fname,self.hLike.reshape((self.hLike.shape[0],-1),order="C"),fmt="%g")

   def loadLike(self,fname):
      myShape=self.hLike.shape[:]
      self.hLike=numpy.loadtxt(fname)
      self.hLike.shape=myShape
      self.hLike = self.hLike.view(ndarray_ADD)


   def callQual(self):
      "Return the call quality scores, i.e. base e log of posterior probability ratios between the two most likely calls"
      return numpy.sort(self.hLike,axis=-1)[:,:,:2].ptp(axis=2)
      # "Return the phase call quality scores. i.e. base e log of posterior probability ratios between the two phases."
      #return self.hLike[:,:,1:3].ptp(axis=2)



   def writeQual(self,fname):
      "Write call qualities (in base 10) to the given file."
      numpy.savetxt( fname, self.callQual()/numpy.log(10.0), fmt="%g", delimiter="\t" )

   def setCallThreshold(self,val):
      assert val >= 1.0, "The call threshold must not be less than one."
      self.callThreshold = val

   def writeVCF(self,fname,allow_inferred_genotypes=False):
      "Write current information (haplotypes, Alleles and frequencies, ..)  to a file in VCF format"
      import pysam
      vcf = pysam.VCF()
      
      vcf.setheader([("source","SLRP-%s"%(__version__) )])
      vcf.setsamples(self.famInfo)
      AFi = pysam.cvcf.FORMAT('AF1',pysam.VCF.NT_NUMBER,1,'Float',"MAP estimate of the site allele frequency of the first ALT allele",'.')
      GTf = pysam.cvcf.FORMAT('GT',pysam.VCF.NT_NUMBER,1,'String','Genotype','.')
      vcf.setformat({GTf.id:GTf})
      vcf.setinfo({AFi.id:AFi})


      phases = self.callPhases(allow_inferred_genotypes)


      def vcfGenerator(phases,poses,alleles,AF,samples):

         d  = { "chrom":self.chrom,
                "id":".",
                "qual":".",
                "filter":None,
                "format":["GT"] }
         if d["chrom"] is None:
            d["chrom"] = "Unkn"
            
         phase2str = {(0,0):["0|0"],
                      (0,1):["0|1"],
                      (1,0):["1|0"],
                      (1,1):["1|1"],
                      (2,2):["0/1"],
                      (3,3):["./."],
                      (3,1):[".|1"],
                      (1,3):["1|."],
                      (3,0):[".|0"],
                      (0,3):["0|."]}
         phase_arr = numpy.empty((4,4),dtype=numpy.object)
         for k,v in phase2str.iteritems(): phase_arr[k]=v


         for p,A,f,h in it.izip(poses,alleles,AF,phases):
               d["pos"] = p
               d["info"] = {"AF1":[1.0-f] }
               d["ref"] = A[0]
               d["alt"] = [ A[1] ]
               d.update((s,dict(GT=phase_arr[g[0],g[1]] )) for s,g in it.izip(samples,h))
               yield d

      phase_strm = vcfGenerator(phases, self.snpPos, self.alleles, self.zeroAF, self.famInfo)

      printerr("Openning "+fname)
      outstrm = open(fname,"w")
      vcf.write(outstrm, phase_strm)
      printerr("Closing")
      outstrm.close()
      


      
   def writeFAD(self,fname,allow_inferred_genotypes=False):
      "Write current haplotypes to file fname in FAD format"
      self.writeFADstrm(open(fname,"w"),allow_inferred_genotypes)

   def writeFAM(self,fname):
      "Write the sample identifiers to file fname in the order they are output"
      open(fname,"w").writelines("%s\n"%(x) for x in self.famInfo)

   def normalizedLike(self,marker=None):
      "Return normalized max-margin posteriors ('likelihoods') in range [0-1]"
      if marker is  None:
         tlike = numpy.exp(-self.hLike).transpose((2,0,1))
         tlike /= tlike.sum(axis=0)
         tlike = tlike.transpose((1,2,0))
      else:
         tlike = numpy.exp(-self.hLike[:,marker,:])
         tlike = (tlike.transpose() / tlike.sum(axis=1)).transpose()

      return tlike


   def genotypeProbs(self):
      "Return array of genotype (posterior)probabilities"
      
      probs = numpy.exp(-self.hLike).transpose((2,0,1))
      probs /= probs.sum(axis=0)
      probs = numpy.dstack((probs[0,:,:],probs[1:3,:,:].sum(axis=0), probs[3,:,:]))
      assert (probs<=1.0).all()
      assert (probs>=0.0).all()
      probs[ self.hLike.max(axis=2) < self.maxNoise ] = 0.0
      return probs.transpose((1,0,2))
   
   def writeIMPUTE(self,outFile):
      "Write output in IMPUTE file format"

      CHR = str(self.chrom)

      base = [CHR+" rsX %s","%s","%s"] + ["%s","%s","%s"]*self.indivs
      base = [CHR+" rsX %d","%c","%c"] + ["%g","%g","%g"]*self.indivs

      gProbs = self.genotypeProbs()
      assert (numpy.argsort(gProbs.strides) == [1,0,2]).all(), "gProbs is in dubious state for reshaping"

      #outD = numpy.hstack((self.snpPos.reshape((-1,1)),self.alleles.view(numpy.int8),gProbs.reshape( (self.markers,3*self.indivs) ) )
      outI = it.izip(self.snpPos,self.alleles,gProbs.reshape((self.markers,3*self.indivs)))
      gen = ("%s rsX %d %s %s "%(CHR,p,A[0],A[1])+" ".join("%g"%(x) for x in gp) for (p,A,gp) in outI)

      open(outFile,"w").writelines(gen)
      #base%(bpPos,A0,A1) for 
      

   def thresholdCalls(self,phases):
      "Remove phase/imputation calls that are supported too weakly"
      callThresholdP = self.callThreshold / (1.0 + self.callThreshold)

      nLike = self.normalizedLike().transpose((1,0,2))  # nLike.shape == (markers,indivs,4)
      # Places where we cant make reliable full calls:
      badCalls = nLike.max(axis=2) < callThresholdP

      phases[ badCalls  ] = self.haplos[badCalls]


      # Places where we didnt *impute* a full genotype
      badCalls = numpy.logical_and(badCalls, self.haplos[:,:,0] == 3)

      firstOne = numpy.logical_and( nLike[:,:,(1,3)].sum(axis=2) > callThresholdP, badCalls)
      phases[firstOne,0]=1
      del(firstOne)

      secondOne = numpy.logical_and(nLike[:,:,(2,3)].sum(axis=2) > callThresholdP, badCalls)
      phases[secondOne,1]=1
      del(secondOne)

      firstZero = numpy.logical_and( nLike[:,:,(0,2)].sum(axis=2) > callThresholdP, badCalls)
      phases[firstZero,0]=0
      del(firstZero)

      secondZero = numpy.logical_and(nLike[:,:,(0,1)].sum(axis=2) > callThresholdP, badCalls)      
      phases[secondZero,1]=0
      del(secondZero)

      return phases

   def thresholdCallsYield(self,all_phases):
      "Remove phase/imputation calls that are supported too weakly"
      callThresholdP = self.callThreshold / (1.0 + self.callThreshold)

      for marker in xrange(self.markers):
         phases = all_phases[marker,:,:]
         nLike = self.normalizedLike(marker)  # nLike.shape == (indivs,4)
         assert nLike.shape == (self.indivs,4)
         # Places where we cant make reliable full calls:
         badCalls = nLike.max(axis=1) < callThresholdP

         phases[ badCalls  ] = self.haplos[marker,badCalls]


         # Places where we didnt *impute* a full genotype
         badCalls = numpy.logical_and(badCalls, self.haplos[marker,:,0] == 3)

         firstOne = numpy.logical_and( nLike[:,(1,3)].sum(axis=1) > callThresholdP, badCalls)
         phases[firstOne,0]=1
         del(firstOne)

         secondOne = numpy.logical_and(nLike[:,(2,3)].sum(axis=1) > callThresholdP, badCalls)
         phases[secondOne,1]=1
         del(secondOne)

         firstZero = numpy.logical_and( nLike[:,(0,2)].sum(axis=1) > callThresholdP, badCalls)
         phases[firstZero,0]=0
         del(firstZero)

         secondZero = numpy.logical_and(nLike[:,(0,1)].sum(axis=1) > callThresholdP, badCalls)      
         phases[secondZero,1]=0
         del(secondZero)

         yield phases



   def callPhases(self,allow_inferred_genotypes=False):
      "Return an array of called phases"

      _unreach =gc.collect()
      if _unreach > 0: printerr("Unreachable objects: %d"%(_unreach))

      idx2phase = numpy.array([(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)], dtype = numpy.int8)

      # Call most likely phase:
      # 0 = 00
      # 1 = 10
      # 2 = 01
      # 3 = 11
      hLike = self.hLike
      if not allow_inferred_genotypes:  # Only allow phase calls that are consistent with observed genotypes
         self.computeGenos()
         hLike = hLike.view(numpy.ma.MaskedArray)
         hLike.fill_value = numpy.finfo(hLike.dtype).max
         hLike.mask = numpy.zeros(hLike.shape, dtype=bool )
         hLike.mask[self.geno==0] = numpy.array([False, True, True, True],dtype=bool )
         hLike.mask[self.geno==1] = numpy.array([True, True, True, False],dtype=bool )
         hLike.mask[self.geno==2] = numpy.array([True, False, False, True],dtype=bool )

         
      posCalls = hLike.argmin(axis=2)




      #fadA = numpy.array([ numpy.fromstring(x, sep="", dtype=numpy.int8) for x in self.fad["haplos"] ] , dtype = numpy.int8) - ord('0')
      #fadA.shape = (self.markers,-1,2)  # = (markers,indivs,2)



      #phases_old = numpy.array(idx2phase[posCalls]).transpose((1,0,2))
      printerr("Calling most likely alleles")
      phases = idx2phase[posCalls].transpose((1,0,2))
      #assert (phases_old==phases).all()
      phases.shape = (self.markers, -1, 2) # = (markers,indivs,2)


      if self.callThreshold > 1.0:
         printerr("Thresholding..")
         #phases = self.thresholdCalls(phases)
         phases = self.thresholdCallsYield(phases)

      printerr("About done..")
      return phases
         
   def writeFADstrm(self,outf,allow_inferred_genotypes=False):
      "Call haplotypes from current likelihoods and write to file fname in FAD format"

      phases = self.callPhases(allow_inferred_genotypes)
      
      outf.writelines("%d\t%s\n"%(pos,(hl+ord('0')).tostring()) for pos,hl in it.izip(self.snpPos,phases)  )




   def Gather(self, sendbuf, root = 0):
      "Gathers arbitrary size numpy arrays from non root processes"

      comm = MPI.COMM_WORLD

      totSize = numpy.zeros( self.worldSize, dtype = numpy.int32)
      mySize  = numpy.array([ sendbuf.nbytes], dtype = numpy.int32)

      comm.Gather(mySize, totSize, root = root)

      if self.myRank == root:
         #print "totSize:",totSize
         recvBuf = [ numpy.empty( (subSize / (4*sendbuf.itemsize), 4) , dtype = sendbuf.dtype )
                     for subSize in totSize ]
         reqs = [ comm.Irecv(rbuf, source = i, tag = TAG_NON_ROOT_GATHER) for i,rbuf in enumerate(recvBuf) ]
      else:
         #print self.myRank,sendbuf
         reqs = []
         recvBuf = []

      sendReq = comm.Isend(sendbuf, dest = root, tag = TAG_NON_ROOT_GATHER)
      
      for Req in reqs:
         Req.Wait()

      sendReq.Wait()


      if self.myRank == root :
         #print recvBuf
         recvBuf = numpy.concatenate([ x for x in recvBuf if len(x)>0 ])
         #print recvBuf
         
      

         
      return recvBuf
      

      
   def Barrier(self):
      "MPI Barrier. Need to synchronize all jobs (including the server)"
      if self.worldSize > 1:
         printerr("MPI barrier")
         
         MPI.COMM_WORLD.Barrier()
         

   def phaseMPI(self, iterations = 10):
      "Wrapper for phasing over MPI"

      comm = MPI.COMM_WORLD
      
      self.myRank = comm.Get_rank()
      self.worldSize = comm.Get_size()

      #      sys.stdout.write("Starting hLike process %d of %d on %s.\n"  % (self.myRank , self.worldSize, MPI.Get_processor_name()))
      if self.myRank == 0:
         self.firstCP2P = numpy.array(self.firstCP2P, dtype = self.dataType)
      else:
         self.firstCP2P = numpy.ones(5, dtype = self.dataType)
      comm.Bcast( self.firstCP2P, root = 0)

      
      #self.hLike = mpi_ndarrayMsg(self.hLike, copyData = False)
      self.hLike = mpi_ndarrayRMA(self.hLike, copyData = False)
         
      self.hLike.Broadcast()


      #if self.myRank == 0:
      #   fromRMA= self.hLike[5,5:10,:]
      #   sys.stdout.write("hLike[5,5:10,:] on  %d of %d on %s is  %s.\n"  % (self.myRank , self.worldSize, MPI.Get_processor_name(), str(fromRMA.shape) ) )


#      sys.stdout.write("Starting CAj process %d of %d on %s.\n"  % (self.myRank , self.worldSize, MPI.Get_processor_name()))
      self.CAj = mpi_ndarrayMsg(self.CAj)
      self.CAj.Broadcast()
      self.CAj = self.CAj.data

      
#      sys.stdout.write("Starting CPj process %d of %d on %s.\n"  % (self.myRank , self.worldSize, MPI.Get_processor_name()))
      self.CPj = mpi_ndarrayMsg(self.CPj)
      self.CPj.Broadcast()
      self.CPj = self.CPj.data

#      sys.stdout.write("Starting ibd_regions process %d of %d on %s.\n"  % (self.myRank , self.worldSize, MPI.Get_processor_name()))

      # Scatter the ibd Segments
      # Broadcast the necessary buffer size
      self.__waitIBD()
      if self.ibd_regions is not None:
         basicCounts = numpy.array([self.ibd_regions.shape[0],self.markers,self.indivs],
                                   dtype = numpy.int32)
      else:
         basicCounts = numpy.zeros((3,), dtype = numpy.int32)

#      sys.stdout.write("Broadcasting ibd_regions %s process %d of %d on %s.\n"  % (str(ibdSegments),self.myRank , self.worldSize, MPI.Get_processor_name()))

#      print ibdSegments,type(ibdSegments)
      comm.Bcast(basicCounts , root = 0)
      #      sys.stdout.write("Got broadcast  ibd_regions %s process %d of %d on %s.\n"  % (str(ibdSegments),self.myRank , self.worldSize, MPI.Get_processor_name()))
      ibdSegments,self.markers,self.indivs = basicCounts


      # Create necessary buffers 
      chunkSize = int( ibdSegments * 1.0 / ( self.worldSize - 1) + 1) 

#      if self.myRank == 0: print "OK:",self.CAj[0]

      # Scattering manually
      if self.myRank == 0 :
         assert chunkSize * ( self.worldSize - 1) >= len(self.ibd_regions)
         reqs = []
         sendBuf = self.ibd_regions[ ["ind1","ind2","beginM","endM"] ].view((numpy.int32,4))

         lastChunkSize =  len(sendBuf) - (self.worldSize-2)*chunkSize 
         #print "LAST CHUNK SIZE:",lastChunkSize
         comm.send( lastChunkSize, dest = self.worldSize - 1 , tag = TAG_CHUNK_SIZE)
         
         #print sendBuf
         for i in range(1, self.worldSize):
            Req = comm.Isend( sendBuf[ ((i-1)*chunkSize):( min( len(sendBuf), (i)*chunkSize) ) ], dest = i, tag = TAG_CHUNK)
            reqs.append( Req )

         for Req in reqs:
            #print "Waiting for",Req
            Req.Wait()

      else:
         #         sys.stdout.write("Making buffers process %d of %d on %s.\n"  % (self.myRank , self.worldSize, MPI.Get_processor_name()))
         if self.myRank == (self.worldSize - 1):
            chunkSize = comm.recv(source = 0, tag = TAG_CHUNK_SIZE)

         recvBuf = numpy.empty((chunkSize,4), dtype = numpy.int32 )
#         print "recv",recvBuf.shape

         comm.Irecv(recvBuf, source = 0, tag = TAG_CHUNK).Wait()

         #self.ibd_regions = recvBuf.view(dtype = numpy.dtype(self.ibd_regions_dtype))
         self.ibd_regions = recvBuf.view( dtype = {'names':["ind1","ind2","beginM","endM"],
                                             'formats':[numpy.int32,numpy.int32,numpy.int32,numpy.int32]} )
         #print self.ibd_regions


      # create lock for weave inline compilation
      if self.myRank == 0:
         _inlineLock = numpy.array([0])
      else:
         _inlineLock = None
      
      self.__inlineLock = MPI.Win.Create(_inlineLock)



      # Do the phasing proper:

      if self.myRank > 0:
         self.phase(iterations)
         self.hLike.Finished()
         _unreach =gc.collect()
         if _unreach > 0: printerr("Unreachable objects: %d"%(_unreach))
      else:
         def mpiExceptHook(err_type, value, tb):
            sys.__excepthook__(err_type, value, tb)
            likeName = "longRangePhaseMPI.like." + MPI.Get_processor_name()
            sys.stderr.write("Exception in %s process %d\n"%(MPI.Get_processor_name(), MPI.COMM_WORLD.Get_rank() ))
            sys.stderr.write("Trying to write likelihoods to %s\n"%( likeName ))
            
            self.writeLike(likeName)
            sys.stderr.write("Succeeded. Hurray!\n")
            MPI.COMM_WORLD.Abort()
            
         sys.excepthook = mpiExceptHook

         step_len = min( self.slice_len, self.markers)
         for r in range(iterations):
            for firstBase in range(0, self.markers, step_len):
               self.hLike.Serve()
               self.Barrier()
         
         #for i in range(iterations):
         #   self.Barrier()
            
      #print MPI.Get_processor_name(),self.myRank,self.CAj.shape,
      #print MPI.Get_processor_name(),self.myRank,self.CPj.shape,
      #print MPI.Get_processor_name(),self.myRank,self.ibd_regions.shape
      #print MPI.Get_processor_name(),self.myRank,self.hLike.shape,type(self.hLike.data)

      # Synchronize
      self.Barrier()

      #print self.myRank,self.ibdSegments[:10]
      

      if self.myRank == 0:
         ibdSegments = numpy.empty((0,4),dtype = numpy.int32)
      else:
         ibdSegments = numpy.array( list(self.ibdSegments) ,
                                    dtype = numpy.int32)
         ibdSegments.shape = (-1,4)
      ibdSegments = self.Gather(ibdSegments, root = 0)
      self.ibdSegments = ibdSegments


         


      # Clean up
      try:
         self.hLike.win.Free()
      except AttributeError:
         pass
      self.__inlineLock.Free()
      MPI.Finalize()
      
      if self.myRank == 0:
         self.hLike = self.hLike.data
      printerr( "Continuing")


   def set_slice_len(self,slice_length=-1):
      "Set the length of the slice to phase at one time as number of markers. Non positive values set the default, full chromosome lengh"
      if slice_length < 1:
         self.slice_len = self.markers
      else:
         self.slice_len = slice_length
         

   def ____assist_arrays(self):
      "Make the assistant arrays for phasing. Keep them thread local"
      if not hasattr(self.__assist,"backtrack"):
         #self.__assist.cp2pN=numpy.zeros((5,self.markers),dtype=self.dataType,order='F')
         self.__assist.cp2pP=numpy.zeros((5,self.markers+1),dtype=self.dataType,order='F')
         self.__assist.ca2p=numpy.zeros((5,self.markers+1),dtype=self.dataType,order='F')
         #self.__assist.p2ca=numpy.zeros((5,self.markers),dtype=self.dataType,order='F')
         #self.__assist.p2cpN=numpy.zeros((5,self.markers+1),dtype=self.dataType,order='F')
         #self.__assist.p2cpP=numpy.zeros((5,self.markers),dtype=self.dataType,order='F')
         self.__assist.backtrack=numpy.zeros((5,self.markers+1),dtype=numpy.int,order='F')


   def filterIBDcover(self, coverLimit, lenLimit = -1):
      "Filter IBD segments according to haplotype coverage and length (hard length limit"
      #segLengthDesc = self.ibd_regions["score"].argsort()[::-1]
      segLengthDesc = numpy.arange(len(self.ibd_regions))
      ibdCoverCounts = numpy.zeros( (self.markers, self.indivs,2),
                                    dtype=numpy.uint32)

      self.__waitIBD()

      ibdSequence = (self.ibd_regions[i] for i in segLengthDesc if self.ibd_regions["endM"][i]-self.ibd_regions["beginM"][i] +1 >= lenLimit)

      allocedSize = 1000
      nextIdxToFill = 0
      newIBD = numpy.recarray(shape=(allocedSize,),dtype = self.ibd_regions_dtype)
      addCount,skipCount = 0,0

      for ind1,ind2,score,beginM,endM,beginBP,endBP in ibdSequence:
            i1view = ibdCoverCounts[beginM:endM+1,ind1/2,ind1%2]
            i2view = ibdCoverCounts[beginM:endM+1,ind2/2,ind2%2]
            if min(i1view.min(),i2view.min()) < coverLimit:
               #printerr("Adding",ind1,ind2,beginM,endM)
               i1view += 1
               i2view += 1
               if len(newIBD) <= nextIdxToFill:
                  i1view = None
                  i2view = None
                  printerr("Extending newIBD to",len(newIBD)*2)
                  newIBD.resize((len(newIBD)*2,))
                  
               newIBD[nextIdxToFill] = (ind1,ind2,score,beginM,endM,beginBP,endBP)
               nextIdxToFill += 1
               addCount += 1
            else:
               skipCount += 1
               printerr("Skipping",ind1,ind2,score,beginM,endM,"Because",i1view.min(),i2view.min())
               #print self.ibd_regions[(self.ibd_regions["ind1"]==4)&(self.ibd_regions["beginM"]<=3896)&(self.ibd_regions["endM"]>=4015)]
               pdb.set_trace()

      i1view = None
      i2view = None
      
      self.ibd_regions = newIBD
      newIBD = None
      self.ibd_regions.resize((nextIdxToFill,))
      
      return (addCount,skipCount,ibdCoverCounts)

      

   allocedIBD_dtype = { 'names' : ( "ind1firstHaplo", "ind2firstHaplo",  "beginMarker",  "lastMarkerFilled", "p2h1", "p2h2",  "prevMeanSqrDiff", "endMarker"),
                        'formats' : (numpy.int32,     numpy.int32,       numpy.int32,    numpy.int32,   numpy.object, numpy.object, (numpy.float64,(1,1)), numpy.int32) }



   def phasePreProcHighMem(self, ibdSegmentCalls = None,min_cover = 15,min_ibd_length = 10):
      "Compute the first step of SLRP. i.e. find putative IBD segments. This version needs a lot of memory with large datasets."
      
      allPairs_iter = ((2*ind1,2*ind2)  for ind2 in xrange(self.indivs) for ind1 in xrange(ind2))

      numPairs = self.indivs * ( self.indivs - 1 ) / 2
      allPairs = numpy.fromiter(it.combinations(xrange(0,2*self.indivs,2),2),dtype="int,int",count=numPairs).view(int).reshape(-1,2)
      #random.shuffle(allPairs)
      ibdRegions=[]
      newTime=time.time()
      startTime=newTime

      self.computeGenos()
      self.computeLogLikeTable()
      
      if self.poolSize <= 1 or numPairs < 3 * self.poolSize :
         if self.poolSize > 1:
            self.set_workers(1)
            
         ibdRegions=self.computeOneIBDnoUpdate( allPairs )
      else:
         #TODO: use numpy rec arrays instead of lists. Should be much faster.
         chunkSize = int( numPairs * 1.0 / self.poolSize + 1) 

         printerr("Chunks of size",chunkSize)
         #subsetPairs=list( list(it.islice(allPairs,chunkSize))  for i in range(0, numPairs, chunkSize) )
         subsetPairs=[allPairs[i:(i+chunkSize),:] for i in range(0, numPairs, chunkSize) ]
         
         printerr("subsets",len(subsetPairs),[len(x) for x in subsetPairs])
         ibdRegions = self.pmap(self.computeOneIBDnoUpdate, subsetPairs )
         ibdRegions = numpy.concatenate(ibdRegions)
         newTime = time.time()


      candidateCount = ibdRegions.shape[0]
      ibdScore = self.snpCMpos[ibdRegions[:,3]] - self.snpCMpos[ibdRegions[:,2]]
      ibdScore.shape = (-1,1)
      ibdRegions = numpy.hstack((ibdRegions,(1000*ibdScore))) # Score as integer milliMorgans
      ibdRegions = numpy.ascontiguousarray(ibdRegions,dtype=numpy.int32)
      #ibdRegions=ibdRegions[(ibdRegions[:,0]==4)]
      #&(ibdRegions[:,2]<=3896)&(ibdRegions[:,3]>=4015)]
      #ibdRegions[:,0]=ibdRegions[:,1]
      #ibdRegions[:,0]=0
      #ibdRegions[:,1]=0
      ibdRegions = self.c_ext.IBD_filter(ibdRegions,self.indivs,min_cover,min_ibd_length)
      printerr( "All pairs took %g mins"%( ( time.time() - startTime ) / 60.0 ))
      printerr("Using %d / %d = %g%% of found putative IBD regions"%(ibdRegions.shape[0],candidateCount,ibdRegions.shape[0]*100.0/candidateCount))


      myStack = numpy.hstack(( ibdRegions[:,:2], numpy.reshape(ibdRegions[:,4],(-1,1)), ibdRegions[:,2:4], self.snpPos[ibdRegions[:,2:4]]) )
      self.ibd_regions = numpy.ascontiguousarray(myStack, dtype=numpy.int32)

      self.ibd_regions = self.ibd_regions.T.view(self.ibd_regions_dtype_int)[0]


      if ibdSegmentCalls is not None:
         open(tools.addSuffix(ibdSegmentCalls,".aibd"),"w").writelines("%d\t%d\t%g\t%d\t%d\t%d\t%d\n"%(x[0],x[1],float(x[2]),x[3],x[4],x[5],x[6]) for x in self.ibd_regions)
         printerr("Max end",self.ibd_regions["endM"].max())



   def phasePreProc(self, ibdSegmentCalls = None,min_cover = 15,min_ibd_length = 10):
      "Compute the first step of SLRP. i.e. find putative IBD segments"
      
      ibdRegions=[]
      newTime=time.time()
      startTime=newTime
      
      self.computeGenos()
      self.computeLogLikeTable()
      
      assert self.geno.flags.c_contiguous, "Genotype array must be C-contiguous"
      assert self.geno.strides[1] == 1, "Geno array must be indexable by first coordinate."
      assert self.LLtable.shape == (self.markers,4,4), "Badly shaped LL table"
      assert self.LLtable.flags.c_contiguous, "LLtable array must be C-contiguous"
      peakThreshold = self.LLtable.max();
      dipThreshold = -self.LLtable.min() * 1.5;
      peakThreshold = numpy.log(4.0)
      dipThreshold = 1000.0
      printerr("Log Likelihood values range from %g to %g."%(self.LLtable.min(),self.LLtable.max()))
      printerr("Calling IBD segments with LL score of at least %g containing no decrease of score %g."%(peakThreshold,dipThreshold))
#      printerr("Naa.. Just kidding. I'm really calling IBD segments with LL score of at least %g."%(peakThreshold))

      assert peakThreshold > 0
      assert dipThreshold > 0
      if self.poolSize <= 1 or self.indivs < self.poolSize * 3 :
         ibdRegions=self.c_ext.LLscan_and_filter(0,self.indivs,self.geno,self.LLtable,peakThreshold,dipThreshold,min_cover,min_ibd_length )
      else:
         chunkSize = int( self.indivs * 1.0 / self.poolSize + 1) 

         #subsetPairs=list( list(it.islice(allPairs,chunkSize))  for i in range(0, numPairs, chunkSize) )
         subsetPairs=[(i,min(self.indivs,i+chunkSize)) for i in range(0, self.indivs, chunkSize) ]
         
         printerr("Chunks of size",chunkSize)
         printerr("subsets",len(subsetPairs),subsetPairs)
         ibdRegions = self.pmap(lambda x:self.c_ext.LLscan_and_filter(x[0],x[1],self.geno,self.LLtable,peakThreshold,dipThreshold,min_cover,min_ibd_length ), subsetPairs )
         ibdRegions = numpy.concatenate(ibdRegions)
         ibdRegions = tools.uniqueRows(ibdRegions)
         newTime = time.time()


      #ibdRegions=ibdRegions[(ibdRegions[:,0]==4)]
      #&(ibdRegions[:,2]<=3896)&(ibdRegions[:,3]>=4015)]
      #ibdRegions[:,0]=ibdRegions[:,1]
      #ibdRegions[:,0]=0
      #ibdRegions[:,1]=0
      printerr( "All pairs took %g mins"%( ( time.time() - startTime ) / 60.0 ))
      printerr("Using %d found putative IBD regions"%(ibdRegions.shape[0]))

      # Shuffle the update order. This should avoid overcommiting and improve load balancing.
      numpy.random.shuffle(ibdRegions)


      myStack = numpy.hstack(( ibdRegions[:,:2], numpy.reshape(ibdRegions[:,4],(-1,1)), ibdRegions[:,2:4], self.snpPos[ibdRegions[:,2:4]]) )
      self.ibd_regions = numpy.ascontiguousarray(myStack, dtype=numpy.int32)

      self.ibd_regions = self.ibd_regions.T.view(self.ibd_regions_dtype_int)[0]


      if ibdSegmentCalls is not None:
         open(tools.addSuffix(ibdSegmentCalls,".aibd"),"w").writelines("%d\t%d\t%g\t%d\t%d\t%d\t%d\n"%(x[0],x[1],float(x[2]),x[3],x[4],x[5],x[6]) for x in self.ibd_regions)
         printerr("Max end",self.ibd_regions["endM"].max())




   def estimate_max_message_mem(self):
      "Return the estimate for the maximum memory consumption for the message passing in bytes"

      step_len = min( self.slice_len, self.markers)
      ibd_bytes = numpy.dtype(self.allocedIBD_dtype).itemsize
      dtype_bytes = numpy.dtype(self.dataType).itemsize

      max_mem = 0.0
      for firstBase in range(0, self.markers, step_len):
         lastBase = firstBase + step_len
         toAllocIBDindicator =  numpy.logical_and( numpy.logical_and( firstBase<= self.ibd_regions["endM"], self.ibd_regions["beginM"] < lastBase),
                                               (self.ibd_regions["endM"] - self.ibd_regions["beginM"] + 1) >= self.minIBDlength)




         toAllocIBD = (self.ibd_regions[x] for x in toAllocIBDindicator.nonzero()[0])
         overhang = 10
         nAllocIBD = toAllocIBDindicator.sum()
         nMarkers = sum(int(x["endM"] - x["beginM"]) + 2 * overhang  for  x in toAllocIBD )

         this_mem = nAllocIBD*ibd_bytes + 2.0 * nMarkers * 4.0 * dtype_bytes
         
         printerr("Expected memory for messages at slice %s-%s: %g GB"%(firstBase,lastBase, this_mem * 2**(-30)))

         max_mem = max(max_mem,this_mem)

      return max_mem
       
   def phase(self,iterations=10,intFADbase=None,intIBDbase=None,use_max_product=True):
      "Compute the long range phase over IBD segments in allocedIBD"

      if iterations < 1:  # Don't do anything if there is nothing to do
         return

      self.__waitIBD()
      assert(self.ibd_regions is not None)
      

      allocedIBD = numpy.array([], dtype = self.allocedIBD_dtype)

      step_len = min( self.slice_len, self.markers)

#      ibd_segment_cache = set()

      pre_mem = resident()
      max_msg_mem = self.estimate_max_message_mem()
      printerr("Expected RSS memory requirement above %g GB"%( (pre_mem + max_msg_mem) * 2**-30))

      #self.ibdSegments = set()
      self.ibdSegments = numpy.empty((0,4), dtype = int )

      for firstBase in range(0, self.markers, step_len):
         lastBase = firstBase + step_len
         #allocedIBD = []
         printerr("RSS before allocations: %g GB"%(resident()*2.0**(-30)))

         #if len(allocedIBD)>0: printerr(sys.getrefcount(allocedIBD["p2h1"][0]), "should be 2 at line %d."%(tools.lineno()))

         if len(allocedIBD) > 0:
            _selector = allocedIBD["endMarker"] > firstBase
            _tmp = allocedIBD[ _selector ]
            alloced = None
            allocedIBD = _tmp.copy()
            _tmp = None
            #allocedIBD = allocedIBD[ allocedIBD["endMarker"] > firstBase ].copy()
            allocedIBD["lastMarkerFilled"] = numpy.minimum(lastBase, allocedIBD["endMarker"])
            allocedIBD["prevMeanSqrDiff"] = 1e99
            gc.collect()


         _unreach =gc.collect()
         if _unreach > 0: printerr("Unreachable objects: %d"%(_unreach))


         #if lastBase < self.markers:
         printerr("RSS after passing wave: %g GB"%(resident()*2.0**(-30)))



         # Segments that start within this span of markers

         oldIBDcount = len(allocedIBD)
         # Indicators to self.ibd_regions for rows to add to allocedIBD
         toAllocIBDindicator =  numpy.logical_and( numpy.logical_and( firstBase<= self.ibd_regions["beginM"], self.ibd_regions["beginM"] < lastBase),
                                               (self.ibd_regions["endM"] - self.ibd_regions["beginM"] + 1) >= self.minIBDlength)


         printerr("Future allocation stats: #ibd regions: %d  #allocated IBD: %d  #to allocate ibd: %d"%(self.ibd_regions.shape[0], len(allocedIBD),
                                                                                                         sum(toAllocIBDindicator)))


         _unreach =gc.collect()
         if _unreach > 0: printerr("Unreachable objects: %d"%(_unreach))

         toAllocIBD = (self.ibd_regions[x] for x in toAllocIBDindicator.nonzero()[0])
         overhang = 10
         newIBD = numpy.array( [(int(x["ind1"] - x["ind1"] % 2),
                                    int(x["ind2"] - x["ind2"] % 2), 
                                    int(max(0, int(x["beginM"]) - overhang)),
                                    int(min(self.markers-1, int(x["endM"]) + overhang, lastBase)), 
                                    numpy.zeros((min(self.markers-1,x["endM"] + overhang) - max(0, x["beginM"] - overhang) + 1, 4), dtype = self.dataType),
                                    numpy.zeros((min(self.markers-1,x["endM"] + overhang) - max(0, x["beginM"] - overhang) + 1, 4), dtype = self.dataType),
                                    numpy.array([1e99], dtype = self.dataType),
                                    min(self.markers-1, int(x["endM"]) + overhang ) ) \
                                   for  x in toAllocIBD ] ,
                                  dtype = self.allocedIBD_dtype)

#          altIBD = numpy.fromiter( ( (int(x["ind1"] - x["ind1"] % 2),
#                                     int(x["ind2"] - x["ind2"] % 2), 
#                                     int(max(0, int(x["beginM"]) - overhang)),
#                                     int(min(self.markers-1, int(x["endM"]) + overhang, lastBase)), 
#                                     numpy.zeros((min(self.markers-1,x["endM"] + overhang) - max(0, x["beginM"] - overhang) + 1, 4), dtype = self.dataType),
#                                     numpy.zeros((min(self.markers-1,x["endM"] + overhang) - max(0, x["beginM"] - overhang) + 1, 4), dtype = self.dataType),
#                                     numpy.array([1e99], dtype = self.dataType),
#                                     min(self.markers-1, int(x["endM"]) + overhang ) ) \
#                                    for  x in toAllocIBD ) , count = toAllocIBDindicator.sum(),
#                                   dtype = self.allocedIBD_dtype)

         printerr("RSS after allocations: %g GB"%(resident()*2.0**(-30)))

         _unreach =gc.collect()
         if _unreach > 0: printerr("Unreachable objects: %d"%(_unreach))


         #allocedIBD.sort(key = lambda x:(x[2],-x[3]))
         newIBD.sort(order = [ "beginMarker","endMarker"] )  # TODO: Check results!! sort order is different from the list version

         #aIBD = allocedIBD
         old_aIBD_len = len(allocedIBD)
         nIBD_len = len(newIBD)
         
         #allocedIBD = numpy.hstack( [allocedIBD, newIBD] )
         allocedIBD.resize((old_aIBD_len+nIBD_len,))
         allocedIBD[old_aIBD_len:] = newIBD
         newIBD = None


         _unreach =gc.collect()
         if _unreach > 0: printerr("Unreachable objects: %d"%(_unreach))

         printerr("RSS after stacking: %g GB"%(resident()*2.0**(-30)))
         _unreach =gc.collect()
         if _unreach > 0: printerr("Unreachable objects: %d"%(_unreach))

         if len(allocedIBD) == 0:
            # Skip to next slice if there are no IBD segments to analyse
            continue


         printerr("Allocated %gGB for %d IBD regions overlapping markers %d - %d"%( sum(x[4].nbytes+x[5].nbytes+x[6].nbytes for x in allocedIBD) * 1.0 / 2.0**30,
                                                                                    len(allocedIBD), firstBase, min(self.markers, lastBase) ))

         if self.poolSize <= 1 or len(allocedIBD) < 2*self.poolSize:
            if self.poolSize > 1:
               self.set_workers(1)
            self.iterateOverIBD(allocedIBD,iterations,intermedFAD=intFADbase,intermedIBD=intIBDbase,use_max_product=use_max_product)
         else:
            # Trying to load balance chunks
            cum_len = numpy.cumsum(allocedIBD["lastMarkerFilled"]-allocedIBD["beginMarker"] ) 
            chunk_count = self.poolSize

            chunk_size_markers = cum_len[-1] / chunk_count
            breaks = numpy.searchsorted(cum_len, numpy.array([chunk_size_markers * i for i in range(1, chunk_count) ] ) )
            prev_break = 0
            chunk_slices = []
            for next_break in breaks:
               chunk_slices.append(allocedIBD[prev_break:next_break])
               prev_break = next_break
            chunk_slices.append(allocedIBD[prev_break:])
            printerr("Chunks of plausible IBD regions: "+str(map(len,chunk_slices)))
            


                                        
            # Threading
            #chunkSize = int( len(allocedIBD) * 1.0 / self.poolSize + 1)
            # TODO: move the threading deeper down. From here, we can only give one chunk for each thread, which is bad from load balancing perspective.
            handythread.foreach(lambda chunk: self.iterateOverIBD(chunk, iterations, intermedFAD=intFADbase,intermedIBD=intIBDbase),
                                chunk_slices,
#                                [allocedIBD[(i*chunkSize):((i+1)*chunkSize)] for i in range(self.poolSize) ],
                                threads = self.poolSize )
            chunk_slices = None
         #ibd_segment_cache.update(self.ibdSegments)
      else:
         pass
      #self.ibdSegments = list(ibd_segment_cache)
      #self.ibdSegments = None
      #self.ibdSegments = ibd_segment_cache
      self.__assist = None



            
