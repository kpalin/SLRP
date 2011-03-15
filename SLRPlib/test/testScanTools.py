from nose import *
from nose.tools import *
import numpy



class testScanToolsImport:
    def __testFloatImport(self):
        "Test import scanTools with float32/float bindings"
        import SLRPlib.scanTools
        c_ext = SLRPlib.scanTools.c_ext(numpy.float32)


    def __testFloatImport(self):
        "Test import scanTools with float32/float bindings"
        import SLRPlib.scanTools
        c_ext = SLRPlib.scanTools.c_ext(numpy.float64)


class testIBDfilter:
    def setup(self):
        import SLRPlib.scanTools
        self.c_ext = SLRPlib.scanTools.c_ext(numpy.float64)


    def testSingleFullCover(self):
        "Test simplest case with only one candidate that passes"
        cand_ibd = numpy.array([[0,2,0,1000,99]],dtype=numpy.int32)
        out_ibd = self.c_ext.IBD_filter(cand_ibd,2,1,1)

        eq_(cand_ibd.shape,out_ibd.shape)
        ok_((cand_ibd==out_ibd).all())

    def testSingleFullCover2(self):
        "Trio with two segments"
        cand_ibd = numpy.array([[0,2,0,1000,99],
                                [2,4,0,1000,99],
                                ],dtype=numpy.int32)

        out_ibd = self.c_ext.IBD_filter(cand_ibd,3,1,3)

        eq_(cand_ibd.shape,out_ibd.shape)
        ok_((cand_ibd==out_ibd).all())

    def testSingleFullCover3(self):
        "Trio with three segments. One discarded."
        cand_ibd = numpy.array([[0,2,0,1000,99],
                                [2,4,0,1000,99],
                                [0,4,0,1000,99],
                                ],dtype=numpy.int32)

        assert cand_ibd.strides[0]==20
        assert cand_ibd.strides[1]==4
        out_ibd = self.c_ext.IBD_filter(cand_ibd,3,1,3)

        eq_(out_ibd.shape,(2,5))
        ok_((out_ibd==cand_ibd[:2]).all())

        


    def testSingleFullCover4(self):
        "Duo with three segments. One discarded."
        cand_ibd = numpy.array([[0,2,0,500,99],
                                [0,2,501,1000,99],
                                [0,2,0,1000,99],
                                ],dtype=numpy.int32)

        assert cand_ibd.strides[0]==20
        assert cand_ibd.strides[1]==4
        out_ibd = self.c_ext.IBD_filter(cand_ibd,3,1,3)

        eq_(out_ibd.shape,(2,5))
        ok_((out_ibd==cand_ibd[:2]).all())

        

    def testSingleFullCover5(self):
        "Due with three segments. all included."
        cand_ibd = numpy.array([[0,2,0,500,99],
                                [0,2,502,1000,99],
                                [0,2,0,1000,99],
                                ],dtype=numpy.int32)

        assert cand_ibd.strides[0]==20
        assert cand_ibd.strides[1]==4
        out_ibd = self.c_ext.IBD_filter(cand_ibd,3,1,3)

        eq_(out_ibd.shape,(2,5))
        ok_((out_ibd==cand_ibd[:2]).all())

        


        
