from nose import *
from nose.tools import *
import tempfile


from SLRPlib import SLRP
from SLRP import IBD_diplotype_codes as c

class testSLRPlibImport:

    
    def testSLRPlibImport(self):
        "Testing SLRPlib import"
        t = SLRP.longRangePhase()

        
    def testnew2oldCode0(self):
        "Test new2oldCode()"
        t = SLRP.longRangePhase()
        # 1 ibd 1
        # 0     0
        #
        # 1 ibd 1
        # 0     0
        eq_( t._new2oldCode(1,1,0,0), (Ellipsis,0,0,1,1))




    def testnew2oldCode1(self):
        "Test new2oldCode()"
        t = SLRP.longRangePhase()
        # 1 ibd 0
        # 0     1
        #
        # 1 ibd 1
        # 0     0
        eq_( t._new2oldCode(2,1,2,0), (Ellipsis,0,0,2,1))

    def testnew2oldCode2(self):
        "Test new2oldCode()"
        t = SLRP.longRangePhase()
        # 11     0
        # 10 ibd 1
        #
        # 10 ibd 1
        # 00     0
        # new2oldCode( switchHet, sameHet, both share 1, used to share 0 1 haplotype)
        eq_( t._new2oldCode(2,1,0,2), (Ellipsis,2,2,2,1))

                                    


    def testnew2oldCode3(self):
        "Test new2oldCode()"
        t = SLRP.longRangePhase()
        # 11     1
        # 10 ibd 1
        #
        # 10 ibd 1
        # 00     1
        # new2oldCode( hom1, hom1, will share 0 1 haplotype, used to share 0 1 haplotype)
        eq_( t._new2oldCode(3,3,2,2), (Ellipsis,2,2,3,3))

                                    
    def testnew2oldCode4(self):
        "Test new2oldCode()"
        t = SLRP.longRangePhase()
        # 11     1
        # 10 ibd 0
        #
        # 10 ibd 1
        # 00     1
        # new2oldCode( same phase het, hom1, will share 0 1 haplotype, used to share 0 1 haplotype)
        eq_( t._new2oldCode(1,3,2,2), (Ellipsis,2,2,1,3))

                                    
    def testnew2oldCode4(self):
        "Test new2oldCode()"
        t = SLRP.longRangePhase()
        # 11     0
        # 10 ibd 1
        #
        # 10 ibd 1
        # 00     1
        # new2oldCode( swich het, hom1, will share 1 1 haplotype, used to share 0 1 haplotype)
        eq_( t._new2oldCode(2,3,0,2), (Ellipsis,2,2,2,3))

                                    
    def testnew2oldCode5(self):
        "Test new2oldCode()"
        t = SLRP.longRangePhase()
        # 11     0
        # 10 ibd 1
        #
        # 10 ibd 0
        # 00     0
        # new2oldCode( swich het, hom0, will share 1 1 haplotype, used to share 0 1 haplotype)
        eq_( t._new2oldCode(2,0,0,2), (Ellipsis,2,2,2,0))

    def testnew2oldCode6(self):
        "Test new2oldCode() Switch"
        t = SLRP.longRangePhase()
        # 11         0 new ibd
        # 10 old ibd 1
        #
        # 10 old ibd 1
        # 00         0 new ibd
        # new2oldCode( swich het, same het, will share 0 0 haplotype, used to share 0 1 haplotype)
        eq_( t._new2oldCode(c.hHetSw, c.hHetEq, c.ibd00,c.ibd01), (Ellipsis,c.p10,c.p01,c.h01,c.h10))

                                    
if __name__ == "__main__":
    apu = testSLRPlibImport()
    apu.testnew2oldCode4()
