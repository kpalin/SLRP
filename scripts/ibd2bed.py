#!/usr/bin/env python
#-*- coding: utf-8 -*-

""" 
Covert SLRP output ibd file to more reasonable bed like tsv format
Created at Thu Oct  3 14:34:39 2013 by kpalin
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert SLRP output ibd file to more reasonable bed like tsv format")
    
    parser.add_argument("-i", "--ibd",
                        help="ibd file [default:%(default)s]",
                        required=True)

    parser.add_argument("-v", "--vcf",
                        help="vcf used with SLRP (preferably output) [default:%(default)s]",
                        required=True)
    parser.add_argument("-c", "--chrom",
                        help="Chromosome used with SLRP (Default is the first chromosome in the VCF file) [default:%(default)s]",
                        default=None)
    
    parser.add_argument("-V", "--verbose",default=False,action="store_true",
                        help="Be more verbose with output [default:%(default)s]" )
    
    args = parser.parse_args()
    
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    from gzip import GzipFile
    if args.ibd.endswith(".gz"):
        inIBD = GzipFile(args.ibd)
    else:
        inIBD = open(args.ibd)

    if args.vcf.endswith(".gz"):
        inVCF = GzipFile(args.vcf)
    else:
        inVCF = GzipFile(args.vcf)

    from itertools import dropwhile
    CHROMit = dropwhile(lambda x:x.startswith("##"), inVCF)
    header = CHROMit.next()
    assert header.startswith("#CHROM"), "File %s doesn't look like VCF file. Missing #CHROM on '%s'"%(args.vcf,header[:80])
    
    header = header.strip().split("\t")
    samples = header[9:]

    if args.chrom is None:
        CHROM = CHROMit.next().split("\t",2)[0]
    else:
        CHROM = args.chrom

    import sys
    outStrm = sys.stdout
    logging.info("Coversion will lose information about indecies of start and end markers")
    outStrm.write("#CHROM\tSTART\tSTOP\tSAMPLE1,SAMPLE2\tSCORE\tHAPLO1,HAPLO2\n")
    for line in inIBD:
        hap1,hap2,score,bginMark,endMark,bginBP,endBP = line.strip().split()
        hap1,hap2 = int(hap1),int(hap2)
        sample1,sample2 = samples[hap1/2],samples[hap2/2] 
        hapIdx1,hapIdx2 = hap1%2, hap2%2
        outStr="%s\t%s\t%s\t%s,%s\t%s\t%d,%d\n"%(CHROM,bginBP,endBP,sample1,sample2,score,hapIdx1,hapIdx2)
        outStrm.write(outStr)

    return args
    
    
if __name__ == '__main__':
    args=main()
    	
