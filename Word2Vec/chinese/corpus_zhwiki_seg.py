import os
import sys
import logging
import jieba
from optparse import OptionParser


def seg_with_jieba(infile, outfile):
    '''segment the input file with jieba'''
    with open(infile, 'r', encoding='utf-8') as fin, open(outfile, 'w', encoding='utf-8') as fout:
        i = 0
        for line in fin:
            seg_list = jieba.cut(line)
            seg_res = ' '.join(seg_list)
            fout.write(seg_res)
            i += 1
            if i % 1000 == 0:
                logger.info('handing with {} line'.format(i))
        return i


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(program)
    logger.info('running ' + program + ': segmentation of corpus by jieba')

    # parse the parameters
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='infile', default='data/corpus.zhwiki.simplified.done.txt', help='input file to be segmented')
    parser.add_option('-o', '--output', dest='outfile', default='data/corpus.zhwiki.segwithb.txt', help='output file segmented')
    options, args = parser.parse_args()
    infile = options.infile
    outfile = options.outfile

    try:
        i = seg_with_jieba(infile, outfile)
        logger.info('i=%d words, segment the infile finished' % i)
    except Exception as err:
        logger.info(err)
