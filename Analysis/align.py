import argparse
from tracking2 import align_frames

parser = argparse.ArgumentParser(description='Align the skeleton with respect to the reference skeleton')

parser.add_argument('skel_csv', help='Skeleton file to be aligned')
parser.add_argument('ref_csv', help='Reference frame to align the skeleton to')
parser.add_argument('-o', '--output-csv', default='skel_align.csv', help='output file')
parser.add_argument('-w', '--weights', default=None, help='file of joint weights')

args = parser.parse_args()

align_frames(args.skel_csv, args.ref_csv, args.output_csv, args.weights)