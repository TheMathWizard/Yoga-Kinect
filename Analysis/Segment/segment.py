import argparse
from segment2 import segment_time

parser = argparse.ArgumentParser(description='Divide the yoga period into three parts, \
	before posture, in-posture and after posture')
parser.add_argument('in_csv', help='file containing the joints(preferably smoothed)')
parser.add_argument('ref_csv', help='file containing the reference positions')
#parser.add_argument('-d', '--degrees', type=int, choices=[1,2,3], default=3, help='degrees of freedom for the \
#	affine transform')
parser.add_argument('-o', '--output-csv', default='segments.csv',help='file to write data to')

args = parser.parse_args()

segment_time(args.in_csv, args.ref_csv, args.output_csv)
