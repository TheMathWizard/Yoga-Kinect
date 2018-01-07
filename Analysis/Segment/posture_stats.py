import argparse
from segment2 import calc_stats

parser = argparse.ArgumentParser(description='Calculate and write the posture stats to a file')
parser.add_argument('skel_csv', help='Skeleton for which to calculate the stats')
parser.add_argument('ref_csv', help='Reference frame against which to calculate stats')
parser.add_argument('-w', '--weights', default=None, help='file containing joint weights')
parser.add_argument('-e', '--error-data', default='error.csv', 
	help='Error data of population wrt the reference frame')
parser.add_argument('-o', '--output', default='out.csv', help='output file')
args = parser.parse_args()

calc_stats(args.skel_csv, args.ref_csv, args.weights, args.error_data, args.output)




