import argparse
from tracking2 import process_video

parser = argparse.ArgumentParser(description='Smoothen the joint movements and output them as video and a csv file')
parser.add_argument("in_csv", help="file containing the joints")
parser.add_argument("-r", "--rgb-vid", default="colors.avi", help='name of the rgb video')
parser.add_argument("-d", "--depth-vid", default="depth.avi", help='name of the depth video')
parser.add_argument("-b", "--background", choices=['rgb', 'depth', 'none'], default='none',help="Background on which the \
	skeleton will be superimposed(default: none)")
parser.add_argument("-o", "--out-vid", default='output.avi', help='name of output video')
parser.add_argument("-oc", "--out-csv", default='output.csv', help='name of output csv file')
parser.add_argument("-st", "--start-frame", type=int, default=1, help='frame number to start from')
parser.add_argument("-end", "--end-frame", type=int, default=-1, help='frame number to end at')
parser.add_argument("-m", "--median", type=float, default=2, help='window size for the median filter(in seconds)')
parser.add_argument("-exp", "--exponential", type=float, default=0.5, help='half life for the exponential filter(in seconds)')
args = parser.parse_args()

process_video(args.in_csv, args.rgb_vid, args.depth_vid, args.background, args.out_vid, args.out_csv,
	args.start_frame, args.end_frame, args.median, args.exponential)