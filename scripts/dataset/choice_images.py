import argparse
import os
import re
from pathlib import Path
from shutil import copyfile, rmtree

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='choice image')
    parser.add_argument(
        '--data_root', type=str, default="", help='data_root')
    parser.add_argument(
        '--fps', type=int, default="5", help='fps')
    parser.add_argument(
        '--sec', type=int, default="5", help='second')
    parser.add_argument(
        '--delete', action='store_true', help='delete')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    count = 0
    for dirname, _, filenames in os.walk(args.data_root):
        # if any(dataset in dirname for dataset in ["center", "left", "right"]):
        for filename in filenames:
            num_list = re.findall('\d+', filename)
            file_num = num_list[0]
            if not int(file_num) % (args.fps * args.sec) == 0:
                if args.delete:
                    os.remove(os.path.join(dirname, filename))
            else:
                print(file_num)
                count += 1    
    print("file count = ", count)

if __name__ == '__main__':
    main()
