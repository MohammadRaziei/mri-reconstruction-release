import numpy as np
import json
import mridata
import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default="data", help='Choose path where downloaded on')
args = parser.parse_args()

pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)

tables = pd.read_csv('mridata.csv')

for i in range(0,100):
    print("\n=============\n>> download :", i, "th")
    mridata.download(tables.iloc[i]['UUID'], folder=args.outdir)









