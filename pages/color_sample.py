import streamlit as st
import os
import csv
import numpy as np


def read_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        color = [[int(v) for v in row] for row in reader]
        return color


def file_dir():
    filedir = os.listdir("color")
    for i in range(len(filedir)):
        filedir[i] = filedir[i].replace(".csv", "")
    filedir = tuple(filedir)
    return filedir


def pallet_write(col, index):
    r = pallet[index][0]
    g = pallet[index][1]
    b = pallet[index][2]
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    color[:, :] = [r, g, b]
    col.image(color)


fdir = file_dir()
color = st.selectbox("Select color pallet", fdir)
pallet = read_csv("./color/" + str(color) + ".csv")

for i in range(len(pallet) // 4 + min(1, len(pallet) % 4)):
    col1, col2, col3, col4 = st.columns(4)
    if i * 4 + 0 < len(pallet):
        pallet_write(col1, i * 4 + 0)
    if i * 4 + 1 < len(pallet):
        pallet_write(col2, i * 4 + 1)
    if i * 4 + 2 < len(pallet):
        pallet_write(col3, i * 4 + 2)
    if i * 4 + 3 < len(pallet):
        pallet_write(col4, i * 4 + 3)
