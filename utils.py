import os, errno
import numpy as np
from datetime import datetime


def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

class Timer:
    def __init__(self):
        self.start_time = datetime.now()
        self.last_time = self.start_time

    def get_time_from_last(self, update=True):
        now_time = datetime.now()
        diff_time = now_time - self.last_time
        if update:
            self.last_time = now_time
        return diff_time.total_seconds()
    
    def get_time_from_start(self, update=True):
        now_time = datetime.now()
        diff_time = now_time - self.start_time
        if update:
            self.last_time = now_time
        return diff_time.total_seconds()
    

def is_paren(tok):
    return tok == ")" or tok == "("

def deleaf(tree):
    nonleaves = ''
    for w in tree.replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split()