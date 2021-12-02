import os
import pandas as pd
import requests
from IPython.display import Markdown, display
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from six.moves import range
from sklearn import metrics
from sklearn.metrics import roc_curve
import numpy as np


import pickle
import hashlib

def url_to_path(
    url,
    base_dir = "./dataset",
    ext = "mp4"
):
    name = "_".join(url.split("/")[-2:])
    path = f"{base_dir}/{name}.{ext}"
    return path

def download_original_dataset(
    urls, 
    base_dir = "./dataset"
):
    for url in tqdm(urls):
        path = url_to_path(url, base_dir, "mp4")

        with open(path, "wb") as handle:
            response = requests.get(url, stream=True)
            for data in response.iter_content():
                handle.write(data)

                
def wav2vec_path(path):
    name = path.split("/")[-1].replace(".wav", ".p")
    pickle_path = f"./wav2vec/{name}"
    return pickle_path

def vgg_path(path):
    name = path.split("/")[-1].replace(".wav", ".p")
    pickle_path = f"./vgg/{name}"
    return pickle_path

def pickle_load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def pickle_dump(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)    
                
                
def convert_mp4_to_wav(src, dst, sr=16000):
    os.system(f'ffmpeg -i {src} -acodec pcm_s16le -ar 16000 {dst}')        
    

def printmd(string):
    display(Markdown(string))
    
def md5(path):
    with open(path, 'rb') as file:
        md5_hash = hashlib.md5()
        md5_hash.update(file.read())
        return md5_hash.hexdigest()
    
def open_wav(path):
    sr, signal = wavfile.read(path)    
    return signal

def tf_open_wav(path):
    file = tf.io.read_file(path)
    waveform, sr = tf.audio.decode_wav(file)
    return waveform

def read_wav(path):
    return tf.transpose(tf_open_wav(path))[0, :].numpy()


# https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
def compute_eer(y, y_score):
    """
    :param y: True binary labels in range {0, 1} or {-1, 1}. If labels are not binary, pos_label should be explicitly given
    :param y_score: Target scores, can either be probability estimates of the positive class, confidence values,
    or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
    :return: eer, thresh
    """
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh.item()


def compute_frr_far(tar, imp):
    tar_unique, tar_counts = np.unique(tar, return_counts=True)
    imp_unique, imp_counts = np.unique(imp, return_counts=True)
    thresholds = np.unique(np.hstack((tar_unique, imp_unique)))

    pt = np.hstack((tar_counts, np.zeros(len(thresholds) - len(tar_counts), dtype=np.int)))
    pi = np.hstack((np.zeros(len(thresholds) - len(imp_counts), dtype=np.int), imp_counts))

    pt = pt[np.argsort(np.hstack((tar_unique, np.setdiff1d(imp_unique, tar_unique))))]
    pi = pi[np.argsort(np.hstack((np.setdiff1d(tar_unique, imp_unique), imp_unique)))]

    fr = np.zeros(pt.shape[0] + 1, dtype=np.int)
    fa = np.zeros(pi.shape[0] + 1, dtype=np.int)

    for i in range(1, len(pt) + 1):
        fr[i] = fr[i - 1] + pt[i - 1]

    for i in range(len(pt) - 1, -1, -1):
        fa[i] = fa[i + 1] + pi[i]

    frr = fr / max(len(tar), 1)
    far = fa / max(len(imp), 1)

    thresholds = np.hstack((thresholds, thresholds[-1] + 1e-6))
    return thresholds, frr, far


def compute_eer_v2(tar, imp):
    tar_imp, frr, far = compute_frr_far(tar, imp)

    index_min = np.argmin(np.abs(frr - far))
    eer = 100.0 * np.mean((frr[index_min], far[index_min]))
    thr = tar_imp[index_min]

    return eer, thr


def plot_spec(path):
    import torch
    import torchaudio
    import matplotlib.pyplot as plt
    
    print(path)
    transform = torchaudio.transforms.Spectrogram(n_fft=300, normalized=True, return_complex=False)
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    spectrogram = transform(waveform).log2()[0, :, :]
    plt.xlabel("time")
    plt.ylabel("~freq")
    plt.imshow(spectrogram.numpy(), origin='lower')    
    
    
def load_json(path):
    import json
    with open(path) as json_file:
        return json.load(json_file)