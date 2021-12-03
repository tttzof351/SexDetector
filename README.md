# SexDetector

1. [Data preparation](./notebooks/data_preparation.ipynb)
   1. Remove duplicates
   2. Download dataset
   3. Convert to 16KHz wav
   4. Remove empty files and duplicated md5
   5. Check correct filtering
   6. Incorrect files
2. [Analytics](./notebooks/analytics.ipynb)
    1. Duration
    2. Spectrogram
    3. NISQA
    4. Wav2Vec2-TSNE
    5. VGGISH-TSNE
3. [Train vgg-catboost](./notebooks/train_vgg_catboost.ipynb)
   1. Split train-validation-test
   2. VGG-CatBoost train
   3. 
4. 


- install: pip install -r requirements.txt
- research report: notebook.ipynb
- vgg_catboost_model: ~14% EER
- rawnet_model: ~17% EER


# Inference wav

```python
from inference import *
from utils import *

fn_rawnet_predict = build_rawnet_model()
fn_catboost_predict = build_vgg_catboost_model()

# WAV must be in SR=16KHz and more 3 sec and less 8 sec
waveform = read_wav(your_wav_file_path)

is_bad_rawnet, bad_prob_rawnet = fn_rawnet_predict(waveform, 16000)
is_bad_catboost, bad_prob_catboost = fn_catboost_predict(waveform, 16000)

print("is_bad_rawnet:", is_bad_rawnet, "; ~prob:", bad_prob_rawnet)
print("is_bad_catboost:", is_bad_catboost, "; ~prob:", bad_prob_catboost)
```

# Inference mp4/url

- see: notebook.ipynb::Test big file

# TODO:
- fusion models
- resnet
- augmentations
- add SAR databases
- scheduling/annealing lr
- cross validation
