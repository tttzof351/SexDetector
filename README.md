# SexDetector

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
