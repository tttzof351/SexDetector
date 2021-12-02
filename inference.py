from utils import *

def build_vgg_catboost_model():
    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np
    from catboost import CatBoostClassifier
    
    vgg_model = hub.load('https://tfhub.dev/google/vggish/1')
    catbost_model = CatBoostClassifier()
    catbost_model.load_model("./weights/catboost.cbm")

    TARGET_SAMPLE_RATE = 16000
    MAX_LEN = 8
    MIN_LEN = 3
    SEGMENT_LEN = 5    
    
    THRESHOLD = 0.95403    
    BAD_MARKER = 0
    GOOD_MARKER = 1
    
    def align_emb(vgg):
        if vgg.shape[0] > SEGMENT_LEN:
            vgg = vgg[:SEGMENT_LEN, :]
        else:
            padding = np.zeros((SEGMENT_LEN - vgg.shape[0], 128))
            vgg = np.concatenate([vgg, padding])
        return vgg.flatten()
    
    def fn_predict(waveform, sr):
        assert sr == TARGET_SAMPLE_RATE
        assert len(waveform.shape) == 1
        
        if len(waveform) > TARGET_SAMPLE_RATE * MAX_LEN:
            raise Exception("segment too large")
        elif len(waveform) < TARGET_SAMPLE_RATE * MIN_LEN:
            raise Exception("segment too small")
        
        emb = vgg_model(waveform).numpy()
        emb = align_emb(emb)
        
        predict = catbost_model.predict_proba([emb])
        
        bad_predict = predict[0, BAD_MARKER]
        good_predict = predict[0, GOOD_MARKER]
        
        if good_predict > THRESHOLD:
            return 0, bad_predict
        else:
            return 1, bad_predict
    
    return fn_predict
    

def build_rawnet_model():
    import torch
    import numpy as np
    from models.model_builders import AudioClassificationModel
    
    rawnet_conf = load_json("./models/model_config.json")
    rawnet = AudioClassificationModel(**rawnet_conf)
    rawnet.load_state_dict(torch.load('./weights/rawnet_weights.pth'))
    
    rawnet.eval()
    rawnet.training = False
    rawnet.requires_grad = False
    
    TARGET_SAMPLE_RATE = 16000
    MAX_LEN = 8
    MIN_LEN = 3
    SEGMENT_LEN = 5    
    
    THRESHOLD = 0.68858    
    BAD_MARKER = 0
    GOOD_MARKER = 1
    
    def crop(waveform, sec_len = 5, sr: int = 16000, add_batch_dim: bool = False):
        waveform = waveform[0, :]

        ds_len = sr * sec_len

        if waveform.shape[0] > ds_len:
            waveform = waveform[:ds_len] 
        elif waveform.shape[0] < ds_len:
            padding = torch.zeros(ds_len - waveform.shape[0])
            waveform = torch.cat((waveform, padding), 0)

        waveform = torch.unsqueeze(waveform, 0)
        if add_batch_dim:
            waveform = torch.unsqueeze(waveform, 0)

        return waveform
    
    def fn_predict(waveform, sr):        
        assert sr == TARGET_SAMPLE_RATE
        assert len(waveform.shape) == 1        

        if len(waveform) > TARGET_SAMPLE_RATE * MAX_LEN:
            raise Exception("segment too large")
        elif len(waveform) < TARGET_SAMPLE_RATE * MIN_LEN:
            raise Exception("segment too small")                    
        
        waveform = np.expand_dims(waveform, 0)
        waveform = torch.from_numpy(waveform)
        
        waveform = crop(waveform, sec_len = SEGMENT_LEN, sr = TARGET_SAMPLE_RATE, add_batch_dim=True)
        predict = rawnet.forward_softmax(waveform).detach().numpy()
        
        bad_predict = predict[0, BAD_MARKER]
        good_predict = predict[0, GOOD_MARKER]
        
        if good_predict > THRESHOLD:
            return 0, bad_predict
        else:
            return 1, bad_predict
        
        
    return fn_predict