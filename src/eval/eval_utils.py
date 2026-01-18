import numpy as np
from astropy.stats import jackknife
import soundfile as sf
import scipy

def load_audio(audio_path, dataset_sampling_rate):
    audiowav, _ = sf.read(audio_path)
    audiowav = audiowav.astype(np.float32)
    if len(audiowav.shape) > 1:
        audiowav = np.mean(audiowav, axis=-1)

    if dataset_sampling_rate != 16000:
        new_num_samples = round(audiowav.shape[-1]*float(16000)/dataset_sampling_rate)
        audiowav = scipy.signal.resample(audiowav, new_num_samples)

    return audiowav

def compute_retrieval_metric(indices, 
                             all_querys, all_keys, 
                             gt_query_key,
                             retrieval_type='at'):
    
    R1, R5, R10, mAP10 = [], [], [], []
    for i, query in enumerate(all_querys):
        pred_keys = [all_keys[idx] for idx in indices[i, :10]]

        if retrieval_type=='at':
            preds = []
            pred_key_temp = []
            for pred in pred_keys:
                if pred not in pred_key_temp and pred in gt_query_key[query]:
                    pred_key_temp.append(pred)
                    preds.append(True)
                else:
                    preds.append(False)

            preds = np.asarray(preds)

        elif retrieval_type=='ta':
            preds = np.asarray([gt_query_key[query] == pred for pred in pred_keys])
        
        # Given that only one correct audio file for each caption query
        R1.append(np.sum(np.any(preds[:1]), dtype=float))
        R5.append(np.sum(np.any(preds[:5]), dtype=float))
        R10.append(np.sum(np.any(preds[:10]), dtype=float))

        positions = np.arange(1, 11, dtype=float)[preds[:10] > 0]
        if len(positions) > 0:
            precisions = np.divide(np.arange(1, len(positions) + 1, dtype=float), positions)
            avg_precision = np.mean(precisions, dtype=float)
            mAP10.append(avg_precision)

        else:
            mAP10.append(0.0)
    
    # Jackknife estimation with 95% confidence interval on evaluation metrics
    estimate, bias, std_err, conf_interval = jackknife.jackknife_stats(np.asarray(R1), np.mean, 0.95)
    print("R1", f"{estimate:.3f}", f"[{conf_interval[0]:.3f}, {conf_interval[1]:.3f}]")

    estimate, bias, std_err, conf_interval = jackknife.jackknife_stats(np.asarray(R5), np.mean, 0.95)
    print("R5", f"{estimate:.3f}", f"[{conf_interval[0]:.3f}, {conf_interval[1]:.3f}]")

    estimate, bias, std_err, conf_interval = jackknife.jackknife_stats(np.asarray(R10), np.mean, 0.95)
    print("R10", f"{estimate:.3f}", f"[{conf_interval[0]:.3f}, {conf_interval[1]:.3f}]")

    estimate, bias, std_err, conf_interval = jackknife.jackknife_stats(np.asarray(mAP10), np.mean, 0.95)
    print("mAP10", f"{estimate:.3f}", f"[{conf_interval[0]:.3f}, {conf_interval[1]:.3f}]")