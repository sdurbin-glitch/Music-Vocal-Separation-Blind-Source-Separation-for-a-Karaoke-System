from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from NMF_train import NMF_train
from cluster_bases import cluster_bases

project_dir = Path(__file__).resolve().parent
# Load the mixed audio file
filename = project_dir / "mixture.wav"
speech, sr = librosa.load(filename, sr=None)

# Compute the STFT and magnitude spectrogram.
n_fft = 2048
hop_length = 256
spectrogram = librosa.stft(speech, n_fft=n_fft, hop_length=hop_length, center=False, win_length=n_fft)
M = np.abs(spectrogram)

# train unsupervised NMF model
n_components = 20
n_iter = 250
epsilon = 1e-10
rng = np.random.default_rng(0)
B_init = rng.random((M.shape[0], n_components))
W_init = rng.random((n_components, M.shape[1]))

B, W = NMF_train(M, B_init, W_init, n_iter)

# Review groups and then choose the vocal-like one manually.
n_groups = 6
labels = cluster_bases(B, W, n_clusters=n_groups)
group_estimates = [B[:, labels == cluster] @ W[labels == cluster, :]for cluster in range(n_groups)]

total_estimate = sum(group_estimates)
group_masks = [estimate / np.maximum(total_estimate, epsilon) for estimate in group_estimates]
group_audio = []
for cluster, mask in enumerate(group_masks):
    audio = librosa.istft(mask * spectrogram,hop_length=hop_length,win_length=n_fft,center=False)
    group_audio.append(audio)
    sf.write(project_dir / f"nmf_group_{cluster}.wav", audio, sr)

# vocal-like cluster. The accompaniment is reconstructed as the sum of every other group.
vocal_cluster = [2,3]
remaining_clusters = [cluster for cluster in range(n_groups) if cluster not in vocal_cluster]
accomp_mask = sum(group_masks[cluster] for cluster in remaining_clusters)

vocal_mask = sum(group_masks[cluster] for cluster in vocal_cluster)
vocal_audio = librosa.istft(vocal_mask * spectrogram,hop_length=hop_length,win_length=n_fft,center=False,)
accomp_audio = librosa.istft(accomp_mask * spectrogram,hop_length=hop_length,win_length=n_fft,center=False,)

#save the separated audio files as wav
sf.write(project_dir / "vocals_rec.wav", vocal_audio, sr)
sf.write(project_dir / "accomp_rec.wav", accomp_audio, sr)