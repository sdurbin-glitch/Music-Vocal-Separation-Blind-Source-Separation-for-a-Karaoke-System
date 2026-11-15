#!/usr/bin/env python3
import librosa
import museval
import numpy as np
import os
import soundfile as sf
import zipfile


class ICA:

    def fit(self, M):
        # input is m by T, m mixed signals sampled T times
        n, T = M.shape

        # first, center each signal
        mean_signal = np.mean(M,axis=1, keepdims=True)
        M_c = M - mean_signal

        # whiten the signals
        R = 1/T*M_c@M_c.T
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        C = np.diag(1/np.sqrt(eigenvalues))@eigenvectors.T
        ## another way
        #U, S, Vh = np.linalg.svd(M_c, full_matrices=False)
        #C = np.sqrt(T)*np.diag(1/S)@U.T
        X = C@M_c

        # calculate fourth moment
        # literally given in slides
        ## first attempt obviously won't fit in memory
        #gram = X.T@X
        #D = X@np.diag(np.diag(gram))@X.T
        gram_diag = np.sum(X**2, axis=0)
        D = (X * gram_diag)@X.T

        # diagonalize it
        # its symmetric so call the better eig func
        _, U = np.linalg.eigh(D)

        # calc unmixing
        A = U.T@C

        # do the unmixing
        H = A@M
        return H, A



if __name__ == "__main__":

    ica = ICA()
    with zipfile.ZipFile("musdb18hq.zip") as archive:
        root = zipfile.Path(archive)
        train = root/"train"
        test = root/"test"
        for dir1 in root.iterdir():
        # no training in vanilla ica
        #if dir1.name == "train" or dir1.name == "test":
        # save wavs in tmp to inspect, only 1 at a time for space reasons
            for dir2 in dir1.iterdir():
                accompany_list = []
                vocals = None
                for wav in dir2.iterdir():
                    with (dir2/f"{wav.name}").open("rb") as stream: 
                        # only need 2 channels for mixture
                        # "2 sources"
                        # otherwise load mono
                        audio, sr = librosa.load(stream, sr=None,
                                                 mono=(wav.name!="mixture.wav"))
                                                 
                        sf.write(f"/tmp/{wav.name}", audio.T, sr)
                        if wav.name == "vocals.wav":
                            vocals = audio
                    if wav.name == "mixture.wav":
                            H,A = ica.fit(audio)
                            sf.write(f"/tmp/source1.wav", H[0,:], sr)
                            sf.write(f"/tmp/source2.wav", H[1,:], sr)
                    else:
                        if wav.name != "vocals.wav":
                            accompany_list.append(audio)
                accompaniment = np.sum(accompany_list, axis=0)
                sf.write(f"/tmp/accompaniment.wav", accompaniment, sr)
                references = np.stack([vocals, accompaniment])

                # calc metrics
                SDR, _, SIR, SAR, permutation = museval.metrics.bss_eval(
                    references,
                    H,
                    window=sr,
                    hop=sr,
                    filters_len=512,
                    framewise_filters=False,
                    bsseval_sources_version=True,
                    compute_permutation=True,
                )
                SDR_median = np.nanmedian(SDR, axis=1)
                SIR_median = np.nanmedian(SIR, axis=1)
                SAR_median = np.nanmedian(SAR, axis=1)
                print(f"{SDR_median=}")
                print(f"{SIR_median=}")
                print(f"{SAR_median=}")
                #print(f"{SDR=}\n {ISR=}\n {SIR=}\n {SAR=}\n {permutation=}")
