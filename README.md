# Music Vocal Separation: Blind Source Separation for a Karaoke System

[![Python 3.8+](https://shields.io)](https://python.org)
[![Evaluation](https://shields.io)](https://github.com)
[![Dataset](https://shields.io)](https://zenodo.org)

**Group 4 Project Proposal**  
*Contributors: Emma Bai, Seth Durbin, Andrew Zigerelli*

---

## Problem and Significance

Karaoke and singing practice frequently require instrumental accompaniments, but original studio stems are often unavailable. We investigate how to recover these components from a mixed audio signal.

Given a mixture:
$$x(t) = v(t) + a(t)$$

Our goal is to isolate:
*   $v(t)$ : The vocal track
*   $a(t)$ : The accompaniment track

This blind source separation (BSS) task is fundamentally underdetermined, as infinitely many pairs of $(v, a)$ can sum to the identical mixture. For practical karaoke applications, the primary challenge lies in a performance trade-off: residual vocal leakage makes the accompaniment difficult to use, whereas overly aggressive suppression artifactually damages overlapping instrumental frequencies.

---

## Approach

### Baselines
*   **Unsupervised NMF:** Our primary baseline. Non-negative Matrix Factorization is chosen because it models the spectral structure of music directly and bypasses the strict mixing assumptions required by Independent Component Analysis (ICA). It factorizes a magnitude spectrogram into spectral bases and their corresponding time-varying activations.
*   **ICA:** Independent Component Analysis will serve as a secondary baseline to evaluate where its assumptions of linear mixing and statistical independence fail on real-world musical mixtures.
*   **Structural Baselines:** We will also include alternative methods such as REPET and Robust PCA—which assume repeating or low-rank structures in the accompaniment—in our performance comparisons.

### Main Method: Supervised NMF with $L_1$ Regularization
Our core approach leverages supervised NMF utilizing a separate, pre-trained spectral dictionary for each underlying source.

1.  **Training:** We compute magnitude spectrograms of isolated vocal and accompaniment stems to learn a distinct, non-negative dictionary for each source.
2.  **Separation:** At separation time, these two dictionaries are concatenated and held fixed. We estimate the activation matrices from the mixture signal alone. 
3.  **Masking:** Combined dictionary-activation magnitude estimates define Wiener-style soft masks, which are applied to the complex Short-Time Fourier Transform (STFT) of the mixture. We retain the original mixture phase and apply the inverse STFT (iSTFT) to reconstruct time-domain tracks.
4.  **Sparsity Constraints:** Motivated by sparse representation theory, we incorporate an $L_1$ regularization penalty on the activation matrices. Because these pre-trained dictionaries span a wide array of spectral profiles, a single audio frame should theoretically require only a few basis vectors; the penalty encourages the model to explain frames using a sparse subset of atoms.

### Research Focus
1.  Will supervised NMF (incorporating distinct vocal and accompaniment dictionaries with $L_1$ regularization) yield superior separation performance compared to unsupervised NMF and ICA baselines?
2.  How does the number of spectral components specified in each dictionary affect the overall separation quality?
3.  Does enforcing temporal sparsity on the active dictionary components in each frame improve source isolation and reduce audio artifacts?

---

## Dataset and Evaluation

### Data Pipeline
We utilize the **MUSDB18-HQ** dataset, featuring 150 full-length tracks provided as uncompressed 44.1 kHz stereo mixtures alongside isolated stems.

*   **Vocal Target:** Isolated vocal stem.
*   **Accompaniment Target:** Linear sum of drums, bass, and "other" stems.
*   **Pre-processing:** All audio tracks are downmixed from stereo to mono.

| Parameter | Value |
| :--- | :--- |
| **FFT Length** | 2048 |
| **Window Size** | 2048 |
| **Hop Length** | 256 |
| **Window Function** | Hann |
| **Train-Test Split** | 2:1 |

### Metrics
*   **Objective Metrics:** We evaluate full test songs using the standardized `museval` (`BSSEval v4`) Python package. This benchmark computes frame-wise:
    *   Source-to-Distortion Ratio (**SDR**)
    *   Source-to-Interference Ratio (**SIR**)
    *   Source-to-Artifact Ratio (**SAR**)
*   **Subjective Metrics:** Quantitative benchmarks will be supplemented by a subjective listening test.

---

## References

*   Rafii, Z., Liutkus, A., Stöter, F.-R., Mimilakis, S. I., & Bittner, R. (2019). *MUSDB18-HQ -- an uncompressed version of MUSDB18*. Zenodo. [https://doi.org](https://doi.org)
