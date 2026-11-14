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

This blind source separation task is fundamentally underdetermined, as infinitely many pairs of $(v, a)$ can sum to the identical mixture. For practical karaoke applications, the primary challenge lies in a performance trade-off: residual vocal leakage makes the accompaniment difficult to use, whereas overly aggressive suppression artifactually damages overlapping instrumental frequencies.

---

## Approach

### Baselines
Our baseline will be unsupervised NMF, this is because it models the spectral structure of music directly and it does not rely on the mixing assumptions which ICA depends on.
We discovered that NMF factorizes the magnitude spectrogram into spectral bases and their activations over time, and rebuilds the vocals and accompaniment by grouping components and applying soft masks.
We will also run ICA as a secondary baseline, in order to see where its assumptions of linear mixing and statistically independent sources will still break down on real music.
Other approaches, such as REPET and robust PCA, which assume a repeating or low-rank accompaniment, will also be included in our comparisons.

### Main Method: Supervised NMF with $L_1$ Regularization
Our main method will then be the supervised NMF with a separate spectral dictionary for each source.These two dictionaries are from isolated vocal and accompaniment stems, and we can give each dictionary component a known source label. We will compute magnitude spectrograms of the vocal and accompaniment stems and use NMF to learn one non-negative dictionary for each stem in training. At separation time, we will concatenate the two dictionaries and will keep them fixed, estimate the activations from the mixture alone, and combine each dictionary with its own activations to get a magnitude estimate for each source. These estimates then define Wiener-style soft masks, which we apply to the complex STFT of the mixture; we will keep the mixture phase and use an inverse STFT to reconstruct the vocals and accompaniment.

Besides, motivated by sparse representation, we also examine an L1 penalty on the activations. These dictionaries will cover many sounds before the regulation, a single frame typically needs only a few of them, and the penalty encourages the model to explain the frames with fewer atoms.

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
