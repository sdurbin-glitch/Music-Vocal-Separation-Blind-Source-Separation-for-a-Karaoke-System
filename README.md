# Music-Vocal-Separation-Blind-Source-Separation-for-a-Karaoke-System

\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue]{hyperref}

\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}

\title{\textbf{Project Proposal}\\[4pt]
\large Music Vocal Separation: Blind Source Separation for a Karaoke System}
\author{Group 4\\[2pt]
Emma Bai \quad Seth Durbin \quad Andrew Zigerelli}
\date{\today}

\begin{document}
\maketitle

%--------------------------------------------------------------
\section{Problem and Significance}

Karaoke and singing practice will often require instrumental accompaniments, but original stems are often unavailable.
We therefore study how to recover them from the mixture of music: given $x(t) = v(t) + a(t)$, we estimate the vocals $v(t)$ and the accompaniment $a(t)$ from $x(t)$ alone.
The problem then becomes underdetermined, since there are infinitely many pairs $(v, a)$ sum to the same mixture.
For karaoke, the practical difficulty is a trade-off: leftover vocals make the accompaniment hard to use, but suppressing them too aggressively will damage the instruments that overlap with the voice.

%--------------------------------------------------------------
\section{Proposed Approach}

\textbf{Baselines.}
Our baseline will be unsupervised NMF, this is because it models the spectral structure of music directly and it does not rely on the mixing assumptions which ICA depends on.
We discovered that NMF factorizes the magnitude spectrogram into spectral bases and their activations over time, and rebuilds the vocals and accompaniment by grouping components and applying soft masks.
We will also run ICA as a secondary baseline, in order to see where its assumptions of linear mixing and statistically independent sources will still break down on real music.
Other approaches, such as REPET and robust PCA, which assume a repeating or low-rank accompaniment, will also be included in our comparisons.

\textbf{Main method.}
Our main method will then be the supervised NMF with a separate spectral dictionary for each source.These two dictionaries are from isolated vocal and accompaniment stems, and we can give each dictionary component a known source label. We will compute magnitude spectrograms of the vocal and accompaniment stems and use NMF to learn one non-negative dictionary for each stem in training. At separation time, we will concatenate the two dictionaries and will keep them fixed, estimate the activations from the mixture alone, and combine each dictionary with its own activations to get a magnitude estimate for each source. These estimates then define Wiener-style soft masks, which we apply to the complex STFT of the mixture; we will keep the mixture phase and use an inverse STFT to reconstruct the vocals and accompaniment.

Besides, motivated by sparse representation, we also examine an L1 penalty on the activations. These dictionaries will cover many sounds before the regulation, a single frame typically needs only a few of them, and the penalty encourages the model to explain the frames with fewer atoms.

\textbf{Research focus.}

Our study will address the following questions:
\begin{enumerate}
    \item Will supervised NMF (with separate vocal, accompaniment dictionaries, and regulation) achieve better separation than unsupervised NMF and ICA?
    \item How will the number of spectral components in each dictionary affect the separation quality?
    \item Will encouraging fewer dictionary components to be active in each time frame improve separation quality?
\end{enumerate}

%--------------------------------------------------------------
\section{Dataset and Evaluation}

\textbf{Data.}
We will use MUSDB18-HQ \cite{musdb18hq}. This data set contains 150 full songs as 44.1 kHz stereo mixtures with vocals, drums, bass, and other stems. We will take the vocal stem as the vocal reference and the sum of the other three as the accompaniment reference. All audio files will be downmixed from stereo to mono. The STFT parameters for data set processing are FFT length = 2048, window size = 2048, hop length = 256, using Hann windowing. The train-test split will be 2:1.


\textbf{Metrics.}
We will evaluate full test songs using the standard python package museval (BSSEval v4) for computing frame-wise source-to-distortion, source-to-interference, and source-to-artifact ratios (SDR, SIR, SAR) to ensure reproducible benchmark scoring.  This is followed by a listening test.

%--------------------------------------------------------------
\section{Timeline and Team Roles}
% TODO: adjust to the course schedule and confirm division of labor.

\begin{tabular}{@{}lp{0.85\textwidth}@{}}
\toprule
\textbf{Week} & \textbf{Milestone} \\
\midrule
1 & Set up MUSDB18-HQ, the STFT pipeline, and SDR/SIR/SAR evaluation; run the unsupervised NMF and ICA baselines. \\
2 & Learn the vocal and accompaniment dictionaries and run supervised NMF separation with soft masks. \\
3 & Add the L1 sparsity penalty; tune dictionary size and penalty strength on validation songs. \\
4 & Run the final evaluation on the test songs, the listening test, and write the report. \\
\bottomrule
\end{tabular}

%--------------------------------------------------------------
\begin{thebibliography}{9}
\bibitem{musdb18hq} Z.~Rafii, A.~Liutkus, F.-R.~St\"oter, S.~I.~Mimilakis, and R.~Bittner,
  ``MUSDB18-HQ -- an uncompressed version of MUSDB18,'' Zenodo, 2019.
  \url{https://doi.org/10.5281/zenodo.3338373}
\end{thebibliography}

\end{document}
