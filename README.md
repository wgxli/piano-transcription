This is a deep-neural-network based piano transcription tool I created back in summer 2017.
I was invited to present this research at the [2017 Joint Mathematics Meetings](https://jointmathematicsmeetings.org/meetings/national/jmm2017/2180_program_thursday.html).

![Image of probability piano-roll](https://github.com/wgxli/piano-transcription/raw/master/evaluation-1.png)

Requirements
------------
The transcriber requires the following libraries on Python 3, and all their dependencies:

* NumPy
* Tensorflow
* Keras
* soundfile
* resampy
* music21

Usage
-----
To run the system, execute the following command:

`python3 onsets.py '/path/to/audio.ogg'`

The onset times of detected notes will be printed to `stdout`.

Performance
-----------
Under the standard `mir_eval` benchmarks, the system consistently achieves an 80% f-measure
on the MAPS piano dataset,
which, to the best of my knowledge,
was on par with general (i.e. no per-piano fine-tuning)
state-of-the art systems at the time.

I submitted this system to the
[2017 MIREX music information retrieval contest](https://www.music-ir.org/mirex/wiki/2017:Multiple_Fundamental_Frequency_Estimation_%26_Tracking_Results_-_MIREX_Dataset).
Unfortunately, it did worse than anticipated, with an f-measure of 36%.
I suspect this was due to unanticipated audio normalization issues.
I did not have the time or resources to develop the system further,
but I hope to continue development at some point.

How it Works
------------
The audio file is first converted to a time-frequency representation
via the constant-Q tranform (CQT).
We use a time resolution of 20 frames per second,
with 36 bins per octave and a Q-factor of 96.

The input to the neural network is
a 16-frame window of the CQT magnitude spectrum.
The network outputs a vector of 88 probabilities for
the likelihood of a note onset at the central frame.
We then use the probability piano-roll
to estimate note onsets with some simple post-processing.

The network was trained on a procedurally generated dataset
of random chords, synthesized with a variety of soundfonts.
This reduces overfitting dramatically compared to traditional approaches (training over a curated dataset).

More details are available in the [abstract](https://www.music-ir.org/mirex/abstracts/2017/SL1.pdf) for my MIREX submission.
The dataset is described in more detail (with an older version of the DNN model)  in my [paper](https://arxiv.org/abs/1707.08438).
Please keep in mind that I wrote these papers in high school;
looking back now, my writing style could use some significant improvements.
