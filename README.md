# MultiLexScaled

#### Lexicon-based sentiment analysis, using multiple lexica and scaled against representative text corpora.

Easy-to-use, high-quality sentiment analysis. Instead of trying to develop yet another general-purpose sentiment analysis lexicon, we average across 8 widely-used ones that have different strengths and weaknesses. In addition, we calibrate against a set of representative texts and adjust each individual lexicon's score so that its mean is 0 (the neutral point) and the standard deviation is 1. We rescale the final average so that its standard deviation is 1 as well, to produce a sentiment measure that is readily interpretable (relative to the benchmark used for scaling).


### Citation

If using MultiLexScaled, please cite the _PLOS One_ paper in which it is introduced and validated (available as "van der Veen & Bleich 2024.pdf" in this repo):

van der Veen, A. Maurits, and Erik Bleich. 2024. "The advantages of lexicon-based sentiment analysis in an age of machine learning." _PLoS ONE_ 20(1): e0313092. https://doi.org/10.1371/journal.pone.0313092


_MultiLexScaled_ is designed to be easy to use through Jupyter notebooks, in the Anaconda python environment. It can easily be retooled to run from the command line or in some other environment.

This repo includes:
- code files to tokenize texts, calculate raw sentiment scores, and calibrate against a benchmark
- several notebooks with code to preprocess all the lexica used (and URLs for where to acquire them), perform sentiment analysis on a small set of texts or on large corpora, and generate new scalers
- scalers developed based on several benchmarks, which have been shown to work well in practice for a range of texts
- replication notebooks and datasets for the tables and figures in the associated paper

It is possible to generate a new benchmark from any representative corpus of texts. Note, however, that this must be a corpus for which it is reasonable to assume that the mean sentiment is neutral. When in doubt, it is probably preferable to use one of the supplied scalers.


### Installation & set-up

1. Download the repo
2. Run the notebook to pre-process the lexica (change to work with local folder paths)
3. Open the notebook(s) that apply sentiment analysis and, before running, adjust folder paths & filenames
4. To calibrate, either use one of the supplied scalers, or generate one based on your own representative corpus of choice.

### A note on the sentiment analysis notebooks

There are 3 sentiment analysis notebooks:
- `mark up individual texts` is designed to illustrate how sentiment is calculated for a small number of texts. It calculates a scaled sentiment score and displays marked-up text showing which words served as intensifiers/modifiers (words such `very`, `hardly`, etc.) and which words are captured by one or more sentiment lexica
- `pandas` is designed to read a corpus into a dataframe and do both the valence calculation and calibration on the dataframe, without generating intermediate files. It works fine for corpora up to about 50,000 texts (of 500-1,000 words length on average) but depending on memory and processor speed larger corpora are probably better handled with the `file-based` option
- `file-based` is designed to handle large corpora that need not fit into memory all at once. It also includes basic multiprocessing to make the processing somewhat faster (even on single-processor machines). It saves the results of intermediate stages (cleaned text, valences, calibrated valences for individual lexica) in separate files.

### A note on the available scalers/calibraters

Currently, 3 different scalers are included:
- US: based on 48,283 texts drawn from 17 representative US newspapers, over the period 1996-2015, by sampling from a random selection of dates any articles containing any of 17 different keywords labeled as neutral by the labMT lexicon
- UK: based on 59,404 texts drawn from 16 representative UK newspapers (including 7 additional associated Sunday papers) using the same sampling and selection procedure as for the US corpus
- USUK: based on the combination of the preceding 2 corpora (107,687 articles total)

Coming shortly, 2 additional scalers:
- CA: based on 22,860 texts drawn from 6 representative Canadian newspapers, using the same selection process
- AU: based on 24,114 texts drawn from 7 representative Australian newspapers, using the same selection process

More details on the selection process are in the working paper listed above. Newspaper titles are available upon request.

For most general purposes, either the US or the USUK scalers are probably best. For country-specific corpora from the UK, Canada, or Australia, it may be preferable to use those national scalers.

### Questions/comments/suggestions

Always welcome -- please email maurits@wm.edu


```python

```
