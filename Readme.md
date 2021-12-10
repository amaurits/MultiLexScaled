# MultiLexScaled

#### Lexicon-based sentiment analysis, using multiple lexica and scaled against representative text corpora.

Easy-to-use, high-quality sentiment analysis. Instead of trying to develop yet another general-purpose sentiment analysis lexicon, we average across 8 widely-used ones that have different strengths and weaknesses. In addition, we calibrate against a set of representative texts and adjust each individual lexicon's score so that its mean is 0 (the neutral point) and the standard deviation is 1. We rescale the final average so that its standard deviation is 1 as well, to produce a sentiment measure that is readily interpretable (relative to the benchmark used for scaling).

_MultiLexScaled_ is designed to be easy-to-use through Jupyter notebooks, in the Anaconda python environment. It can easily be retooled to run from the command line or in some other environment.

This repo includes:
- codefiles to tokenize texts, calculate raw sentiment scores, and calibrate against a benchmark
- several notebooks with code to preprocess all the lexica used (and URLs for where to acquire them), perform sentiment analysis on a small set of texts or on large corpora, and generate new scalers
- scalers developed based on several benchmarks, which have been shown to work well in practice for a range of texts

It is possible to generate a new benchmark from any representative corpus of texts. Note, however, that this must be a corpus for which it is reasonable to assume that the mean sentiment is neutral. When in doubt, it is probably preferable to use one of the supplied scalers.


### Installation & set-up

1. Download the repo
2. Run the notebook to pre-process the lexica (change to work with local folder paths)
3. Open the notebook(s) that apply sentiment analysis and, before running, adjust folder paths & filenames
4. To calibrate, either use one of the supplied scalers, or generate one based on your own representative corpus of choice.

### Future changes

In the near future, the code will be made more pandas-friendly, so that it can be run on a pandas dataframe without generating several different output files.


### Questions/comments/suggestions

Always welcome -- please email maurits@wm.edu


```python

```
