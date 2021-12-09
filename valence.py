# encoding: utf-8

# valence.py
# by A. Maurits van der Veen

# Modification history:
# 2018-10-15: 2to3, floor division, 'w' vs 'wb'
# 2019-02-26: newline = '' added to csv output files
# 2019-07-26: new sentiment analysis structure
# 2021-11-25: general clean-up and reorganization
# 2021-11-26: extract calibration functions to calibrate.py
# 2021-12-01: clean up an re-organize

# Code to calculate positive/negative polarity of texts using multiple sentiment analysis lexica

# More simply, can also be used to assess the relative prevalence of any other word category
# represented by a lexicon, but is probably overkill for that purpose.

# **************************** Main valence calculation function **************************

def getValence(fulltext, lexiconlist, wildlexicon=[], wild='*',
               wordlist=set(), ignore=(), skip=(),
               modifiers={},
               negaters=("not", "no", "n't", "neither", "nor", "nothing", "never", "none",
                         "lack", "lacked", "lacking", "lacks", "missing", "without", "absence", "devoid"),
               stopwords=('a', 'an', 'and', 'the', 'to', 'as'),
               separator='', scaling='words',
               ignorex=True, flagwords=False,
               need2tokenize=False, makelower=False,
               skippunct = True,
               textnr=-1, chunknr=0, updateinterval=100000):
    """Calculate valences for a text, optionally using modifiers or negaters.

    fulltext should contain a text, preprocessed for optimal use with the lexica
    supplied (this includes tokenization, unless need2tokenize is set to True).

    lexiconlist contains a list of lexica to use. For each lexicon a separate
    valence will be calculated. Lexica should be dictionaries of the form {word: valence, ...}.
    They may have wild-cards (specified with the 'wild' parameter).

    wildlexicon contains, for every lexicon in lexiconlist, a flag indicating whether
    it contains wildcards. If wildlexicons is not specified, it will be calculated here,
    but for corpus-level valence calculations it is more efficient not to re-calculate it each time.

    wordlist contains a list of all words to look up in any lexicon. In theory it contains
    the union of all words sets across the lexica in lexiconlist. If it is not pre-specified,
    it will be generated in here. For corpus-level valence calculations, it only needs to
    be calculated once, so it is more efficient to calculate in advance and pass it in,
    The list can also be used to remove words from consideration in the valence calculation,
    even if they are in one or more lexica.

    ignore and skip are lists of words to ignore in the lexicon calculation. They have the same
    effect, skip is generally used for "words" that are not really words (punctuation, etc.),
    while ignore is used for words that are actual words but we want to skip (keywords etc.).
    Ignored words are included in the length-based scaling of valences; skipped words are not.
    skippunct is a separate parameter that makes us treat all characters in string.punctuation
    as members of skip.

    modifiers, if supplied, should have the dictionary form {modifier: mod_fraction, ...}
    Modifiers are applied using the SO-CAL method (see Taboada 2011); using the SO-CAL
    modifier data is therefore highly recommended. Multi-word modifying phrases should be
    connected by underscores, not spaces (i.e. "a_little" not "a little").
    Note: the modifying effect of a word trumps its independent valence: any modifiers that are
    also in a valence lexicon will get ignored as valence words.

    megaters are a basic set of negation words (these are not in the SO-CAL modifier list).
    The default is the standard SO-CAL list.

    stopwords is a list of words to skip over without clearing out any accumulated intensification/
    modification information

    separator is a character/string to split on, turning this function into a valence calculator
    for a set of texts. Default is '' (empty string), for a single-text calculation

    ignorex applies to result of Lexicoder-style preprocessing, which convert 'well' and 'okay'
    to 'xwell' and 'xokay' in constructions where they are likely not to be valenced, and add
    'plusx' and 'minusx' as modifiers. Setting ignorex to True removes those 'x' prefixes; setting
    it to False keeps them and also adds 'plusx' and 'minusx' to the modifier list, if not already
    included.

    flagwords specifies whether to return information about which words were involved in the valence
    calculation, either as valence-carrying words or as modifiers

    need2tokenize can be used to tokenize a text on the fly, using nltk's word tokenizer
    makelower can be used to make a text lowercase on the fly.
        Both need2tokenize and makelower are better implemented at the corpus level

    scaling specifies whether and how to scale the calculated valence. Default is "words", which
    specifies dividing the calculated valence by the length (in words) of the text. The alternative
    is 'separator' and is only meaningful if the text is split into multiple units (e.g. sentences).

    textnr, chunknr, and updateinterval are used to display progress updates; useful for keeping track
    of progress if this function is invoked in a list comprehension or a parallel processing set-up.
    """
    import itertools
    import string
    if need2tokenize:
        from nltk.tokenize import word_tokenize

    # Progress update (useful to keep track if invoking in list comprehension, for example)
    if textnr % updateinterval == 0:
        print("Processing text {}{}".format(textnr, '' if chunknr == 0 else (" of chunk {}".format(chunknr))))

    nrlexica = len(lexiconlist)
    if len(wildlexicon) == 0:
        wildlexicon = [haswilds(lexicon, wild=wild) for lexicon in lexiconlist]

    # Add basic modifiers from Lexicoder-style language preprocessing, if modifiers will be used.
    if not ignorex and len(modifiers) > 0:
        if 'minusx' not in modifiers:
            modifiers['minusx'] = -0.5
        if 'plusx' not in modifiers:
            modifiers['plusx'] = 1

    # Check if we need to process subunit by subunit
    if separator != '':
        texts = fulltext.split(separator)
    else:
        texts = [fulltext,]
    textvalences = []

    for text in texts:

        valences = [0, ] * (nrlexica)
        hitdict = {}
        modifier = 1  # Start out with no valence modification
        modlist,  modifiersused = [], []

        # Preprocess text as needed
        if makelower:
            text = text.lower()
        textwords = word_tokenize(text) if need2tokenize else text.split()
        if ignorex:  # remove the 'x' from 'xwell' and 'xokay', which may be introduced in Lexicoder-style preprocessing
            textwords = [x if x not in ('xokay', 'xwell') else x[1:] for x in textwords]

        nrwords = len(textwords)
        wordcount_adj = 0

        for count, word in enumerate(textwords):

            # If word is a to-be-skipped item (punctuation, separator, etc.), adjust word count and reset modifier
            # Note: this set-up means we can not have modifier phrases that include to-be-skipped items!
            if word in skip or (skippunct and word in string.punctuation):
                modifier = 1
                wordcount_adj += 1
                modifiersused = []
                continue

            # Make sure word was not part of a modifier already handled
            if count not in modifiersused:

                # Handle modifiers, incl. multi-word modifying phrases (with underscores in our dictionary)
                # Longer phrases trump shorter ones; none is longer than 4 words
                if count < nrwords - 3:
                    wordx = '_'.join(textwords[count:count + 4])
                    if wordx in modifiers:
                        modifier *= 1 + modifiers[wordx]
                        modifiersused += list(range(count, count + 4))
                        continue
                if count < nrwords - 2:
                    wordx = '_'.join(textwords[count:count + 3])
                    if wordx in modifiers:
                        modifier *= 1 + modifiers[wordx]
                        modifiersused += list(range(count, count + 3))
                        continue
                if count < nrwords - 1:
                    wordx = '_'.join(textwords[count:count + 2])
                    if wordx in modifiers:
                        modifier *= 1 + modifiers[wordx]
                        modifiersused += [count, count + 1]
                        continue
                if word in modifiers:
                    modifier *= 1 + modifiers[word]
                    modifiersused.append(count)
                    continue

                # Check for negation next
                # Note: we get here only if no modifiers caught
                # Note 2: Taboada et al. 2011 use a shift of 4 in the opposite direction
                #         but we use a multiplier of -0.5 because not all lexica run from -5 to 5
                if word in negaters:
                    modifier *= -0.5  # equivalent to (1 + -1.5) in the Taboada set-up
                    modifiersused.append(count)
                    continue

                # Look up valences; multiply by modifier value
                # Note 1: we get here only if neither modifiers nor negaters caught
                # Note 2: we could simply skip wordinset test and always query each lexicon separately,
                #         but this is faster.
                if wordinset_wild(word, wordlist, wild) and not wordinset_wild(word, ignore, wild):
                    lexmatches = [lexiconmatch_wild(word, lexicon, wild) if lexwild \
                                      else lexiconmatch(word, lexicon) \
                                  for lexicon, lexwild in zip(lexiconlist, wildlexicon)]

                    # Track number of lexicon hits as needed (note: works correctly only if nr lexica < 10)
                    # For more than 10 lexica, either change to 0.01, or else stop tracking positive & negative separately)
                    if flagwords:
                        wordhits = sum([1 if x > 0 else (0.1 if x < 0 else 0) for x in lexmatches])
                        # Note: Hits could cancel out for words that are positive and negative in different lexica
                        if wordhits != 0:
                            # Flag both word and modifiers, as appropriate
                            hitdict[count] = wordhits
                            # Note: could add direction flag to combined modifier phrase too,
                            # but that's a bit more than needed right now
                            modlist += modifiersused

                    # Update sentence valences, applying modifier
                    valences = [x + y for x, y in zip(valences, [modifier * x for x in lexmatches])]

                # Finally, reset modifier, unless this was a stopword
                # Note: this means stopwords should not be in our lexica!
                if word not in stopwords:
                    modifier = 1
                    modifiersused = []

        # Store text length & valences for text
        textvalences.append([nrwords - wordcount_adj,] + valences)

    # Aggregate across sub-units
    valencesums = [sum(val) for val in zip(*textvalences)]

    # Scale by word length or number of subunits
    if scaling == 'words':
        wordcount = valencesums[0]
        valencesumsAdj = valencesums if wordcount == 0 else \
            [wordcount, ] + [vsum / float(wordcount) for vsum in valencesums[1:]]
    else:  # scaling == 'subunits':
        nrunits = len(textvalences)
        valencesumsAdj = [vsum / float(nrunits) for vsum in valencesums]

    # Return aggregated value and subunit values, as appropriate
    # Note that subunit values are unscaled!
    if separator == '':  # Just return single, aggregated value, optionally with hit data
        if flagwords:  # Return aggregated value, plus info about words flagged (and how many times)
            # Construct result text by adding info about modifier status and number of lexica hit in parentheses
            flaggedtext = []
            for counter, word in enumerate(textwords):
                if counter in hitdict:
                    hitinfo = hitdict[counter]
                    neghits = int(10 * (hitinfo % 1))
                    infostring = '-' + str(neghits) if neghits > 0 else ''
                    poshits = int(hitinfo)
                    posstring = '+' + str(poshits) if poshits > 0 else ''
                    if len(posstring) > 0:
                        if len(infostring) > 0:
                            infostring += ','
                        infostring += posstring
                    flaggedtext += [word, '(' + infostring + ')']
                elif counter in modlist:
                    flaggedtext += [word, '(*)']
                else:
                    flaggedtext.append(word)
            return valencesumsAdj, ' '.join(flaggedtext)
        else:
            return valencesumsAdj
    else:  # Also return individual unit values, flattened together into a single list
        return valencesumsAdj + list(itertools.chain.from_iterable(textvalences))


# **************************** Corpus-level processing **************************

def calc_corpus_valence(corpusfile, valencefile, lexnames, lexiconlist, mods,
                        idcol=0, textcols=(1,), modify=True, negaters=(), ignore=(), skip=(),
                        subunit_valencefile='', separator=' . ',
                        header=False, wild='*',
                        need2tokenize=False, makelower=False, skippunct=True,
                        nrjobs=1, texts_per_job=250000, multiplerowspertext=False):
    """Calculate text-level valences for entire corpus; optionally save subunit-level valence as well.

    valencefile (the output) will contain the data in idcol, plus a word count column,
        plus the individual valences calculated.

    Ignore valence of any terms passed in through ignore parameter, but do count in word count (for scaling)
    Skip any terms in skip parameter, and don't count in word count (used for punctuation &c.)

    For very large files, this can be memory-intensive. Two ways to minimize the load:
    - with nrjobs=1, use 'yield' (in the function yieldtext) to load only 1 line at a time
    - with nrjobs>1, run the main valence computation in multiple rounds

    TODO: convert this function to an i/o wrapper around pandas dataframe-based operation
    (as is already done in calibrate.py).
    """
    import csv
    import gc
    import multiprocessing as mp
    from functools import partial
    from operator import itemgetter

    csv.field_size_limit(1000000000)

    # Ignore separator if no subunit file is specified
    if subunit_valencefile == '':
        separator = ''

    # Generate a union of lexiconkeys, using any wildcards included to keep length down
    allterms = allkeys(lexiconlist)
    # See which of our ignore terms are not in allterms, so need to be ignored separately in getValence
    ignoreX = set(ignore) - allterms
    # Update allterms to ignore words in our ignore set
    allterms -= set(ignore)

    # Add a flag indicating whether a lexicon has wildcards
    wildlexicon = [haswilds(lex) for lex in lexiconlist]
    nroutputs = 1 + len(lexnames)  # 1 for wordcount

    if nrjobs == 1:
        # This is an elegant list comprehension and does not require reading the whole corpus at once,
        # but it is not parallelizable and hence slower
        valences = [[id,] + getValence(text, lexiconlist, wildlexicon, wild,
                                    allterms, ignoreX, skip,
                                    mods if modify else dict(),
                                    negaters, separator=separator,
                                    need2tokenize=need2tokenize, makelower=makelower, skippunct=skippunct,
                                    textnr=count, updateinterval=5000) \
                    for count, (id, text) in enumerate(yieldtext(corpusfile, capsmatter=True,  # use makelower inside getValence if needed
                                                                 idcol=idcol, textcols=textcols, header=header))]

    else:  # parallelize -> read in entire corpus first

        corpusdata = []
        with open(corpusfile, 'r', errors='ignore') as inf:
            inreader = csv.reader(inf)
            if header:
                headerdata = next(inreader)
            for row in inreader:
                corpusdata.append([row[idcol], ' '.join(row[x] for x in textcols)])
        nrtexts = len(corpusdata)
        # imdb hardcoding
        if nrtexts > 50000:
            corpusdata = corpusdata[:50000]
            nrtexts = 50000

        # First calculate number of rounds
        rounds = 1 + nrtexts // (nrjobs * texts_per_job)  # at least 1 round!
        jobs_rounds = rounds * nrjobs
        chunksize = 1 + ((nrtexts - 1) // jobs_rounds)
        print("Run in {} jobs; {} round{} per job; {} texts per jobround.".format(
              nrjobs, rounds, '' if rounds == 1 else 's', chunksize))

        # Divide corpus into chunks.
        textchunks = [corpusdata[x * chunksize: (x + 1) * chunksize] for x in range(jobs_rounds)]

        # Define partial function to match with chunks
        partial_getValences = partial(getValences, lexiconlist=lexiconlist,
                                      wildlexicon=wildlexicon, wild=wild,
                                      wordlist=allterms, ignore=ignoreX, skip=skip,
                                      mods=mods if modify else dict(),
                                      negs=negaters, separator=separator,
                                      need2tokenize=need2tokenize, makelower=makelower,
                                      skippunct=skippunct, updateinterval=10000)
        # Run parallel processes
        gc.collect()
        valences = []
        for round in range(rounds):
            print("Round %d" % round)
            procPool = mp.Pool(processes = nrjobs)
            results = procPool.map(partial_getValences,
                                   enumerate(textchunks[round * nrjobs: (round + 1) * nrjobs]))
            procPool.close()
            procPool.join()
            procPool.terminate()
            # Combine the results
            for resultnr, result in sorted(results, key=itemgetter(0)):
                valences += result
            results = []  # Empty out results, to free up memory
            gc.collect()

    # Now we have the valences on a per-row basis
    # If we have texts that run across multiple rows (a legacy problem with long texts), combine across rows
    if multiplerowspertext:
        valencesX = valences
        valences = []

        curid = -1
        curvalence = ['dummy',]
        for valence_info in valencesX:
            id = valence_info[0]
            if id != curid:
                valences.append([curid,] + curvalence)
                curid = id
                curvalence = valence_info[1:]
            else:  # combine with previous
                newtotals = [x + y for x, y in zip(curvalence[0:nroutputs], valence_info[1:nroutputs+1])]
                newunitlevel = curvalence[nroutputs:] + valence_info[nroutputs+1:]
                curvalence = newtotals + newunitlevel

        valences.append([curid,] + curvalence)  # Flush last entry
        valences.pop(0)  # First valence stored was a dummy

    # Save to output file (may be large!)
    # As appropriate, save subunit data too
    # Note: subunit data will be assigned sequence numbers, placed in the second column

    with open(valencefile, 'w') as outf:
        outfile = csv.writer(outf)
        newheader = ['id', 'nrwords'] + lexnames
        outfile.writerow(newheader)  # Header line -- note: adjust for values  prior to textcol
        outfile.writerows((x[:nroutputs + 1] for x in valences))  # 1 for id; cut off to ignore subunit data

    if subunit_valencefile != '':
        with open(subunit_valencefile, 'w') as outf:
            outfile = csv.writer(outf)
            newheader = ['id', 'sequencenr', 'nrwords'] + lexnames
            outfile.writerow(newheader)  # Header line
            for valence_info in valences:
                keepdata = valence_info[0]
                subunitinfo = valence_info[nroutputs + 1:]
                nrunits = int(len(subunitinfo) / nroutputs)
                for unit in range(nrunits):
                    valdata = subunitinfo[unit * nroutputs: (unit + 1) * nroutputs]
                    # Need to scale valdata by first item (word count)
                    wordcount = valdata[0]
                    valdata = [wordcount,] + [x / float(wordcount) for x in valdata[1:]]
                    outfile.writerow(keepdata + [unit,] + valdata)

    return  # output written out; no need to return anything


def getValences(textchunk, lexiconlist, wildlexicon, wild,
                wordlist, ignore, skip, mods, negs, separator='',
                need2tokenize=False, makelower=False, skippunct=True,
                updateinterval=100000):
    """Calculate valences for a chunk of texts.

    Called in the parallel processing part of calc_corpus_valence.
    """
    chunknr, chunktexts = textchunk
    valences = [[id,] + getValence(text, lexiconlist, wildlexicon, wild,
                                   wordlist, ignore, skip, mods, negs,
                                   separator=separator,
                                   need2tokenize=need2tokenize, makelower=makelower,
                                   chunknr=chunknr, textnr=count, updateinterval=updateinterval) \
                for count, (id, text) in enumerate(chunktexts)]
    return (chunknr, valences)


# ***************************** Valence mark-up **************************

def formatvalencemarking(sentimentstring, format='color', scaleparam=-1):
    """Take a string containing valence markers and mark up in color or bold/underline.

    Valence markers follow a word, and are:
    (*) for intensifiers/modifiers
    (<val>) for net count of positive & negative lexicon presence (from -8 to 8).

    Formatting is either
    'color' - magenta text for intensifiers/modifiers
            - white text on blue background for negative / red background for positive
              (darker shade for presence in more lexica)
    'bw' - underline for intensifiers/modifiers
         - boldface for valence words
    """
    from sty import fg, bg, rs  # see https://pypi.org/project/sty/

    # Set up styles to use for color (sty documentation is at https://sty.mewo.dev/)

    # Used the online monochrome scale generator to generate 8 (our number of lexica) reds and 8 blues
    # https://pinetools.com/monochromatic-colors-generator

    # Set of 8 generated by taking as base colors #0000ff, #ff0000, and #008000 respectively (blue, red, green),
    # asking for 17 steps on a scale and ignoring first 5 and last 4 (red and blue) or the first 3, 10-12, 14, and 16-17 (green);
    # the lighter ones work better with black text; the darker with white.

    # myblues = ['00009f', '0000bf', '0000df', '0000ff', '1f1fff', '3f3fff', '5f5fff', '7f7fff']
    # myreds = ['9f0000', 'bf0000', 'df0000', 'ff0000', 'ff1f1f', 'ff3f3f', 'ff5f5f', 'ff7f7f']
    # mygreens = ['#005f00', '#007f00', '#009f00', '#00bf00', '#00df00', '#00ff00', '#7fff7f', '#bfffbf']

    # Converted the hex format to rgb format for input into sty
    # Note: these are from darkest to lightest and we want lightest to darkest, so reverse using slice
    # mybluesRGB = [hex2rgb(x) for x in myblues[::-1]]
    # myredsRGB = [hex2rgb(x) for x in myreds[::-1]]
    # mygreensRGB = [hex2rgb(x) for x in mygreens[::-1]]

    # Hard-code the results of hex2rgb to make this function more standalone & faster
    mybluesRGB = [(127, 127, 255), (95, 95, 255), (63, 63, 255), (31, 31, 255),
                  (0, 0, 255), (0, 0, 223), (0, 0, 191), (0, 0, 159)]
    myredsRGB = [(255, 127, 127), (255, 95, 95), (255, 63, 63), (255, 31, 31),
                 (255, 0, 0), (223, 0, 0), (191, 0, 0), (159, 0, 0)]
    mygreensRGB = [(191, 255, 191), (127, 255, 127), (0, 255, 0), (0, 223, 0),
                   (0, 191, 0), (0, 159, 0), (0, 127, 0), (0, 95, 0)]

    # Set up styles to use for bw
    bold = '\033[1m'
    underline = '\033[4m'
    endformat = '\033[0m'

    modstring = sentimentstring.split()
    maxindex = len(modstring) - 1
    wordstrings = []

    skipnext = False
    for count, word in enumerate(modstring):

        if skipnext:  # Skip if this was a flag we've already processed
            skipnext = False
            continue

        if count < maxindex:  # see if next item is a flag

            if modstring[count + 1] == '(*)':  # modifier -> magenta text or underline
                if format == 'color':
                    wordstrings.append(fg.da_magenta + word + fg.rs)
                else:  # format == 'bw'
                    wordstrings.append(underline + word + endformat)
                skipnext = True

            elif modstring[count + 1][0] == '(' and modstring[count + 1][-1] == ')':
                sentval = modstring[count + 1][1:-1]

                # see if positive and negative; if so, get net value
                sentparts = sentval.split(',')
                if len(sentparts) >  1:
                    sentval = sum([int(x) for x in sentparts])
                else:
                    sentval = int(sentval)

                if sentval < 0:  # negative -> shades of blue
                    sentval = -1 - sentval  # change to be from 0 to 7
                    if format == 'color':
                        wordstrings.append(bg(*mybluesRGB[sentval]) + (fg.white if sentval > 2 else '') + word + rs.all)
                    else:
                        wordstrings.append(bold + word + endformat)
                elif sentval > 0: # positive -> shades of green
                        sentval = sentval - 1  # change to be from 0 to 7
                        if format == 'color':
                            wordstrings.append(bg(*mygreensRGB[sentval]) + (fg.white if sentval > 4 else '') + word + rs.all)
                        else:
                            wordstrings.append(bold + word + endformat)
                else:
                    wordstrings.append(word)

                skipnext = True
            else:
                wordstrings.append(word)
        else:
            wordstrings.append(word)
    return ' '.join(wordstrings)


# **************************** Auxiliary functions **************************

def yieldtext(filename, idcol=0, textcols=(1,), header=False, capsmatter=True):
    """Read text data from a file 1 line at a time (to avoid loading whole file into memory).

    Optionally lower-case the text in the specified textcols before returning the line.
    """
    import csv
    csv.field_size_limit(1000000000)

    with open(filename, 'r', encoding='utf-8') as corpusfile:
        inreader = csv.reader(corpusfile)
        if header:
            dummy = next(inreader)
        for line in inreader:
            if capsmatter:
                yield([line[idcol], ' '.join(line[textcol] for textcol in textcols)])
            else:
                yield([line[idcol], ' '.join(line[textcol].lower() for textcol in textcols)])


def load_lex(filename):
    """Load a lexicon (key: value) from a file."""
    import csv
    with open(filename, 'r') as infile:
        return {row[0]: float(row[1]) for row in csv.reader(infile)}


def haswilds(lexicon, wild='*'):
    return any([x[-1] == wild for x in list(lexicon.keys())])


def lexiconmatch(word, lexicon):
    """Return 0 if word not in lexicon; lexicon valence otherwise."""
    return lexicon[word] if word in lexicon else 0


def lexiconmatch_wild(word, lexicon, wild='*'):
    """Return 0 if word not in lexicon; lexicon valence otherwise.

    Note: accepts a wildcard (default '*') for '0 or more letters'.
    """
    if word in lexicon:
        return lexicon[word]
    else:
        if word[-1] != wild:
            word += wild
        while len(word) > 2:
            if word in lexicon:
                return lexicon[word]
            else:
                word = word[:-2] + wild
    return 0


def wordinset_wild(word, wordset, wild='*'):
    """Return True if word not in wordset, nor matched by wildcard.

    Note: accepts a wildcard (default '*') for '0 or more letters'.
    """
    if word in wordset:
        return True
    else:
        if word[-1] != wild:
            word += wild
        while len(word) > 2:
            if word in wordset:
                return True
            else:
                word = word[:-2] + wild
    return False


def allkeys(lexiconlist):
    """Generate a list of all terms across the lexica in lexiconlist.

    Filter out any terms subsumed by terms with wildcards
    """
    # Combine all keys into one large set
    lexiconkeys = set()
    for lexicon in lexiconlist:
        lexiconkeys |= set(lexicon.keys())
    # Remove entries subsumed by wildcards
    todelete = [x for x in lexiconkeys if subsumed(x, lexiconkeys, report=False)]
    for subsumedword in todelete:
        lexiconkeys.remove(subsumedword)
    # Return the resulting set
    return lexiconkeys


def subsumed(origx, words, report=True):
    """See whether origx is subsumed by a wildcarded entry in words."""
    if origx[-1] == '*':
        x = origx[:-2] + '*'
    else:
        x = origx + '*'
    while len(x) > 1:
        if x in words:
            if report:
                print(x, 'subsumes', origx)
            return x
        else:
            x = x[:-2] + '*'
    return False

