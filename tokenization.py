# encoding: utf-8

# tokenization.py
# by A. Maurits van der Veen

# Modification history:
# 2019-09-20 - Copied from punctuation.py
# 2020-02-19: handle curly quotes
# 2021-12-08: clean-up and some reorganization

__author__ = 'maurits'

# Pre-tokenize raw text corpora, for easier processing down the line.

# Functions to clean punctuation in texts, including splitting sentences
# and stripping commas. Works on texts in multiple languages, though it is
# most complete for English-language texts.

# Languages supported at the moment:
# Danish, Dutch, English, French, German, Italian, Spanish
# (English by far the most complete)

# ******************************* Wrapper functions *********************************************

def punctuationPrepPlus(aText):
    """Run our own punctuation preprocessor, followed by the nltk version,
    to (potentially) capture any remaining sentence splitting not correctly
    handled by ours. In most cases, this will not change anything over
    just running our own by itself.
    """
    return nltk_punctuation(punctuationPreprocess(aText))


def nltk_punctuation(aText):
    """Run nltk sentence tokenizer to split text cleanly into sentences;
    the re-construct text, with periods set off by spaces."""

    import nltk

    sent_tok = nltk.data.load('tokenizers/punkt/english.pickle')
    return ' . '.join(sent_tok.tokenize(aText, realign_boundaries=True))


def preprocess_texts(inputfile, outputfile, lang='english',
                     inheader=True, outheader=False,
                     stripspecial=False, stripcomma=False,
                     textcols=(10, 12), keepcols=(0,),
                     append=False,
                     reportinterval=5000):
    """Take csv file with raw text & create clean csv file with id,text format.

    Merge text fields specified in textcols and preprocess (focused on punctuation cleaning).
    """
    import csv
    if stripspecial:
        import unidecode  # Used to force decoding into ascii characters only

    csv.field_size_limit(1000000000)

    with open(inputfile, 'r', encoding='utf-8', errors='ignore') as inf, \
            open(outputfile, 'a' if append else 'w', newline='', encoding='utf-8') as outf:
        inreader = csv.reader(inf)
        output = csv.writer(outf)

        if inheader:
            oldheader = next(inreader)
            if outheader:
                output.writerow([oldheader[x] for x in keepcols] + ['cleantext',])

        for counter, article in enumerate(inreader):
            article_text = ' . '.join([article[col] for col in textcols])

            # Optionally, remove any special characters (warning: will strip accented characters too!)
            if stripspecial:
                try:
                    article_text = unidecode(article_text)
                except:
                    try:
                        article_text = unidecode(article_text.decode('utf-8'))
                    except:
                        article_text = unidecode(article_text.decode('ascii', 'ignore'))

            # Call main preprocessing function
            preppedtext = punctuationPreprocess(article_text, lang)
            # Optionally remove commas (rarely of any use)
            if stripcomma:
                preppedtext = stripcommas(preppedtext)

            # Write text plus any columns specified in keepcols
            output.writerow([article[col] for col in keepcols] + [preppedtext,])
            if counter % reportinterval == 0:
                print(("Processing text %d" % counter))

    return counter  # Return number of texts processed


# ******************************* Auxiliary functions *********************************************

def stripcommas(aText):
    """Remove commas from text.

    Commas separate sub-phrases within a sentence and are often useful to keep
    in. They are also used in our punctuation preprocessor, so don't strip
    them out until after running that function on a text.
    """
    import re
    # Replace by a space in case comma directly connects two words.
    aText = re.sub(",", " ", aText)
    # Remove any multi-space sections (some may result from the above step)
    return re.sub(" {2,}", " ", aText)


def stripperiods(aText):
    """Remove periods from text."""
    import re
    # Replace by a space in case comma directly connects two words.
    aText = re.sub(r"\.", " ", aText)
    # Remove any multi-space sections (some may result from the above step)
    return re.sub(" {2,}", " ", aText)


def removeperiods(filename):
    """Remove all periods from a file."""
    import csv
    with open(filename + '.csv', 'rU') as inf, \
            open(filename + 'X.csv', 'w', newline='') as outf:
        outwriter = csv.writer(outf)
        for entry in csv.reader(inf):
            outwriter.writerow((entry[0], stripperiods(entry[1])))


# *************************** Main preprocessing function *****************************************

def punctuationPreprocess(aText, lang='english'):
    """Clean text, including simplifying punctuation down to periods and commas only.

    Starting point for the regexes here is the punctuation preprocessing
    in Lexicoder, but it has been extensively revised & changed.

    Punctuation characters to deal with include (see string.punctuation):
    - !, ?, and ; which become periods;
    - brackets (), [], and {} which become commas;
    - colon, single and double quotes (:, ', ") which become commas
    - a single dash inside a word, which we leave untouched
    - ibid. following a space, which becomes a comma (like a double dash)
    - special characters &, %, and / which we spell out: and, percent, or
    - special characters +, = which we spell out: plus, is equal to
    - special char $ which we spell out and move after the dollar value
    - remaining special chars #, *, <, >, @, \, ^, _, |, ~ which we surround
      by spaces (may think about better things to do later)
    """
    import re

    # 1a. Handle websites (very simplistic right now: xx prefix and periods
    # become spaces). Just deals with web address (e.g. www.wm.edu),
    # not with specific files (e.g. www.wm.edu/index.html), as the latter
    # will rarely be mentioned in newspaper articles
    aText = re.sub(r"www\d{0,3}[.]([a-zA-Z0-9])+[.]([a-z]{2,4})",
                   "xx\\1 xx\\2", aText)
    aText = re.sub(r"([a-zA-Z0-9])+[.]([a-zA-Z0-9])+[.]([a-z]{2,4})",
                   "xx\\1 xx\\2 xx\\3", aText)

    # 1b. Remove phone numbers in 4-3-4 (UK), 3-3-4 (US), 3-4 (both) formats
    # Could think about handling the +44 (0) format for the UK and
    # the 1- or +1 formats for the US
    aText = re.sub("\\bd4\[ \t\n\r\f\v-.]d3[ \t\n\r\f\v-.]d4", "", aText)
    aText = re.sub("\\bd3[ \t\n\r\f\v-.]d3[ \t\n\r\f\v-.]d4", "", aText)
    aText = re.sub("\\b\(d3\)[ \t\n\r\f\v-.]d3[ \t\n\r\f\v-.]d4", "", aText)
    aText = re.sub("\\bd3[ \t\n\r\f\v-.]d4", "", aText)

    # 1c. Replace curly quotes by straight quotes
    aText = re.sub('“', '"', aText)
    aText = re.sub('”', '"', aText)
    aText = re.sub("‘", "'", aText)
    aText = re.sub("’", "'", aText)

    # 2. Language-specific substitutions
    if lang == 'dansk':
        aText = punctuation_danish(aText)
    elif lang == 'deutsch':
        aText = punctuation_german(aText)
    elif lang == 'english':
        aText = punctuation_english(aText)
    elif lang in ('español', 'espanol'):
        aText = punctuation_spanish(aText)
    elif lang in ('français', 'francais'):
        aText = punctuation_french(aText)
    elif lang == 'italiano':
        aText = punctuation_italian(aText)
    elif lang == 'nederlands':
        aText = punctuation_dutch(aText)
    elif lang == 'norsk':
        aText = punctuation_norwegian(aText)
    else:
        # print("no language-specific processing")
        pass

    # 2a. Arab words with internal '
    # (Should in general be handled in a separate translation dictionary)
    aText = re.sub("\\bba'ath", 'baath', aText, flags=re.IGNORECASE)

    # 3c. Remaining special characters just get surrounded by spaces,
    #     except underscores, which we assume to be deliberate concatenators
    aText = re.sub(r"([#*<>@\\^|~])", " \\1 ", aText)

    # 4. Non-sentence-ending periods after single upper- or lower-case letter (e.g. in a list)
    aText = re.sub("( [a-zA-Z])\\. ", "\\1 ", aText)

    # 5. Simplify punctuation

    # A. sentence breaks become periods (including semi-colon)
    aText = re.sub("!", ".", aText)
    aText = re.sub(r"\?", ".", aText)
    aText = re.sub(";", ".", aText)
    # Sequences of periods (ellipsis) become just 1
    aText = re.sub("[\\.]{2,}", ".", aText, flags=re.IGNORECASE)

    # B. single and double quotes become commas -- is this a good idea?
    # Note that we may have handled apostrophes already in language-specific treatments
    # (French, Italian)
    aText = re.sub(r'"', ", ", aText)
    aText = re.sub(r"'", ", ", aText)

    aText = re.sub(":", ",", aText)  # Note: this will also break up time-of-day
    aText = re.sub("--", ",", aText)
    aText = re.sub(" - ", " , ", aText)
    aText = re.sub(r"\[", ", ", aText)
    aText = re.sub(r"\]", " , ", aText)
    aText = re.sub(r"\(", ", ", aText)
    aText = re.sub(r"\)", " , ", aText)
    aText = re.sub(r"\{", ", ", aText)
    aText = re.sub(r"\}", " , ", aText)
    aText = re.sub(r"\.,", ",", aText)
    # simply remove backquotes
    aText = re.sub(r"`", " ", aText)

    # C. Remove multiple spaces and consecutive commas / periods
    aText = re.sub("\s{2,}", " ", aText)
    aText = re.sub(r",( ?,)+", ",", aText)
    aText = re.sub(r"\.( ?\.)+", ".", aText)

    # D. underscores become spaces -> not always a good idea
    aText = re.sub("_", " ", aText)

    # Remove periods from acronyms
    aText = re.sub("(\\.)([A-Z])(\\.) ", "\\1\\2 ", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    # Remove decimal periods
    aText = re.sub("([0-9])(\\.)([0-9])", "\\1dot\\3", aText)

    # In Lexicoder, proper names are disambiguated from words by inserting
    # an underscore. However, this misses too many non-name situations,
    # while the likelihood of a real name being confused for a word we're
    # interested in is small enough that we can accept it.
    # aText = re.sub("([a-z0-9’,] )([A-Z])([a-zA-Z]+)",
    #                "\\1(name) \\2_\\3", aText)

    # Surround period & commas by a space
    aText = re.sub(r"\.", " . ", aText)
    aText = re.sub(r",", " , ", aText)

    # Remove multiple spaces and consecutive commas / periods, one last time
    aText = re.sub("\s{2,}", " ", aText)
    aText = re.sub(r",( ,)+", ",", aText)
    aText = re.sub(r"\.( \.)+", ".", aText)
    return aText


# ******************************* Language-specific preprocessing ***********************************

# Most of these feature, at a minimum, the data to replace fractions and special
# characters by corresponding words, as controlled in the following two functions

def fraction_to_word(aText, fractions):
    """Spell out fractions written with a '/'. """
    import re
    aText = re.sub(r"1/2", fractions[0], aText)
    aText = re.sub(r"1/3", fractions[1], aText)
    aText = re.sub(r"2/3", fractions[2], aText)
    aText = re.sub(r"1/4", fractions[3], aText)
    aText = re.sub(r"3/4", fractions[4], aText)
    aText = re.sub(r"1/5", fractions[5], aText)
    return aText


def char_to_word(aText, charsubst):
    """Handle special characters used instead of words"""
    import re
    aText = re.sub("&", charsubst[0], aText)
    aText = re.sub("%", charsubst[1], aText)
    aText = re.sub(r"\+", charsubst[2], aText)
    aText = re.sub(r"=", charsubst[3], aText)
    aText = re.sub("/", charsubst[4], aText)  # will also split up URLs
    # Also handle abbreviations for numbers: No. 1, Nr. 2, etc.
    # Could do # as a number sign here as well, but not for now
    aText = re.sub("\\b([Nn][or])(\\. )([0-9]+)\\b",
                   charsubst[5] + "\\3", aText)
    return aText

# The language-specific functions are sorted by the order of the language's name
# (in English)

def punctuation_danish(aText):
    """Simplify punctuation for Danish-language texts.

    First draft -- should be expanded considerably.
    """
    import re

    # Spell out fractions and special characters such as &, %, etc.
    # Note: these are not capitalized since we mostly work with lower-case
    aText = fraction_to_word(aText,
                             (' 1div2 ', ' 1div3 ', ' 2div3 ',
                              ' 1div4 ', ' 3div4 ', ' 1div5 '))
    aText = char_to_word(aText,
                         (' og ', ' prosent ', ' plus ', ' er ',
                          ' eller ', ' nummer '))
    return aText


def punctuation_dutch(aText):
    """Simplify punctuation for Dutch-language texts."""
    import re

    # Double comma is the rendering of an opening quote
    aText = re.sub(r",,", '"', aText)
    # on- starting a word means 'niet x'
    aText = re.sub("\\bun-", "niet ", aText, flags=re.IGNORECASE)

    # Remove general possessive 's (very common, no useful info)
    aText = re.sub("\\b([a-z]+)'s\\b", "\\1", aText,
                   flags=re.IGNORECASE)

    # Spell out common abbreviations
    aText = re.sub("\\bca\\.", "circa", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bvs\\.", "versus", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bSt\\.", "Sint", aText, flags=re.IGNORECASE)

    aText = re.sub("\\be\\.d\\.", "en dergelijke", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bo\\.a\\.", "onder anderen", aText, flags=re.IGNORECASE)

    aText = re.sub("\\bd\\.m\\.v\\.", "door middel van", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bd\\.w\\.z\\.", "dat wil zeggen", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bt\\.a\\.v\\.", "ten aanzien van", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bm\\.m\\.v\\.", "met medewerking van", aText, flags=re.IGNORECASE)

    # Spell out fractions and special characters such as &, %, etc.
    aText = fraction_to_word(aText,
                             (' de helft ', ' een derde ', ' twee derde ',
                              ' een vierde ', ' drie vierde ', ' een vijfde '))
    aText = char_to_word(aText,
                         (' en ', ' procent ', ' plus ', ' is gelijk aan ',
                          ' of ', ' nummer '))
    return aText


def punctuation_english(aText):
    """Simplify punctuation for English-language texts."""
    import re

    # 1a. Handle dollar values
    aText = re.sub(r"([0-9.,]+) ?bn\b", "\\1 billion", aText)
    aText = re.sub(r"([0-9.,]+) ?mn\b", "\\1 million", aText)
    aText = re.sub(r"\$([0-9.,]+[0-9])", "\\1 dollar", aText)
    aText = re.sub(r"\bdollar billion\b", "billion dollar", aText)
    aText = re.sub(r"\bdollar million\b", "million dollar", aText)

    # 1b. Handle pound sterling values
    aText = re.sub(r"\bps ?([0-9.,]+[0-9])b\b", "\\1 billion pound sterling", aText)
    aText = re.sub(r"\bps ?([0-9.,]+[0-9])m\b", "\\1 million pound sterling", aText)
    aText = re.sub(r"\bps ?([0-9.,]+[0-9])", "\\1 pound sterling", aText)

    # 2. Handle times of day
    aText = re.sub(r"[0-9]{1,2} ?a\.?m\.?\b", "time_val time_am", aText)
    aText = re.sub(r"[0-9]{1,2} ?p\.?m\.?\b", "time_val time_pm", aText)
    aText = re.sub(r"[0-9]{1,2}:[0-9]{2} ?a\.?m\.?\b", "time_val time_am", aText)
    aText = re.sub(r"[0-9]{1,2}:[0-9]{2} ?p\.?m\.?\b", "time_val time_pm", aText)
    aText = re.sub(r"[0-9]{1,2}:[0-9]{2}", "time_val", aText)

    # 3. Expand contractions
    aText = re.sub("\\bit's\\b", "it is", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bhe's\\b", "he is", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bshe's\\b", "she is", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bain't\\b", "is not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bisn't\\b", "is not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\baren't\\b", "are not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\blet's\\b", "let us", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bwon't\\b", "will not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bcan't\\b", "can not", aText, flags=re.IGNORECASE)
    aText = re.sub("n't", " not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bcannot\\b", "can not", aText, flags=re.IGNORECASE)
    # 'd can be either had or would; could look at context to disambiguate:
    # is next word an infinitive (would) or a participle/object (had)?
    # for now, pick had (more neutral).
    # Also, don't ignore case on the D, as then we'll catch O'Donnell, etc.!
    aText = re.sub("([A-Za-z]+)'d", "\\1 had", aText)
    aText = re.sub("([a-z]+)'ll", "\\1 will", aText, flags=re.IGNORECASE)
    aText = re.sub("([a-z]+)'m", "\\1 am", aText, flags=re.IGNORECASE)
    aText = re.sub("([a-z]+)'ve", "\\1 have", aText, flags=re.IGNORECASE)
    aText = re.sub("([a-z]+)'re", "\\1 are", aText, flags=re.IGNORECASE)

    # 4. Possessive apostrophes -- legacy substitutions from Lexicoder; no longer implemented
    # aText = re.sub("\\bbull's eye\\b", "bullseye", aText, flags=re.IGNORECASE)
    # aText = re.sub("\\bno man's land\\b", "nomansland", aText,
    #                flags=re.IGNORECASE)
    # aText = re.sub("\\bpandora's box\\b", "pandoras box", aText,
    #                flags=re.IGNORECASE)

    # Remove general possessive 's (very common, generally no useful info)
    # Note: maybe should count as a word?!
    aText = re.sub("\\b([a-z]+)'s\\b", "\\1", aText,
                   flags=re.IGNORECASE)

    # 5. Words with multiple meanings where punctuation resolves ambiguity

    # a. too followed by punctuation means it is used in the meaning of 'also'
    aText = re.sub("\\btoo ?([.,:;)'\"\\]])", "also \\1", aText,
                   flags=re.IGNORECASE)
    # b. un- starting a word means 'not x'
    aText = re.sub("\\bun-", "not ", aText, flags=re.IGNORECASE)
    # c. expressions that might otherwise get categorized erroneously
    #    due to use of valence word in non-valence (or different-valence)
    #    context identified by punctuation -> prepend x to the word
    aText = re.sub("-like", " xlike", aText, flags=re.IGNORECASE)
    aText = re.sub(", well,", " xwell,", aText, flags=re.IGNORECASE)
    aText = re.sub(r"[\.]{2,} well", " xwell", aText, flags=re.IGNORECASE)
    aText = re.sub(r"\bWell,", "xwell,", aText)  # issue: this was re.IGNORECASE until 2019-09-13!!
    aText = re.sub("\\bOK\\b", "okay", aText)
    aText = re.sub(", okay,", " xokay", aText, flags=re.IGNORECASE)
    # aText = re.sub("\" okay,", " xokay", aText, flags=re.IGNORECASE)  # From Lexicoder
    # This erroneously would capture something like "he felt OK, after all"
    aText = re.sub("[\\.]{2,} okay", " xokay", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bOkay,", "xokay,", aText)  # issue: this was re.IGNORECASE until 2019-09-13!!

    # 6. Expand common abbreviations

    # Canadian provinces
    aText = re.sub("\\bQue\\.", "Quebec", aText)
    aText = re.sub("\\bOnt\\.", "Ontario", aText)
    aText = re.sub("\\bNfld\\.", "Newfoundland", aText)
    aText = re.sub("\\bAlta\\.", "Alberta", aText)
    aText = re.sub("\\bMan\\.", "Manitoba", aText)
    aText = re.sub("\\bSask\\.", "Saskatchewan", aText)

    # U.S. states (common abbrevs. only)
    aText = re.sub("\\bAla\\.", "Alabama", aText)
    aText = re.sub("\\bCalif\\.", "California", aText)
    aText = re.sub("\\bMass\\.", "Massachusetts", aText)

    # Other abbreviations (esp. forms of address)
    aText = re.sub("\\bvs\\.", "versus", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bSt\\.", "St", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bSte\\.", "Ste", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bi\\.e\\.", "that is", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bet al\\.", "et alii", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bmr\\.", "Mr", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bms\\.", "Ms", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bmrs\\.", "Mrs", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bprof\\.", "Prof", aText, flags=re.IGNORECASE)
    aText = re.sub("\\ba\\. ?m\\. ", "am ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bp\\. ?m\\.", "pm ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bdr\\.", "Dr", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bgen\\.", "gen", aText, flags=re.IGNORECASE)
    aText = re.sub("\\be\\. coli\\b", "ecoli", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bvs\\.", "versus", aText, flags=re.IGNORECASE)
    aText = re.sub("\\binc\\.", "incorporated", aText, flags=re.IGNORECASE)

    # Month abbreviations
    aText = re.sub("\\bJan\\.", "January", aText)
    aText = re.sub("\\bFeb\\.", "February", aText)
    aText = re.sub("\\bMar\\.", "March", aText)
    aText = re.sub("\\bApr\\.", "April", aText)
    aText = re.sub("\\bJun\\.", "June", aText)
    aText = re.sub("\\bJul\\.", "July", aText)
    aText = re.sub("\\bAug\\.", "August", aText)
    aText = re.sub("\\bSept\\.", "September", aText)
    aText = re.sub("\\bOct\\.", "October", aText)
    aText = re.sub("\\bNov\\.", "November", aText)
    aText = re.sub("\\bDec\\.", "December", aText)

    # Spell out fractions and special characters such as &, %, etc.
    aText = fraction_to_word(aText,
                             (' half ', ' one third ', ' two thirds ',
                              ' one fourth ', ' three fourths ', ' one fifth '))
    aText = char_to_word(aText,
                         (' and ', ' percent ', ' plus ',
                          ' is equal to ', ' or ', ' number '))
    return aText


def punctuation_french(aText):
    """Simplify punctuation for French-language texts.

    First draft -- should be expanded considerably.
    """
    import re

    # Spell out common abbreviations
    aText = re.sub("\\bmr\\.? ?", "monsieur ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bmme\\.? ?", "madame ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bmlle\\.? ?", "mademoiselle ", aText, flags=re.IGNORECASE)

    # Expand contractions with apostrophe
    aText = re.sub("aujourd'hui", "aujourdhui", aText, flags=re.IGNORECASE)
    # Note: l' could be le or la, but make it le here (too complicated to figure out which)
    #       s' could be se or si, but make it se here (could figure out but not worth it)
    aText = re.sub("(c|d|j|l|m|qu|r|s|t)'([a-zA-Zàî])", "\\1e \\2", aText, flags=re.IGNORECASE)

    # Strip Romance language quotation markers
    aText = re.sub("«", "", aText)
    aText = re.sub("»", "", aText)

    # Spell out fractions and special characters such as &, %, etc.
    # Note: these are not capitalized since we mostly work with lower-case
    # Removed umlauts on halfte, funftel
    aText = fraction_to_word(aText,
                             (' la moitie ', ' un tiers ', ' deux tiers ',
                              ' un quart ', ' trois quarts ', ' vingt pour cent '))
    aText = char_to_word(aText,
                         (' et ', ' pour cent ', ' plus ', ' est ',
                          ' ou ', ' numero '))
    return aText


def punctuation_german(aText):
    """Simplify punctuation for German-language texts.

    First draft -- should be expanded considerably.
    """
    import re

    # Spell out common abbreviations
    aText = re.sub("\\bca\\.", "circa", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bvs\\.", "versus", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bSt\\.", "Sint", aText, flags=re.IGNORECASE)

    # Spell out fractions and special characters such as &, %, etc.
    # Note: these are not capitalized since we mostly work with lower-case
    # Note 2: in some cases may want to remove umlauts on hälfte, fünftel
    aText = fraction_to_word(aText,
                             (' die hälfte ', ' ein drittel ', ' zwei drittel ',
                              ' ein viertel ', ' drei viertel ', ' ein fünftel '))
    aText = char_to_word(aText,
                         (' und ', ' prozent ', ' plus ', ' ist ',
                          ' oder ', ' nummer '))
    return aText


def punctuation_italian(aText):
    """Simplify punctuation for Italian-language texts.

    Worth looking into additional improvements

    Stampa through July 2008 has accents in the next character space: e' instead of è
    """
    import re

    # 1. Expand common abbreviations with periods (so we don't interpret as sentence markers
    # Good list: http://homes.chass.utoronto.ca/~ngargano/corsi/corrisp/abbreviazioni.html
    # *** to be done ***
    # Sir
    aText = re.sub("\\bSig\\.", "Signore ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bSigg\\.", "Signori ", aText, flags=re.IGNORECASE)
    # Saints
    aText = re.sub("\\bS\\.", "Santo ", aText)  # These could also be Santa/e, or also Santissimo/a
    aText = re.sub("\\bSS\\.", "Santi ", aText) # but that does not matter a lot

    # 2. Expand contractions with apostrophe
    # See: http://www.learnita.net/use-of-apostrophe-in-italian-language/
    # Note: dropped vowel could generally be o or a -- quick and dirty approach here to figure out
    # Use the captured group for bell', grand', etc. to keep capitalization intact
    aText = re.sub("\\b(bell|grand|sant|quest|quell|tutt|mezz|molt|cent|quant)'([aeiouyhAEIOUYH]\\w*?a\\b)",
                   "\\1a \\2", aText, flags=re.IGNORECASE)
    aText = re.sub("\\b(bell|grand|sant|quest|quell|tutt|mezz|molt|cent|quant)'([aeiouyhAEIOUYH])",
                   "\\1o \\2", aText, flags=re.IGNORECASE)

    aText = re.sub("\\b(buon)'([aeiouyhAEIOUYH])", "\\1a \\2", aText, flags=re.IGNORECASE)

    aText = re.sub("\\b(fors|dev|diss)'([aeiouyhAEIOUYH])", "\\1e \\2", aText, flags=re.IGNORECASE)

    aText = re.sub("l'([aeiouyhAEIOUYH]\\w*?a\\b)", "la \\1", aText, flags=re.IGNORECASE)
    aText = re.sub("l'([aeiouyhAEIOUYH])", "lo \\1", aText, flags=re.IGNORECASE)

    aText = re.sub("un'([aeiouyhAEIOUYH])", "una \\1", aText, flags=re.IGNORECASE)

    # note: d'allora and d'ora are both da (could keep caps, but not worth it)
    aText = re.sub("d'ora", "da ora", aText, flags=re.IGNORECASE)
    aText = re.sub("d'allora", "da allora", aText, flags=re.IGNORECASE)

    aText = re.sub("(d|m|t|v|s)'([aeiouyhAEIOUYH])", "\\1i \\2", aText, flags=re.IGNORECASE)
    aText = re.sub("(gl|c)'([eiEI])", "\\1i \\2", aText, flags=re.IGNORECASE)
    aText = re.sub("(n)'([aeiouyhAEIOUYH])", "\\1e \\2", aText, flags=re.IGNORECASE)

    # 3. Add accents where it was previously a single quote following the target letter
    # This is only necessary for La Stampa up through July 2008, but doesn't hurt
    # Note: this may mess up quoted phrases ended in a non-accented vowel
    aText = re.sub("a'", u"à", aText)
    aText = re.sub("e'", u"è", aText)
    aText = re.sub("i'", u"i", aText)
    aText = re.sub("o'", u"ò", aText)
    aText = re.sub("u'", u"ù", aText)
    # Note 2: Sometimes, when capitalized, there is a space between the letter and the single quote
    aText = re.sub("A ?'", u"À", aText)
    aText = re.sub("E ?'", u"È", aText)
    aText = re.sub("I ?'", u"Ì", aText)
    aText = re.sub("O ?'", u"Ò", aText)
    aText = re.sub("U ?'", u"Ù", aText)

    # 4. Remove special quote characters
    aText = re.sub("«", "", aText)
    aText = re.sub("»", "", aText)

    # Spell out fractions and special characters such as &, %, etc.
    # Note: these are not capitalized since we mostly work with lower-case
    # Note: in some cases may want to remove accents on metà, è, più
    aText = fraction_to_word(aText,
                             (u' la metà ', ' un terzo ', ' due terzi ',
                              ' un quarto ', ' tre quarti ', ' un quinto '))
    aText = char_to_word(aText,
                         (' e ', ' per cento ', u' più ', u' è ',
                          ' o ', ' numero '))
    return aText


def punctuation_norwegian(aText):
    """Simplify punctuation for Norwegian-language texts.

    First draft -- should be expanded considerably.
    """
    import re

    # Spell out fractions and special characters such as &, %, etc.
    # Note: these are not capitalized since we mostly work with lower-case
    aText = fraction_to_word(aText,
                             (' 1div2 ', ' 1div3 ', ' 2div3 ',
                              ' 1div4 ', ' 3div4 ', ' 1div5 '))
    aText = char_to_word(aText,
                         (' og ', ' prosent ', ' plus ', ' er ',
                          ' eller ', ' nummer '))
    return aText


def punctuation_spanish(aText):
    """Simplify punctuation for Spanish-language texts.

    Worth looking into additional improvements
    """
    import re

    # 1. Expand common abbreviations with periods (so we don't interpret as sentence markers
    # Good list: http://www.ctspanish.com/words/abbreviations.htm
    # *** to do!

    # Sir/Madam
    aText = re.sub("\\bSr\\.", "Senor ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bSra\\.", "Senora ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bSrta\\.", "Senorita ", aText, flags=re.IGNORECASE)

    # 2. Strip special characters
    aText = re.sub("«", "", aText)
    aText = re.sub("»", "", aText)
    aText = re.sub("¿", "", aText)
    aText = re.sub("¡", "", aText)

    # Spell out fractions and special characters such as &, %, etc.
    # Note: these are not capitalized since we mostly work with lower-case
    # Removed accents on metà, è
    aText = fraction_to_word(aText,
                             (' un medio ', ' un tercio ', ' dos tercios ',
                              ' un cuarto ', ' tres cuartos ', ' un quinto '))
    aText = char_to_word(aText,
                         (' y ', ' por ciento ', ' mas ', ' es igual a ',
                          ' o ', ' numero '))
    return aText


