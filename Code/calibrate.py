# encoding: utf-8

# calibrate.py
# by A. Maurits van der Veen

# Modification history:
# 2021-11-25: extract from valence.py
# 2021-12-01: convert to use pandas, keeping legacy i/o wrapper function

# Code to calibrate valence values; used in conjunction with valence.py

# ****************************************** Calibration ******************************************

def calibrate_features(valencefile, valencecols, neutralscaler='', stdev_adj=1,
                       filtercol='', keepcols=(), adjustvals=False,
                       missing=-999, header=True, colnames=[], outsuffix='_cal'):
    """Calibrate features by standardizing; return means & stdevs.

    Save original valence data plus calibrated data and overall average calibrated valence,
    along with any additional columns specified in keepcols

    This function is just an i/o wrapper around calibrate_valencedata.
    See there (below) for more info about the parameters.
    """
    import pandas as pd

    # Load data
    df = pd.read_csv(valencefile, index_col=0) if header else \
            pd.read_csv(valencefile, names=colnames, index_col=0)

    # Copy relevant columns of df to calibration function
    relevantcols = valencecols.copy()
    if len(filtercol) > 0:
        relevantcols.append(filtercol)
    df2calibrate = df[relevantcols].copy()
    calibrated_df, neutralscaler, stdev_of_avgs = \
        calibrate_valencedata(df2calibrate, valencecols, neutralscaler, stdev_adj,
                              filtercol, scalecol=filtercol if adjustvals else '',
                              missing=missing)

    # Save result, along with the columns to be kept (filtered as needed)
    df_towrite = pd.concat([df[keepcols], calibrated_df], axis=1, join='inner')
    outputfile = '.'.join(valencefile.split('.')[:-1]) + outsuffix + '.csv'
    df_towrite.to_csv(outputfile)

    # Return neutral scaler, for possible later use (if we generated it here),
    # plus some summary data
    return neutralscaler, stdev_of_avgs, len(df)


def calibrate_valencedata(df, valencecols, neutralscaler, stdev_adj,
                          filtercol='', scalecol='', missing=-999):
    """Calibrate valence data in a dataframe.

    df should contain all of valencecols, plus filtercol and scalecol if applicable
    valencecols is a list of the names of the columns to be scaled
    neutralscaler is an sklearn.StandardScaler, specifying the mean and standard deviation of a column,
        which can be used to standardize that column. If no neutralscaler is passed in, we calculate
        it on the data passed in here.
    stdev_adj is the standard deviation of the average of this particular set of standardized columns
    filtercol is the name of a column whose 0 entries will be used to filter out the corresponding rows
    scalecol is the name of a column whose non-0 entries will be used to scale (i.e. divide) the corresponding row data
    missing is the value to be inserted to represent missing data

    Return new dataframe with calibrated data, along with some summary information.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    if len(scalecol) > 0:  # Scale valence data by values in the scalecol
        # This is generally used to divide valences by text length, but usually happens in the valence calculation itself.
        for valencecol in valencecols:
            df[valencecol] = df.apply(lambda row: 0 if row[scalecol] == 0 else row[valencecol]/row[scalecol], axis=1)

    # Calibrate: rescale & recenter features
    if neutralscaler == '':
        neutralscaler = StandardScaler()
        neutralscaler.fit(df[valencecols].to_numpy())
        divide_at_end = True  # Need to divide avg. valence by std. dev. at the end to rescale
    else:
        divide_at_end = False
    scaled_features = neutralscaler.transform(df[valencecols])
    # Convert scaled features back into a dataframe
    calibratedcols = [valencecol + 'X' for valencecol in valencecols]
    scaled_features_df = pd.DataFrame(scaled_features, index=df.index,
                                      columns=calibratedcols)
    df = pd.concat([df, scaled_features_df], axis=1, join='inner')
    # df = df.merge(scaled_features_df, on='id', how='left')

    # Now handle the various options for treating missing info & calculating averages
    denominator = float(len(valencecols))  # When averaging, how much to divide by
    divisor = denominator * stdev_adj      # Combine averaging & re-standardizing step

    if len(filtercol) > 0:
        # Average, option 1: use calibrated values except in case of filter
        print("Nr. items to be filtered out: {}".format(len([x for x in df[filtercol] if x == 0])))
        df['avg_valence'] = df.apply(
                lambda row: missing if row[filtercol] == 0 else sum(row[x] for x in calibratedcols)/divisor, axis=1)
    else:
        # - individual-level adjustment: for original values that were 0,
        #   set to missing, rather than use adjusted value
        for calibratedcol in calibratedcols:
            df[calibratedcol] = df.apply(
                    lambda row: missing if row[calibratedcol[:-1]] == 0 else row[calibratedcol], axis=1)
        # Average, option 2: average the non-'zero' values
        df['avg_valence'] = df.apply(
                    lambda row: average_nonmissing(row, calibratedcols, missing, stdev_adj, countmissings=False), axis=1)
        # Alternatively, to divide by the number of lexica used (i.e. count 'zero' values as 0,
        # instead of the calibrated equivalent), specify countmissings=True

    new_stdev_adj = df['avg_valence'].std()
    if divide_at_end:
        df['avg_valence'] /= new_stdev_adj
    # Return neutral scaler, for possible later use (if we generated it here)
    return df, neutralscaler, new_stdev_adj if divide_at_end else stdev_adj


def calibrate_valences(valences, neutralscaler, stdev_adj,
                       firstvalencecol=1, showcomponents=False):
    """Calibrate valences by standardizing based on scaling data passed in.

    This is a very simplified version of the dataframe-based calibrate_valencedata

    Main data passed in should be a list of valencedata items.
    A valencedata item is a list containing a scaling value and N valence values (where N is the number of lexica used)
    The scaling value is usually the first item in the list, and generally is the word count of the corresponding text.

    Since the data is already scaled, we can ignore the scaling value.
    For each of the valence values, we do the corresponding standardization from the neutralscaler passed in.
    Then we average the resulting values and divide by the stdev_adj to get a calibrated value.

    For each valencedata item we return the calibrated value and optionally the individual calibrated components.

    Note: this is a simplified version of calibrate_features (see above), which we use to calibrate files
          with valence data. At some point, we might decompose calibrate_features to call the present function
          for the central calibration step.
    """
    nrlexica = len(valences[0][firstvalencecol:])
    calibratedlexvalences = neutralscaler.transform([valencedataitem[firstvalencecol:] for valencedataitem in valences])

    divisor = nrlexica * stdev_adj
    calibratedvalences = [sum(vals)/divisor for vals in calibratedlexvalences]

    if showcomponents:
        return [(calval, callexvals) for calval, callexvals in zip(calibratedvalences, calibratedlexvalences)]
    else:
        return calibratedvalences


# ****************************************** Storing & loading calibration data *******************

def load_scaler_fromcsv(filename, featurenames=(), includevar=False, displayinfo=True):
    """Load a calibration scaler from a text file

    Textfile will contain:
    - header line, containing scaler name, number of observations seen,
      number of features (N), standard deviation adjustment, and descriptor string
    - N rows of feature information, containing name, mean, standard deviation, and variance (if includevar)

    Function returns sklearn StandardScaler object,
                     number of features used, number of features available,
                     standard deviation adjustment, and descriptor

    If displayinfo is True, print information about the contents of the scaler before returning

    Note that the standard deviation adjustment is valid only if all features are used
    (i.e. number of features used == number of features available)
    """
    import csv
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Read data from file
    with open(filename, 'r', encoding='utf-8', errors='ignore') as scalerfile:
        scalercsv = csv.reader(scalerfile)

        # Read & parse header
        headerrow = next(scalercsv)
        name = headerrow[0]
        nrfeatures = int(headerrow[2])
        stdev_adj = float(headerrow[3])
        descriptor = headerrow[4]

        # Read & parse individual features
        nrfeaturesused = 0
        featuresused, means, stdevs, variances = [], [], [], []
        for row in scalercsv:
            if len(featurenames) == 0 or (len(featurenames) > 0 and row[0] in featurenames):
                featuresused.append(row[0])
                nrfeaturesused += 1
                means.append(float(row[1]))
                stdevs.append(float(row[2]))
                if includevar:
                    variances.append(float(row[3]))

    # Initialize scaler
    newscaler = StandardScaler()
    newscaler.n_samples_seen_ = int(headerrow[1])
    newscaler.mean_ = np.array(means)
    newscaler.scale_ = np.array(stdevs)
    newscaler.var_ = np.array(variances)

    if displayinfo:
        print("Descriptor:", descriptor)
        print("Lexica used ({}): {}".format(nrfeaturesused, featuresused))
        print("Means:", newscaler.mean_)
        print("Std. devs.:", newscaler.scale_)
        print("Std. dev. of average across lexica (calculated using {} lexica): {}".format(nrfeatures, stdev_adj))

    return newscaler, featuresused, nrfeaturesused, nrfeatures, stdev_adj, descriptor


def write_scaler_tocsv(filename, scaler, featurenames=(), includevar=True,
                       name='calibration', descriptor='description', stdev_adj=1):
    """Write a calibration scaler to a text file

    Textfile will contain:
    - header line, containing scaler name, number of observations seen,
      number of features (N), standard deviation adjustment, and descriptor string
    - N rows of feature information, containing name, mean, standard deviation, and variance (if includevar)
    """
    import csv
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    nrfeatures = len(scaler.mean_)
    variances = scaler.var_ if includevar else np.array(['', ] * nrfeatures)
    if len(featurenames) == 0:
        featurenames = ['feature_' + str(i) for i in range(nrfeatures)]

    with open(filename, 'w', encoding='utf-8') as scalerfile:
        scalercsv = csv.writer(scalerfile)

        # Write header
        scalercsv.writerow((name, scaler.n_samples_seen_, len(scaler.mean_), stdev_adj, descriptor))

        # Write individual feature info
        scalercsv.writerows(zip(featurenames, scaler.mean_, scaler.scale_, variances))

    return


def calibration_subset(calibratedfile, calibrationfile, lexnames, calibrationfile_subset,
                       includevar=True):
    """Generate a calibration file for a subset of lexica represented in the input data.

    Take <calibratedfile>, containing standardized data (mean 0, std. dev. 1) for a set of named lexica.
    Select the subset matching <lexnames>. Average them, and calculate the std. dev. or the result.

    Take <calibrationfile> and select the mean and std.dev. adjustments data for the
    selected lexica. Add the new std. dev. of the average and produce a new calibration file.
    """
    import csv
    import pandas as pd

    # Load mean & std. deviation adjusters for the subset of lexica in lexnames
    with open(calibrationfile, 'r', encoding='utf-8', errors='ignore') as scalerfile:
        scalercsv = csv.reader(scalerfile)

        # Read & parse header
        headerrow = next(scalercsv)
        name = headerrow[0]
        nrobs = int(headerrow[1])
        nrfeatures = int(headerrow[2])
        stdev_adj = float(headerrow[3])
        descriptor = headerrow[4]

        # Read & parse individual features
        nrfeaturesused = 0
        featuresused, means, stdevs, variances = [], [], [], []
        for row in scalercsv:
            if len(lexnames) == 0 or (len(lexnames) > 0 and row[0] in lexnames):
                featuresused.append(row[0])
                nrfeaturesused += 1
                means.append(float(row[1]))
                stdevs.append(float(row[2]))
                if includevar:
                    variances.append(float(row[3]))

    # Calculate standard deviation of the average of the subset
    # Note: if there are any missings with a special value (e.g. -999) rather than NaN,
    # this will mess things up! See the more complex averaging code in calibrate_valencedata
    df = pd.read_csv(calibratedfile)
    df['avg'] = df[[lexname + 'X' for lexname in lexnames]].mean(axis=1)
    new_stdev_adj = df['avg'].std()

    # Write the new subsetted scaler info
    newname = name + ' subset'
    newdescriptor = descriptor + ' (subsetted)'
    with open(calibrationfile_subset, 'w', encoding='utf-8') as scalerfile:
        scalercsv = csv.writer(scalerfile)
        # Write header
        scalercsv.writerow((newname, nrobs, nrfeaturesused, new_stdev_adj, newdescriptor))
        # Write individual feature info
        scalercsv.writerows(zip(featuresused, means, stdevs, variances))


# ****************************************** Auxiliary functions **********************************


def average_nonmissing(row, cols, missing=-999, adjustfactor=1, countmissings=False):
    """Average the non-missing columns in cols, adjusting denominator"""
    nonmissingcols = [row[col] for col in cols if row[col] != missing]
    return missing if len(nonmissingcols) == 0 \
        else sum(nonmissingcols)/((len(cols) if countmissings else len(nonmissingcols)) * adjustfactor)


def convert_scaler(oldfile, newfile, scalername):
    """Function to convert old (.pkl) sentiment calibration files to the current csv format"""
    import pickle

    with open(oldfile, 'rb') as neutralfile:
        neutralscaler, stdev_adj, featurenames, descriptor = \
            pickle.load(neutralfile, encoding='latin1')
        write_scaler_tocsv(newfile, neutralscaler, featurenames,
                           name=scalername, descriptor=descriptor, stdev_adj=stdev_adj)
