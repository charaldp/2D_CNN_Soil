import os.path
import json
import csv


def get_JSON_data(file_in):
    with open(file_in) as data_file:
        data = json.load(data_file)
    return data


class SpectraParser:
    def __init__(self, path, read_complete=False):
        """ path to file of spectra JSON """
        if not os.path.isfile(path):
            raise RuntimeError(path + " is not a valid file")
        if os.path.splitext(path)[1] != ".json":
            raise RuntimeError(path + " does not end in .json")
        self.file = path
        j_data = get_JSON_data(self.file)
        self.input_spectra = j_data["input_spectra"]
        if read_complete and "reduced" in self.input_spectra:
            self.input_spectra = self.input_spectra.replace("reduced", "complete")
        self.skip_first_column = bool(j_data["skip_first_column"])
        self.output_file = j_data["output_file"]
        self.output_column = int(j_data["output_column"])
        self.read_lines = j_data["read_lines"]
        self.tst_indices = j_data["tst_indices"]
        self.cal_folds = j_data["cal_folds"]
        self.uses_fmi = "use_features" in j_data
        if self.uses_fmi:
            self.uses_features = j_data["use_features"]
        self.has_auxiliary = "auxiliary_predictors" in j_data
        if self.has_auxiliary:
            self.auxiliary = j_data["auxiliary_predictors"]

    def x(self):
        """ Returns the X of the dataset (i.e. the spectra) """
        if not os.path.isfile(self.input_spectra):
            raise RuntimeError(self.input_spectra + " is not a valid file")
        x = []
        with open(self.input_spectra, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            #csvreader.next() # skip header
            next(csvreader) # skip header
            for idx, row in enumerate(csvreader):
                if idx in self.read_lines:
                    spectrum = row[1:] if self.skip_first_column else row
                    if self.uses_fmi:
                        s = [float(spectrum[i]) for i in self.uses_features]
                        x.append(s)
                    else:
                        x.append([float(v) for v in spectrum])
        return x

    def x_names(self):
        """ Returns the feature names (i.e. the wavelengths) """
        if not os.path.isfile(self.input_spectra):
            raise RuntimeError(self.input_spectra + " is not a valid file")
        names = []
        with open(self.input_spectra, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            names = next(csvreader)
        return names[1:] if self.skip_first_column else names

    def y(self, output_column=-1):
        """ Returns an output column from the output file """
        if output_column == -1:
            output_column = self.output_column
        if not os.path.isfile(self.output_file):
            raise RuntimeError(self.output_file + " is not a valid file")
        y = []
        with open(self.output_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            #csvreader.next() # skip header
            next(csvreader) # skip header
            for idx, row in enumerate(csvreader):
                if idx in self.read_lines:
                    try:
                        y.append(float(row[output_column]))
                    except ValueError as e:
                        y.append(0)
        return y

    def k_fold(self, k):
        """ Get a pair of [training, validation] indices for the k-th fold """
        val = self.cal_folds[k]
        trn = []
        for i in range(len(self.cal_folds)):
            if i != k:
                trn.append([idx for idx in self.cal_folds[i]])
        trn = sorted([item for sublist in trn for item in sublist])
        return [trn, val]
