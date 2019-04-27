import csv

def write_to_csv(fname, data_dict):
    """
    fname: name of file to create
    data_dict:
        key = name of patient and file
        value = prediction
    """

    f = open(fname,'w')
    csvwriter = csv.writer(f)

    for patient_data, prediction in data_dict.items():
        csvwriter.writerow([patient_data, prediction])

    f.close()
