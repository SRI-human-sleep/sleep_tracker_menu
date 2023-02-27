import pandas as pd

def file_preprocessing(file_name, ncanda_dataset_name, to_drop_name, night_in_processing=None):

    ncanda_dataset = pd.read_excel(ncanda_dataset_name)
    ncanda_dataset = ncanda_dataset.loc[:, ["ID_full", "Gender"]]
    ncanda_dataset.index = ncanda_dataset.ID_full
    ncanda_dataset = ncanda_dataset.drop(columns=["ID_full"])

    to_drop = pd.read_excel(to_drop_name).loc[:, "ID_to_drop"]
    to_drop = to_drop.to_list()

    file = pd.read_csv(file_name)
    file.reset_index(inplace=True)
    id_to_keep = [i[0] for i in file.iterrows() if i[1].ID not in to_drop]

    file = file.iloc[id_to_keep, :]

    dict_to_rep = {
        "Devika": "devika", "Fiona": 'fiona', "Fiona_FINAL": 'fiona', 'JG': 'justin',
        "Justin Greco": "justin", "Lena": 'lena', "Leo": "leo", "Max": "max",
        "Max_FINAL": "max", "Max_Final": "max", "SC": "sc", "Sarah": "sarah"
    }

    file["Scorer"] = file["Scorer"].replace(dict_to_rep)
    file = file.where(file["Scorer"] != 'devika').dropna(how="all")
    file = file.where(file["Scorer"] != 'leo').dropna(how='all')
    file = file.where(file["ID"] != "B-00350-M-2_N3_0YFU").dropna(how='all')
    file = file.where(file["ID"] != "A-00101-M-8_N2_0YFU").dropna(how='all')

    file.night = file.night.replace({'architecture': 'first_night', 'adaptation': 'first_night'})
    if night_in_processing == "overall":
        pass
    else:
        file = file.where(file.night == night_in_processing).dropna(how='all')

    saving_path = r'C:\Users\e34476\OneDrive - SRI International\NCANDA\data\paper\second_paper\pipeline_run'

    file = file.dropna()

    id_to_retain = file.ID.drop_duplicates()
    file = file.loc[id_to_retain.index, ["age", "sex", "night", "site", "Scorer"]]
    file.index = id_to_retain

    file["age"] = file["age"].round(0)
    unique_identifier = pd.Series(
        map(
            lambda x: str(x)[:str(x).find("_")], list(file.index)
        ),
        name="ID",
        index=file.index
    )

    file = pd.concat([file, unique_identifier], axis=1)
    file = file.join(ncanda_dataset)
    file = file.drop(columns=["sex"])
    file.rename(columns={"Gender": "sex"}, inplace=True)
    return file
