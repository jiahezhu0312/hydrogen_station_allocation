import pandas as pd
import geopandas as gpd


def  read_tmja(data_folder:str):
    """Read tmja data and perform basic data preprocessing."""

    tmja = pd.read_csv(data_folder + 'tmja-2019.csv', sep=";")
    tmja["ratio_truck"] = (
        tmja["ratio_PL"].fillna("0").apply(lambda x: float("0" + x.replace(",", ".")))
    )

    tmja["longueur"] = (
        tmja["longueur"].fillna("0").apply(lambda x: float("0" + x.replace(",", ".")))
    )
    tmja["TMJA"] = tmja["TMJA"].fillna(0)
    tmja["xD"] = tmja["xD"].fillna("0").apply(lambda x: float("0" + x.replace(",", ".")))
    tmja["yD"] = tmja["yD"].fillna("0").apply(lambda x: float("0" + x.replace(",", ".")))
    tmja["xF"] = tmja["xF"].fillna("0").apply(lambda x: float("0" + x.replace(",", ".")))
    tmja["yF"] = tmja["yF"].fillna("0").apply(lambda x: float("0" + x.replace(",", ".")))
    tmja = tmja[(tmja["xD"] != tmja["xF"]) | ((tmja["yD"] != tmja["yF"]))]

    tmja["ratio_truck"] = tmja["ratio_truck"].apply(lambda x: x if x < 40 else x / 10)
    tmja["ratio_truck"] = tmja.apply(
        lambda x: x["ratio_truck"]
        if x["ratio_truck"] != 0
        else tmja[tmja["route"] == x["route"]]["ratio_truck"].mean(),
        axis=1,
    )
    tmja["TMJA"] = tmja.apply(
        lambda x: x["TMJA"]
        if x["TMJA"] != 0
        else tmja[tmja["route"] == x["route"]]["TMJA"].mean(),
        axis=1,
    )
    tmja = tmja.astype({"TMJA": float, "longueur": float})
    tmja["TMJA_truck"] = tmja["TMJA"] * tmja["ratio_truck"]
    tmja = tmja[tmja["longueur"] >= 100]
    return tmja


def read_ald(data_folder:str):
    """Read ald data and perform basic data preprocessing."""
    
    detailed = pd.read_excel(data_folder + 'donnees-detaillees.xls', sheet_name=2, header=2)
    detailed.columns = ['e1', 'region', 'numero_aires_region', 'communes_concernees', 'EPL_5000', 'surface', 'P_Transport_et_entreposage'
                        ,'P_commerce', 'P_industrie', 'P_autres', 'salaries_com_entreposage', 'salaries_EPL_5000', 'poids_entrerposage', 'chargement', 'dechargement']

    ald = gpd.read_file(data_folder +'aire-logistiques-donnees-detaillees.zip')
    ald['centroid'] = ald.geometry.centroid

    ald = ald.merge(detailed, on='e1', how='left')
    ald['EPL_5000'] = ald['EPL_5000'].map(lambda x: int(x.split('-')[0]))

    return ald


