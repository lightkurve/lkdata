"""Tools for working with mission catalogs, such as KIC, EPIC, and TIC"""

# This is a lits of VizieR catalogs and their input parameters to be used in the
# query_skycatalog function
CATALOG_DICTIONARY = {
    "kic": {
        "catalog": "V/133/kic",
        "columns": ["KIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "kepmag"],
        "column_filters": "kepmag",
        "rename_in": ("KIC", "pmDE", "kepmag"),
        "rename_out": ("ID", "pmDEC", "Kepler_Mag"),
    },
    "epic": {
        "catalog": "IV/34/epic",
        "columns": ["ID", "RAJ2000", "DEJ2000", "pmRA", "pmDEC", "plx", "Kpmag"],
        "column_filters": "Kpmag",
        "rename_in": ["Kpmag", "plx"],
        "rename_out": ["K2_Mag", "Plx"],
    },
    "tic": {
        "catalog": "IV/39/tic82",
        "columns": ["TIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "Tmag"],
        "column_filters": "Tmag",
        "rename_in": ("TIC", "pmDE", "Tmag"),
        "rename_out": ("ID", "pmDEC", "TESS_Mag"),
    },
    "gaiadr3": {
        "catalog": "I/355/gaiadr3",
        "columns": ["DR3Name", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "Gmag"],
        "column_filters": "Gmag",
        "rename_in": ("DR3Name", "pmDE", "Gmag"),
        "rename_out": ("ID", "pmDEC", "Gaia_G_Mag"),
    },
}
