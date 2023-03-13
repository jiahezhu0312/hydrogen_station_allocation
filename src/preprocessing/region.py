

def add_region(tmja, ald):
    code2region = {
        '52': 'Pays de la Loire'
        ,'24': 'Centre-Val de Loire'
        ,'28': 'Normandie'
        ,'11': 'Île-de-France'
        ,'32': 'Hauts-de-France'
        ,'44': 'Grand Est'
        ,'75': 'Nouvelle-Aquitaine'
        ,'53': 'Bretagne'
        ,'84': 'Auvergne-Rhône-Alpes'
        ,'76': 'Occitanie'
        ,'93': 'Provence-Alpes-Côte d\'Azur'
        ,'27': 'Bourgogne-Franche-Comté'
    }

    REGIONS = {
    'Auvergne-Rhône-Alpes': ['1', '3', '7', '15', '26', '38', '42', '43', '63', '69', '73', '74'],
    'Bourgogne-Franche-Comté': ['21', '25', '39', '58', '70', '71', '89', '90'],
    'Bretagne': ['35', '22', '56', '29'],
    'Centre-Val de Loire': ['18', '28', '36', '37', '41', '45'],
    'Grand Est': ['8', '10', '51', '52', '54', '55', '57', '67', '68', '88'],
    'Hauts-de-France': ['2', '59', '60', '62', '80'],
    'Île-de-France': ['75', '77', '78', '91', '92', '93', '94', '95'],
    'Normandie': ['14', '27', '50', '61', '76'],
    'Nouvelle-Aquitaine': ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87'],
    'Occitanie': ['9', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82'],
    'Pays de la Loire': ['44', '49', '53', '72', '85'],
    'Provence-Alpes-Côte d\'Azur': ['4', '5', '6', '13', '83', '84'],
    }
    ald = ald.copy()
    tmja = tmja.copy()
    dep2r = {}
    for k, v in REGIONS.items():
        for d in v:
            dep2r[d] = k
    ald['region_name'] = ald.region.map(lambda x: code2region[str(x)])
    tmja['region_name'] = tmja.depPrD.map(lambda x: dep2r[str(x)])
    return tmja, ald

def filter_on_region(tmja, ald, region_name):
    return tmja[tmja.region_name==region_name], ald[ald.region_name==region_name]