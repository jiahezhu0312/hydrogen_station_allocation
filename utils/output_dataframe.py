import pandas as pd
def df_phase(cn, prefix):
    fulfilled_demand = dict(cn.nodes(data=prefix+'_h2day'))
    station_size = dict(cn.nodes(data=prefix+'_station_size'))
    data = {n: [fulfilled_demand[n], station_size[n]] for n in fulfilled_demand}

    return pd.DataFrame.from_dict(data, orient='index',
                       columns=['fulfilled_demand'
                                ,'station_size'])