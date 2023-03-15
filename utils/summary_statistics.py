from tabulate import tabulate

def print_table(d):
    header = ['Year','Stations(unit)','Small(%)'
            ,'Medium(%)', 'Large(%)'
            , 'Demand Satisfied(%)', 'Yearly Profit(M ton)'
            , 'Construction Cost(M Euro)', 'Operations Cost(M Euro)'
    ]
    print (tabulate(d, headers=header))


def phase_summary(station_size, fulfilled_demand, profit_ton, operation_rate, station_size_all_phase, phase):
    """ Computes summary statistics at each phase.
    
    Inputs:
        station_size (dict): dictionary containing size of nodes
        market_demand (dict): dictionary containing market demand per node
        fulfilled_demand (dict): dictionary containing demand we satisfy per node
        profit_ton (dict): dictionary containing profit per node

        cost (float/int): cost of production per ton of hydrogen
        operation_rate (float): the percentage of days where a station will be in operation
        profit_margin (float): the profit margin to simulate the price
    Returns:
        Tuple of summary statistics

    """
    ns, s, m, l = n_stations(station_size)
    ds = demand_satisfied_per( fulfilled_demand, phase)
    yp = profit_ton_year(profit_ton, operation_rate) / 1e6
    capex, opex = cost(station_size_all_phase)
    s, m, l,  yp=  100 * round(s, 3), 100 *round(m, 3),  100 *round(l, 3),  round(yp, 3)
    if phase in [2025, 2035]:
        return [phase, ns, s, m, l, ds, yp,  capex, opex]
    else:
        print(phase, ds)
        ds = 100 *round(ds, 3)
        return [phase, ns, s, m, l, ds, yp,  capex, opex]


# ===========================================================================
def cost(station_size_all_phase ):
    """ Compute the construction cost and operation cost at the last phase.
    Inputs:
    station_size_all_phase (List[Dict]) = Deployement plan at each phase

    Outputs:
        Construction cost
        Operation cost
    """
    capex = 0
    opex = 0
    cost_dict = {0: 0, 1: 3, 2: 5, 3: 8}
    opex_dict = {0: 0, 1: 0.3, 2: 0.4, 3:0.56}
    if len(station_size_all_phase)==1:
        capex = sum([cost_dict[station_size_all_phase[0][n]] for n in station_size_all_phase[0]])
    else:
        new_stations_nodes = [ n  for n in station_size_all_phase[0] if (station_size_all_phase[-2][n]==0) and (station_size_all_phase[-1][n]>0) ]
        # enlarged_stations_nodes = [ n  for n in station_size_all_phase[0] if (station_size_all_phase[-2][n]!=0) and (station_size_all_phase[-1][n]!=station_size_all_phase[-2][n]) ]
        capex = sum([cost_dict[station_size_all_phase[-1][n]] for n in new_stations_nodes])
        opex = sum([opex_dict[station_size_all_phase[-2][n]] for n in station_size_all_phase[0]])
    return capex, opex
def n_stations(station_size):
    """ Summary of the number of stations in a network.
    Input:
    station_size (dict): dictionary containing size of nodes
        key = node_id 
        value = 0 (no station) 1(small station) 2 (medium station) 3 (large station)
    
    Returns:
        total number of stations
        percentage of small stations
        percentage of medium stations
        percentage of large stations
    
    """
    station_cnt =  len([v for v in station_size.values() if v >0])
    small_cnt = len([v for v in station_size.values() if v ==1]) 
    medium_cnt = len([v for v in station_size.values() if v ==2]) 
    large_cnt = len([v for v in station_size.values() if v ==3]) 
    return  station_cnt, small_cnt / station_cnt, medium_cnt / station_cnt, large_cnt / station_cnt



def demand_satisfied_per(our, phase):
    """ Compute the percentage of demand in t/day satisfied by our deployement strategy.
    Input:
    market (dict): dictionary containing market demand per node
    our (dict): dictionary containing demand we satisfy per node

        key = node_id 
        value = demand in ton/day
    
    """
    total_our = sum(list(our.values()))/1000
    
    if phase == 2025 or phase == 2035:
        print(phase)
        return 'not defined'
    elif phase == 2030:
        return total_our / 384
    else:
        return total_our / 1559

def profit_ton_year(profit_ton, operation_rate):
    """ Compute the profit per year in ton based on operation rate.
    Input:
    profit_ton (dict): dictionary containing profit per node
        key = node_id 
        value = demand in ton/day
    operation_rate (float): the percentage of days where a station will be in operation
    """
    total_profit_ton = sum(list(profit_ton.values()))
    return total_profit_ton * (365 * operation_rate)