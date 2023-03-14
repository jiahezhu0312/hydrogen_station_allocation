import networkx as nx
from ortools.linear_solver import pywraplp
from tqdm.auto import tqdm

def FRLM(cn, p=10,  R = 400000 ):
    """Linear programming sample."""
   
    # p = 150 # number of refuel point

    ODs = [k for k, v in cn.nodes(data='is_OD') if v] # node names of OD nodes in graph
    od_pairs = [(n1, n2) for n1 in ODs for n2 in ODs if n1 != n2] #tuples of posible od pairs 
    N = [nx.shortest_path(cn, n1, n2) for n1, n2 in od_pairs] # shortest path for each od pair
    D = [nx.path_weight(cn, q, 'weight') for q in N] # length of shortest_path for each od pair
    M_ = [[ q[i] for i in range(1, len(q)) if nx.path_weight(cn, q[:i], 'weight') > R/2] for q in N] #f0r each od pair the set of candidate sites beyong the origion and not within R/2
    M = [len(s) for s in M_]  #f0r each od pair the numbre of candidate sites beyong the origion and not within R/2
    F = [nx.path_weight(cn, q, 'traffic flow') for q in N] # sum of flow on each pair

    def accessible_candidate_site(G, path, m, t):
        """set of accessible candidate site from mth candidate on path q"""
        if t == 0 and m == 1:
            return [path[r] for r in range(len(path)-m, len(path)) if nx.path_weight(cn, path[m:r+1], 'weight') <= R/2]
        elif t == 0 and m > 1:
            return [path[r] for r in range(len(path)-m, len(path))  if nx.path_weight(cn, path[m:r+1], 'weight') < R]
        elif t == 1 and m > 0:
            return [path[r] for r in range(len(path)-m, len(path)) if nx.path_weight(cn, path[m:r+1], 'weight') <= R]
        # Instantiate a Glop solver, naming it LinearExample.
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return

    # Create the two variables and let them take on any non-negative value.

    y = [solver.BoolVar('y_' + str(q)) for q in range(len(N))]
    x = { i: solver.BoolVar('x_' + str(i)) for i in cn.nodes }
    b =[ [[ solver.BoolVar('b^' + str(q) + '_0' + str(m)) for m in range(len(N[q]))] for q in range(len(N)) ]
  ,[[ solver.BoolVar('b^' + str(q) + '_1' + str(m)) for m in range(len(N[q]))] for q in range(len(N)) ]]

    # print('Number of variables =', solver.NumVariables())

    # Constraint 6
    for q in tqdm(range(len(N))):
        solver.Add(b[1][q][M[q]+1] - sum([x[N[q][M[q]+1]]]+ [x[n] for n in accessible_candidate_site(cn, N[q], M[q]+1, 1)]) <= 0)
        solver.Add(b[0][q][M[q]+1]==0)
        solver.Add(sum([sum([ b[t][q][m] for t in [0, 1]]) for m in range(1, M[q]+2)]) == (M[q]+1) * y[q])

        for t in [0, 1]:
            
            if M[q] != 0:
                for m in range(1, M[q]):
                    solver.Add(b[t][q][m] + (-1) **t * x[N[q][m]] <= 1-t)
                    solver.Add(b[t][q][m] - sum( [x[n] for n in accessible_candidate_site(cn, N[q], m, t)]) <= 0)


   

    # Constrqint 11

    solver.Add(sum(list(x.values())) == p)

    # print('Number of constraints =', solver.NumConstraints())

    # Objective function: maximize refulable traffic flow
    solver.Maximize(sum([F[q] * y[q] for q in range(len(N))]))

    # # Solve the system.
    status = solver.Solve()
   

    demand_satisfied =  sum([F[q] * y[q].solution_value() for q in range(len(N))]) / sum(F)
    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print(f'Percentage of demande satisfied: {demand_satisfied}')

    return {k:v.solution_value() for k, v in x.items()}, [ele.solution_value() for ele in y], demand_satisfied



