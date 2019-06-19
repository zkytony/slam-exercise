from grid2d import GridAgent

class GridBayesianAgent(GridAgent):

    def __init__(self, gridworld,
                 s_res, s_maxrange, s_minrange, t0=0):
        super().__init__(gridworld, s_res, s_maxrange, s_minrange, t0=t0)
        
        
