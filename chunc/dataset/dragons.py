"""
"""
from matplotlib import projections
import numpy as np
import sklearn

projection_types = [
    'signmu_mgaugino',
    'signmu_msfermion',
    'signA0_msfermion',
    'signmu_tanbeta',
    'A0_msfermion'
]

class DragonTransform:

    def __init__(self,
        projections
    ):
        self.projections = projections
        for projection in projections:
            if projection not in projection_types:
                pass

    def get_params(self):
        pass

    def set_params(self):
        pass

    def fit_transform(self,
        X
    ):
        results = []
        for projection in self.projections:
            if projection == 'signmu_mgaugino':
                results.append(X[:,1] * np.sign(X[:,4]))
            elif projection == 'signmu_msfermion':
                results.append(X[:,0] * np.sign(X[:,4]))
            elif projection == 'signA0_msfermion':
                results.append(X[:,0] * np.sign(X[:,2]))
            elif projection == 'signmu_tanbeta':
                results.append(X[:,3] * np.sign(X[:,4]))
            elif projection == 'A0_msfermion':
                results.append(X[:,2]/X[:,0])
        results = np.reshape(results, (-1,len(results)))
        return results