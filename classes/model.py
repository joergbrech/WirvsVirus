from pygame.time import get_ticks as time_now
from random import random
import numpy as np

class Model(object):
        def __init__(self):
            self.prameter = {'incubation_time': 1000,
                             'illness_rate': 0.4,
                             'death_rate': 0.02,
                             'survival_time': 2000,
                             'recover_time': 5000,
                             }
        def set_state(self, human):

            if (human.state == 'infected'):
                time_diff = time_now() - human.time_infected
                if (time_diff > self.prameter ['incubation_time']):
                    if (random() < self.prameter['illness_rate']):
                        human.state = 'ill'

            if (human.state == 'ill'): #'ill'):
                time_diff = time_now() - human.time_infected  - self.prameter ['incubation_time']
                if (time_diff > self.prameter['recover_time']):
                    if (random() < self.prameter['death_rate']):
                        human.state = 'dead'
                    else:
                        human.state = 'recovered'
                        
        # A recent study of COVID-19 estimates some of these values for us (Hellewell et al. 2020),
        # so we can use some of their parameter estimates to get our model off the ground.
        # Incubation period = 5 days -> alpha = 0.2
        # R0 = 3.5
        # to get 1/gamma value of 2 days, so gamma = 0.5.
        # Plugging the R0 and gamma values into Equation (6), we get an estimate of beta = 1.75.
        # https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296
        def base_seir_model(self,  init_vals, params, t):
                S_0, E_0, I_0, R_0 = init_vals
                S, E, I, R = [S_0], [E_0], [I_0], [R_0]
                alpha, beta, gamma = params
                dt = t[1] - t[0]
                for _ in t[1:]:
                    next_S = S[-1] - (beta*S[-1]*I[-1])*dt
                    next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt
                    next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
                    next_R = R[-1] + (gamma*I[-1])*dt
                    S.append(next_S)
                    E.append(next_E)
                    I.append(next_I)
                    R.append(next_R)
                return np.stack([S, E, I, R]).T
