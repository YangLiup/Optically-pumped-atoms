# simulator.py --- 
# 
# Filename: simulator.py
# Description: 
# 
# Author:    Yu Lu
# Email:     yulu@utexas.edu
# Github:    https://github.com/SuperYuLu 
# 
# Created: Wed Nov 14 00:34:58 2018 (-0600)
# Version: 
# Last-Updated: Fri Nov 16 00:24:53 2018 (-0600)
#           By: yulu
#     Update #: 152
# 
from .optPumping import OptPumping
from .plot import plotPopulation, plotParameterScan
import numpy as np
import pickle
import sys

class Simulator:
    """Optical Pumping Simulator 
    
    Taking parameter configurations and run the simulation 
    through time steps under the parameters
    """
    
    def __init__(self,
                 Dline1 = 'D1',
                 Dline2 = 'D2',
                 excited_hpf_state1 = 'F1',
                 excited_hpf_state2 = 'F2',
                 I1 = 0.0,
                 I2 = 0.0,
                 detune1 = 5,
                 detune2 = 2,
                 polarization1 = 'pi',
                 polarization2 = 'pi',
                 maxSimulationTime = 500e-6,
                 dt = 10e-9,
                 autoStop = True,
                 config = None):
        
        self.Dline1 = Dline1
        self.Dline2 = Dline2
        self.excited_hpf_state1 = excited_hpf_state1
        self.excited_hpf_state2 = excited_hpf_state2
        self.I1 = I1 # [mW/cm^2]
        self.I2 = I2 # [mW/cm^2]
        self.detune1 = detune1 # [MHz]
        self.detune2 = detune2 # [MHz]
        self.polarization1 = polarization1
        self.polarization2 = polarization2
        self.maxSimulationTime = maxSimulationTime # [sec] # [sec]
        self.dt = dt # [sec]
        self.autoStop = autoStop
        self.numSteps = int(maxSimulationTime / dt)

        if config:
            self.__init__(**self.parseInput(config))

    @classmethod
    def generator(cls, *args, **kwargs):
        return cls.__init__(*args, **kwargs)
    
    def __str__(self):
        return self.dic2str()

    @staticmethod
    def parseInput(infile):
        """
        read input from a file as a dictionary
        """
        params = {}
        # if it is a sys.stdin input 
        if str(type(infile)) == "<class '_io.TextIOWrapper'>":
            for line in infile:
                x = line.rstrip().split(':')
                try:
                    params[x[0].strip()] = float(x[1].split('#')[0].strip())
                except ValueError:
                    params[x[0].strip()] = x[1].split('#')[0].strip()
        # else read from a input file 
        else:
            with open(infile, 'r') as f:
                for line in f:
                    x = line.rstrip().split(':')
                    try:
                        params[x[0].strip()] = float(x[1].split('#')[0].strip())
                    except ValueError:
                        params[x[0].strip()] = x[1].split('#')[0].strip()
                        
        return params
    
    def dic2str(self):
        temp = []
        for key, value in self.__dict__.items():
            temp.append('{0:>20s}\t{1:<10s}'.format(str(key), str(value)))
        return '\n'.join(temp)
                    
    @staticmethod
    def nicePrintStates(pop):
        """
        nicely print the final states population to stdout
        """
        fState = list(pop.keys())
        for f in fState:
            print("\nhpf state:", f)
            print("====================")
            for i,p in enumerate(pop[f][0]):
                mF = -int(f[-1]) + i
                print("mF = {0:1d}{1:10.4f}".format(mF, p))
            print("\n")
    
    def simulate(self):
        # Creat and initialize optPumping object 
        p = OptPumping(self.Dline1,self.Dline2, self.excited_hpf_state1, self.excited_hpf_state2,self.polarization1, self.polarization2)

        # Convert mW/cm^2 to W/m^2
        I1 = self.I1 * 10 
        I2 = self.I2 * 10
        
        # Initialize stop conditions
        steadyIdx = 0 # Step index when reaching steady state 
        breakIdx = 0 # Step index when break simulation
        autoStop = self.autoStop
        
        # Initialization of population dictionary 
        popG = {} # Ground states population dictionary, dic of list of 2d array
        popE1 = {} # Excited states population dictionary, dic of list of 2d array
        popE2 = {} # Excited states population dictionary, dic of list of 2d array

        print("\nProgress:")
        iter_5pct_num = self.numSteps * 0.05
        for i in range(self.numSteps):
            if i % iter_5pct_num == 0:
                print("%d%%" %int(i/self.numSteps * 100), end = ' ')
                sys.stdout.flush()
            if i == 0:
                # Initial states
                popG['F1'] = [p.pop_Ground['F1']]
                popG['F2'] = [p.pop_Ground['F2']]
                
                for f in p.eStates1:
                    popE1[f] = [p.pop_Excited[f]]
                for f in p.eStates2:
                    popE2[f] = [p.pop_Excited[f]]                   
                sumE = popG['F2'][0].sum()
            else:
                newPopG = p.calGroundPop(popG, popE1,popE2, i-1, I1, I2, self.detune1, self.detune2, self.dt)
                newPopE1, newPopE2= p.calExcitedPop(popG,  popE1,popE2, i-1, I1, I2, self.detune1, self.detune2, self.dt)
                for f in p.eStates1:
                    popE1[f].append(newPopE1[f])
                for f in p.eStates1:
                    popE2[f].append(newPopE2[f])
                
                popG['F1'].append(newPopG['F1'])
                popG['F2'].append(newPopG['F2'])

                # popG['F2'].append(newPopG['F2'])
                # unitCheck = p.checkUniformity(newPopG, newPopE)
            
                # if abs( unitCheck- 1) > 0.1:
                #     print("Total population: ", unitCheck, " off too much, cycle: ", i)
                #     return 0

                # Check if steady state reached, then auto break loop
                if autoStop and i > 10:
                    if (abs(popG['F2'][i - 5] - popG['F2'][i])  <  1000 * self.dt).all():
                        steadyIdx = i
                        breakIdx = steadyIdx + int(5e-6 / self.dt)
                        autoStop = False
                    else:
                        pass

                if i == breakIdx:
                    print('\n[*] Steady state reached ! Auto stop ...')
                    autoStop = True
                    break
        clock = np.linspace(0, self.dt * breakIdx, breakIdx+1) if breakIdx else np.linspace(0, self.maxSimulationTime, self.numSteps) # in seconds 
        return (clock, popG, steadyIdx)

    def run(self, verbose = True, plot = True, saveFig = True, saveData = False):
        
        clock, popG, steadyIdx  = self.simulate()

        if verbose: # print to terminal the steady state info 
            if steadyIdx:
                print("\nTime for reaching steady state: {:2.2f} us\n".format(clock[steadyIdx] * 1e6))
                steadyG = {'F1': popG['F1'][steadyIdx],
                           'F2': popG['F2'][steadyIdx]}
                self.nicePrintStates(steadyG)
            else:
                print("\nNo steady state reached, extend the simulation time\nif you want to see it saturates\n")

        if saveData: # save simulation data to pickle file
            toSave = {'params': self.__dict__,
                      'clock': clock,
                      'population_ground': popG,
                      }
            with open('./simulator_output.pickle', 'wb+') as f:
                pickle.dump(toSave, f)
                print("Simulator data save in './simulator_output.pickle'")
        if plot:
            params = {
                "clock": clock,
                "Dline1": self.Dline1,
                "Dline2": self.Dline2,
                "eStates1": [self.excited_hpf_state1],
                "eStates2": [self.excited_hpf_state1],
                "polarization1": self.polarization1,
                "polarization2": self.polarization2,
                "I1": self.I1,
                "I2": self.I2,
                "popG": popG,
                "saveFig": saveFig}
            
            plotPopulation(**params)
