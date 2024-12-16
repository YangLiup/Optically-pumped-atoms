# optPumping.py --- 
# 
# Filename: Functions.py
# Description: 
# 
# Author:    Yu Lu
# Email:     yulu@utexas.edu
# Github:    https://github.com/SuperYuLu 
# 
# Created: Sun Sep 17 16:36:41 2017 (-0500)
# Version: 
# Last-Updated: Wed Nov 14 23:11:08 2018 (-0600)
#           By: yulu
#     Update #: 410
# 

import numpy as np


class OptPumping:
    def __init__(self, Dline1, Dline2, excitedF1,excitedF2, pumpPol1, pumpPol2):
        
        # Load D line transition database and scale natrual linewidth
        # ---------------------------------------------------------------------------
        from .constant import gamma 
        if Dline1 == 'D1':
            if excitedF1 == 'F1':
                self.gamma = gamma * 3 / 8 
                from .TransitionStrength import TransStrengthD1_toF1 as TransStrength1
                from .TransitionStrength import DecayStrengthD1_toF1 as DecayStrength1
            elif excitedF1 == 'F2':
                self.gamma = gamma * 5 / 8 
                from .TransitionStrength import TransStrengthD1_toF2 as TransStrength1
                from .TransitionStrength import DecayStrengthD1_toF2 as DecayStrength1
            else:
                print("D1 line has not excited hpf state ", Dline1)

        elif Dline1 == 'D2':
            if excitedF1 == 'F0':
                self.gamma = gamma * 1 / 16 
                from .TransitionStrength import TransStrengthD2_toF0 as TransStrength1
                from .TransitionStrength import DecayStrengthD2_toF0 as DecayStrength1
            elif excitedF1 == 'F1':
                self.gamma = gamma * 3 / 16 
                from .TransitionStrength import TransStrengthD2_toF1 as TransStrength1
                from .TransitionStrength import DecayStrengthD2_toF1 as DecayStrength1
            elif excitedF1 == 'F2':
                self.gamma = gamma * 5 / 16 
                from .TransitionStrength import TransStrengthD2_toF2 as TransStrength1
                from .TransitionStrength import DecayStrengthD2_toF2 as DecayStrength1
            elif excitedF1 == 'F3':
                self.gamma = gamma * 7 / 16 
                from .TransitionStrength import TransStrengthD2_toF3 as TransStrength1
                from .TransitionStrength import DecayStrengthD2_toF3 as DecayStrength1
            else:
                print("D2 line has not excited hpf state ", Dline1)       
        else:
            print('Unavaliable D line transition !')

        if Dline2 == 'D1':
            if excitedF2 == 'F1':
                self.gamma = gamma * 3 / 8 
                from .TransitionStrength import TransStrengthD1_toF1 as TransStrength2
                from .TransitionStrength import DecayStrengthD1_toF1 as DecayStrength2
            elif excitedF2 == 'F2':
                self.gamma = gamma * 5 / 8 
                from .TransitionStrength import TransStrengthD1_toF2 as TransStrength2
                from .TransitionStrength import DecayStrengthD1_toF2 as DecayStrength2
            else:
                print("D1 line has not excited hpf state ", Dline1)

        elif Dline2 == 'D2':
            if excitedF2 == 'F0':
                self.gamma = gamma * 1 / 16 
                from .TransitionStrength import TransStrengthD2_toF0 as TransStrength2
                from .TransitionStrength import DecayStrengthD2_toF0 as DecayStrength2
            elif excitedF2 == 'F1':
                self.gamma = gamma * 3 / 16 
                from .TransitionStrength import TransStrengthD2_toF1 as TransStrength2
                from .TransitionStrength import DecayStrengthD2_toF1 as DecayStrength2
            elif excitedF2 == 'F2':
                self.gamma = gamma * 5 / 16 
                from .TransitionStrength import TransStrengthD2_toF2 as TransStrength2
                from .TransitionStrength import DecayStrengthD2_toF2 as DecayStrength2
            elif excitedF2 == 'F3':
                self.gamma = gamma * 7 / 16 
                from .TransitionStrength import TransStrengthD2_toF3 as TransStrength2
                from .TransitionStrength import DecayStrengthD2_toF3 as DecayStrength2
            else:
                print("D2 line has not excited hpf state ", Dline2)       
        else:
            print('Unavaliable D line transition !')


        # Initialize pumping matrix based on polarization
        # ---------------------------------------------------------------------------
        try:
            self.pumpMatrix1 = eval('TransStrength1.' + pumpPol1) 
            self.pumpMatrix2 = eval('TransStrength2.' + pumpPol2) 
        except AttributeError:
            print("Incorrect polorization name, please chose one of the following:\n\
            sigmaPlus, sigmaMinux, pi\n")

        # Initialize decay matrix
        # ---------------------------------------------------------------------------
        self.decayMatrix1 = DecayStrength1 
        self.decayMatrix2 = DecayStrength2

        # Initialize transition frequency
        # ---------------------------------------------------------------------------
        self.freq1 = TransStrength1.freq
        self.freq2 = TransStrength2.freq

        # Initialize pump beam polorization
        # ---------------------------------------------------------------------------
        self.pumpPol1 = pumpPol1 # Polorization for pumping beam F1 --> Excited states
        self.pumpPol2 = pumpPol2 # Polorization for pumping beam F2 --> Excited states
        
        # Initialize possible polarization list
        # ---------------------------------------------------------------------------
        self.pol1 = TransStrength1.polarization
        self.pol2 = TransStrength2.polarization

        # Initialize D line value
        # ---------------------------------------------------------------------------
        self.Dline1 = Dline1
        self.Dline2 = Dline2
        
        # Initialize number of excited hyperfine magnetic substates F
        # ---------------------------------------------------------------------------
        # self.numEStates = len(DecayStrength.numSubStates)

        # Initialize excited hyperfine states name
        # ---------------------------------------------------------------------------
        self.eStates1 = TransStrength1.eStates
        self.eStates2 = TransStrength2.eStates
        
        # Initialize ground level population
        # ---------------------------------------------------------------------------
        self.pop_Ground ={
            'F1': np.ones([1,3]) * 1./8,
            'F2': np.ones([1,5]) * 1./8
            }

        # Initialize excited level population
        # ---------------------------------------------------------------------------
        self.pop_Excited1 = {}
        self.pop_Excited2 = {}

        for s,n in zip(DecayStrength1.eStates, DecayStrength1.numSubStates):
            self.pop_Excited1[s] = np.zeros([1, n])

        for l,m in zip(DecayStrength2.eStates, DecayStrength2.numSubStates):
            self.pop_Excited2[l] = np.zeros([1, m])
        # Calculate overall factor for dipole matrix normalization
        # ---------------------------------------------------------------------------
        self.dipoleFactor1, self.dipoleFactor2= self.dipoleScaleFactor()
                
    def dipoleScaleFactor(self):
        """
        Calculate the overall scale factor which leads to:
        Gamma = w^3/(3*pi*e0*hBar*c^3) * sum(all transition matrix squared) * scale factor
        Returned factor is for Metcalf yellow book, Ueg^2
        """
        from .constant import hBar, e0,c
        totTransElement1  = 0 
        for trans in self.decayMatrix1.transition:
            for pol in self.pol1:
                totTransElement1 +=eval('self.decayMatrix1.' + pol + '.' + trans + '.sum()');
        totTransElement2  = 0 
        for trans in self.decayMatrix2.transition:
            for pol in self.pol2:
                totTransElement2 +=eval('self.decayMatrix2.' + pol + '.' + trans + '.sum()'); 
               
        einsteinAFactor1 = (2 * np.pi * self.freq1)**3 / (3 * np.pi * e0 * hBar * c**3)
        einsteinAFactor2 = (2 * np.pi * self.freq2)**3 / (3 * np.pi * e0 * hBar * c**3)
      
        factor1  = self.gamma / (einsteinAFactor1 * totTransElement1)
        factor2  = self.gamma / (einsteinAFactor2 * totTransElement2)

        return factor1, factor2

    
    def reduceMatrix(self,mtx): 
        """
        Accumulate matrix columns to rows, e.g. 
        after apply to shape = (3,4) matrix, it becomes (3, 1) matrix 
        """
        return mtx.sum(axis = 1)

    def einsteinA(self, trans):
        """
        Calculate Einstein A coefficient based on Ueg^2
        """
        from .constant import hBar, e0 ,c
        einsteinAFactor1 = (2 * np.pi * self.freq1)**3 / (3 * np.pi * e0 * hBar * c**3)
        einsteinA1 = einsteinAFactor1 * (trans * self.dipoleFactor1)

        einsteinAFactor2 = (2 * np.pi * self.freq1)**3 / (3 * np.pi * e0 * hBar * c**3)
        einsteinA2 = einsteinAFactor1 * (trans * self.dipoleFactor2)

        return einsteinA1,einsteinA2

    def omega(self, trans, I):
        """
        Calculate rabi frequency based on light intensity 
        and relative transition strength from Metcalf's yellow book
        """
        from .constant import h, e0, c
        Ueg1 = np.sqrt(trans * self.dipoleFactor1)
        Ueg2 = np.sqrt(trans * self.dipoleFactor1)

        return Ueg1 * np.sqrt(2 * I /( e0 * c)) / h, Ueg2 * np.sqrt(2 * I /( e0 * c)) / h
            
    def detuneFactor(self, trans, detune):
        """
        Calculate the detune factor f = gamma / 2 / ((gamma / 2)^2 + delta^2 for 
        light absorption rate: R = Omega^2 * f
        """
        from .constant import e0, hBar, c
        # Calculate natural line width for hpf states
        hpf_gamma1 = (2 * np.pi * self.freq1)**3 /(3 * np.pi * e0 * hBar * c**3) * trans * self.dipoleFactor1
        x, y = hpf_gamma1.shape
        factor1 = np.zeros([x, y])
        for i in range(x):
            for j in range(y):
                # Avoid divided by 0 issue 
                if hpf_gamma1[i,j]== 0:
                    factor1[i,j] = 0
                else:
                    factor1[i,j] = hpf_gamma1[i,j] / 2 / ((hpf_gamma1[i,j] / 2)**2 + detune**2)
                    #print('true')
                    #print(factor[i,j])
        return factor1
    
    
    def calGroundPop(self, popGround, popExcited1, popExcited2, idx, I1, I2, detune1, detune2, dt):
        G1 = popGround['F1'][idx]
        G2 = popGround['F2'][idx]
        newG1 = np.zeros([1, len(G1[0])])
        newG2 = np.zeros([1, len(G2[0])])

        for es in self.eStates1:
            newG1 += -self.reduceMatrix(self.omega(eval("self.pumpMatrix1.F1_" + self.Dline1 + "_" + es), I1)**2/2 * self.detuneFactor(eval("self.pumpMatrix1.F1_" + self.Dline1 + "_" + es), detune1)).T  * G1 \
                     + np.dot(popExcited1[es][idx],  self.einsteinA(eval("self.decayMatrix1.sigmaPlus." + es + "_" + self.Dline1 + "_F1"))) \
                     + np.dot(popExcited1[es][idx], self.einsteinA(eval("self.decayMatrix1.sigmaMinus." + es + "_" + self.Dline1 + "_F1")))\
                     + np.dot(popExcited1[es][idx], self.einsteinA(eval("self.decayMatrix1.pi." + es + "_" + self.Dline1 + "_F1")))           
                 
        for es in self.eStates2:
            newG2 += -self.reduceMatrix(self.omega(eval("self.pumpMatrix2.F2_" + self.Dline2 + "_" + es), I2)**2/2 * self.detuneFactor(eval("self.pumpMatrix1.F2_" + self.Dline2 + "_" + es), detune2)).T * G2 \
                            + np.dot(popExcited2[es][idx], self.einsteinA(eval("self.decayMatrix1.sigmaPlus." + es + "_" + self.Dline1 + "_F2")))\
                            + np.dot(popExcited2[es][idx], self.einsteinA(eval("self.decayMatrix1.sigmaMinus." + es + "_" + self.Dline1 + "_F2")))\
                            + np.dot(popExcited2[es][idx], self.einsteinA(eval("self.decayMatrix1.pi." + es + "_" + self.Dline1 + "_F2")))         
        newG1 = G1 + newG1 * dt  
        newG2 = G2 + newG2 * dt
        pop = {'F1': newG1,\
               'F2': newG2}
        return pop

    def calExcitedPop(self, popGround, popExcited1, popExcited2,idx, I1, I2, detune1, detune2, dt):
        newE1 = {}
        newE2 = {}

        for es in self.eStates1: # loop thru excited states names
            newE1[es] = np.zeros([1, len(popExcited1[es][idx][0])])

        for es in self.eStates2: # loop thru excited states names
            newE2[es] = np.zeros([1, len(popExcited2[es][idx][0])])

        for p in self.pol1:
                for es in self.eStates1: # loop thru excited hyperfine states names 
                # 3.0 factor is to compensate repeating sum of polarization
                    newE1[es] += np.dot(popGround['F1'][idx], self.omega(eval("pumpMatrix1." + 'F1' + "_" + self.Dline1 + "_" + es), I1)**2 /2 * self.detuneFactor(eval("pumpMatrix1." + 'F1' + "_" + self.Dline1 + "_" + es), detune1)) / 3.0 \
                            - self.reduceMatrix(self.einsteinA(eval("self.decayMatrix1." + p + "." + es + "_" + self.Dline1 + "_" + 'F1'))).T * popExcited1[es][idx]
        for p in self.pol2:
                for es in self.eStates2: # loop thru excited hyperfine states names 
                # 3.0 factor is to compensate repeating sum of polarization
                    newE2[es] += np.dot(popGround['F2'][idx], self.omega(eval("pumpMatrix2." + 'F2' + "_" + self.Dline1 + "_" + es), I2)**2 /2 * self.detuneFactor(eval("pumpMatrix2." + 'F2' + "_" + self.Dline2 + "_" + es), detune2)) / 3.0 \
                            - self.reduceMatrix(self.einsteinA(eval("self.decayMatrix2." + p + "." + es + "_" + self.Dline2 + "_" + 'F2'))).T * popExcited2[es][idx]
        for es in self.eStates1:
            newE1[es] = popExcited1[es][idx] + newE1[es] * dt
        for es in self.eStates2:
            newE2[es] = popExcited2[es][idx] + newE2[es] * dt

        return newE1,newE2
                
    
    
    # def checkUniformity(self, popGround,  popExcited):
    #     return popGround['F1'][-1].sum() + popGround['F2'][-1].sum() + sum([popExcited[str(x)][-1].sum() for x in popExcited])
    
