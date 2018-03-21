####################################################################################
### wavepacket.py
### Created by Brian Pickens, Nate Chrisman, and Andrew Shurman
###
### Creates and calculates a full simulation dataset for all time steps
### Object's dataset is later passed to visualization.py for animation
####################################################################################

import numpy as np
import visualization as view
import matplotlib.pyplot as plt

################################
### Begin Wavepacket()
################################
class Wavepacket():

    ################################
    ### Begin __init__()
    ################################
    def __init__(self, latticeSize, totalSeconds, wavepacket = "Gaussian", potential = "Infinite Box", is3D = False):
        """
        Initializes wavepacket simulation parameters and creates necessary arrays
        Arguments
        __________
        latticeSize : integer
            3D shape size of the lattice, which is the resolution of the simulation. Ideally an odd number.
        totalSeconds : integer
            Total seconds simulated
        wavePacket : String
            What kind of initial conditions to set. Currently only accepts "Gaussian"
        potential : String
            What potential to bound the wavepacket in. Currently only accepts "Infinite Box"
        3D : boolean
            Simulate in 3D if true, 2D if false
        Returns Nothing
        """
        # Simulation constants
        self.is3D = is3D
        self.fps = 24
        if self.is3D == True:
            self.sigma = 0.5
            self.deltaX = 0.15 # 0.02
            self.deltaT = 0.25 * (self.deltaX**3)
            self.alpha = 0.5 * self.deltaT/(self.deltaX**2)
        else:
            self.sigma = 0.5
            self.deltaX = 0.02
            self.deltaT = 0.5 * (self.deltaX*self.deltaX)
            self.alpha = 0.25 * self.deltaT/(self.deltaX*self.deltaX)

        # Initial Position
        self.origin = np.ceil(latticeSize / 2)
        self.initialX = self.origin
        self.initialY = self.origin
        self.initialZ = self.origin

        # Initial Momentum
        self.initialkX = 17 * np.pi
        self.initialkY = 17 * np.pi
        self.initialkZ = 17 * np.pi

        # Set simulation parameters
        self.time = 0
        self.latticeSize = latticeSize
        self.totalTime = int((totalSeconds * self.fps) * 2) # Times 2 to include the half time steps
        self.wavepacket = wavepacket
        self.potential = potential

        # Create simulation arrays
        if self.potential == "Infinite Box":
            self.V = np.zeros([self.latticeSize, self.latticeSize, self.latticeSize])

        if self.potential == "Harmonic":
            self.V = np.zeros([self.latticeSize, self.latticeSize, self.latticeSize])
            for x in range(self.latticeSize):
                for y in range(self.latticeSize):
                    for z in range(self.latticeSize):
                        self.V[x, y, z] = 0.5 * ((x - self.origin)**2 + (y - self.origin)**2 + (z - self.origin)**2)

        self.latticeComplex = np.zeros([self.latticeSize, self.latticeSize, self.latticeSize], dtype = complex)
        self.latticeReal = np.zeros([self.totalTime, self.latticeSize, self.latticeSize, self.latticeSize])
        self.latticeImag = np.zeros_like(self.latticeReal)

        # Half the size of real and imag
        self.probDataSet = np.zeros([int(self.totalTime / 2), self.latticeSize, self.latticeSize, self.latticeSize])

        # Returns nothing
        return

    ################################
    ### Begin createLattice()
    ################################
    def createLattice(self):
        """
        Creates a simulation lattice and sets the initial conditions inside the lattice.
        Arguments
        __________
        self : object
            uses parameters set by __init__()
        Returns Nothing
        """

        if self.wavepacket == "Gaussian":

            if self.is3D:
                # 3D Gaussian initial conditions here
                for x in range(self.latticeSize):
                    for y in range(self.latticeSize):
                        for z in range(self.latticeSize):
                            self.latticeComplex[x, y, z] = (np.exp(1j * self.initialkX * x)
                                                                          * np.exp(1j * self.initialkY * y)
                                                                          * np.exp(1j * self.initialkZ * z)
                                                                          * np.exp(-1 * (x - self.initialX)**2 / (2 * self.sigma**2))
                                                                          * np.exp(-1 * (y - self.initialY)**2 / (2 * self.sigma**2))
                                                                          * np.exp(-1 * (z - self.initialZ)**2 / (2 * self.sigma**2)))
                print("-[wavepacket.py] Created 3D lattice with {0} initial conditions ({1} potential)".format(self.wavepacket, self.potential))

            else:
                # 2D Gaussian initial conditions
                # Fill self.latticeComplex with Gaussian wavepacket values
                for x in range(self.latticeSize):
                    for y in range(self.latticeSize):
                        self.latticeComplex[x, y, :] = (np.exp(1j * self.initialkX * x)
                                                                      * np.exp(1j * self.initialkY * y)
                                                                      * np.exp(-0.5 * ((x - self.initialX) / (self.sigma))**2)
                                                                      * np.exp(-0.5 * ((y - self.initialY) / (self.sigma))**2))
                print("-[wavepacket.py] Created 2D lattice with {0} initial conditions ({1} potential)".format(self.wavepacket, self.potential))

            # Split self.latticeComplex into real and imaginary components, fill lattice arrays at t = 0
            self.latticeReal[0] = np.real(self.latticeComplex)
            self.latticeImag[1] = np.imag(self.latticeComplex)

            if self.potential == "Infinite Box":
                self.latticeReal[:, 0, :, :] = 0
                self.latticeReal[:, self.latticeSize - 1, :, :] = 0
                self.latticeReal[:, :, 0, :] = 0
                self.latticeReal[:, :, self.latticeSize - 1, :] = 0
                self.latticeReal[:, :, :, 0] = 0
                self.latticeReal[:, :, :, self.latticeSize - 1] = 0
                self.latticeImag[:, 0, :, :] = 0
                self.latticeImag[:, self.latticeSize - 1, :, :] = 0
                self.latticeImag[:, :, 0, :] = 0
                self.latticeImag[:, :, self.latticeSize - 1, :] = 0
                self.latticeImag[:, :, :, 0] = 0
                self.latticeImag[:, :, :, self.latticeSize - 1] = 0

        else:
            print("-[wavepacket.py] createLattice(): Unknown wavepacket type for initial conditions! (use 'Gaussian')")
            raise ValueError

        # Returns nothing
        return

    ################################



    ################################
    ### Begin integrateLattice()
    ################################
    def integrateLattice(self):
        """
        Integrates lattices using the intial conditions.
        Arguments
        __________
        self : object
            uses parameters set by __init__()
        Returns Nothing
        """
        assert self.latticeReal.shape == self.latticeImag.shape, "Real and Imaginary arrays must be the same shape!"

        if self.is3D:
            for t in range(1, self.totalTime - 1):
                # Calculate next real and imaginary wavefunction, excluding borders
                self.latticeReal[t + 1, 1:-1, 1:-1, 1:-1] = (self.latticeReal[t - 1, 1:-1, 1:-1, 1:-1]
                                                                                        + 2 * ((4 * self.alpha + 0.5 * self.deltaT * self.V[1:-1, 1:-1, 1:-1]) * self.latticeImag[t, 1:-1, 1:-1, 1:-1]
                                                                                         - self.alpha * (self.latticeImag[t, 2:, 1:-1, 1:-1] + self.latticeImag[t, :-2, 1:-1, 1:-1]
                                                                                        + self.latticeImag[t, 1:-1, 2:, 1:-1] + self.latticeImag[t, 1:-1, :-2, 1:-1]
                                                                                        + self.latticeImag[t, 1:-1, 1:-1, 2:] + self.latticeImag[t, 1:-1, 1:-1, :-2])))
                self.latticeImag[t + 1, 1:-1, 1:-1, 1:-1] = (self.latticeImag[t - 1, 1:-1, 1:-1, 1:-1]
                                                                                     - 2 * ((4 * self.alpha + 0.5 * self.deltaT * self.V[1:-1, 1:-1, 1:-1]) * self.latticeReal[t, 1:-1, 1:-1, 1:-1]
                                                                                    + self.alpha * (self.latticeReal[t, 2:, 1:-1, 1:-1] + self.latticeReal[t, :-2, 1:-1, 1:-1]
                                                                                    + self.latticeReal[t, 1:-1, 2:, 1:-1] + self.latticeReal[t, 1:-1, :-2, 1:-1]
                                                                                    + self.latticeReal[t, 1:-1, 1:-1, 2:] + self.latticeReal[t, 1:-1, 1:-1, :-2])))
                # If we're doing the Infinite Box potential, set all edges to 0 due to infinite boundary
                if self.potential == "Infinite Box":
                    self.latticeReal[t + 1, 0, :, :] = 0
                    self.latticeReal[t + 1, self.latticeSize - 1, :, :] = 0
                    self.latticeReal[t + 1, :, 0, :] = 0
                    self.latticeReal[t + 1, :, self.latticeSize - 1, :] = 0
                    self.latticeReal[t + 1, :, :, 0] = 0
                    self.latticeReal[t + 1, :, :, self.latticeSize - 1] = 0
                    self.latticeImag[t + 1, 0, :, :] = 0
                    self.latticeImag[t + 1, self.latticeSize - 1, :, :] = 0
                    self.latticeImag[t + 1, :, 0, :] = 0
                    self.latticeImag[t + 1, :, self.latticeSize - 1, :] = 0
                    self.latticeImag[t + 1, :, :, 0] = 0
                    self.latticeImag[t + 1, :, :, self.latticeSize - 1] = 0

                print("-[wavepacket.py] [{}%] Simulating real and imaginary wavefunctions in 3D...".format(100 * (np.floor(t / self.totalTime))), end = '\r')
        else:
            for t in range(1, self.totalTime - 1):
                # Calculate next real and imaginary wavefunction, excluding borders
                self.latticeReal[t + 1, 1:-1, 1:-1, 0] = (self.latticeReal[t - 1, 1:-1, 1:-1, 0]
                                                                                        + 2 * ((4 * self.alpha + 0.5 * self.deltaT * self.V[1:-1, 1:-1, 0]) * self.latticeImag[t, 1:-1, 1:-1, 0]
                                                                                         - self.alpha * (self.latticeImag[t, 2:, 1:-1, 0] + self.latticeImag[t, :-2, 1:-1, 0]
                                                                                        + self.latticeImag[t, 1:-1, 2:, 0] + self.latticeImag[t, 1:-1, :-2, 0])))
                self.latticeImag[t + 1, 1:-1, 1:-1, 0] = (self.latticeImag[t - 1, 1:-1, 1:-1, 0]
                                                                                     - 2 * ((4 * self.alpha + 0.5 * self.deltaT * self.V[1:-1, 1:-1, 0]) * self.latticeReal[t, 1:-1, 1:-1, 0]
                                                                                    + self.alpha * (self.latticeReal[t, 2:, 1:-1, 0] + self.latticeReal[t, :-2, 1:-1, 0]
                                                                                    + self.latticeReal[t, 1:-1, 2:, 0] + self.latticeReal[t, 1:-1, :-2, 0])))

                # If we're doing the Infinite Box potential, set all edges to 0 due to infinite boundary
                if self.potential == "Infinite Box":
                    self.latticeReal[t + 1, 0, :, 0] = 0
                    self.latticeReal[t + 1, self.latticeSize - 1, :, 0] = 0
                    self.latticeReal[t + 1, :, 0, 0] = 0
                    self.latticeReal[t + 1, :, self.latticeSize - 1, 0] = 0
                    self.latticeImag[t + 1, 0, :, 0] = 0
                    self.latticeImag[t + 1, self.latticeSize - 1, :, 0] = 0
                    self.latticeImag[t + 1, :, 0, 0] = 0
                    self.latticeImag[t + 1, :, self.latticeSize - 1, 0] = 0
                print("-[wavepacket.py] [{}%] Simulating real and imaginary wavefunctions in 2D...".format(100 * (np.floor(t / self.totalTime))), end = '\r')

            print("-[wavepacket.py] [100%] Completed simulation of real and imaginary wavefunctions in 2D                          ")

        return
    ################################



    ################################
    ### Begin calculateProbability()
    ################################
    def calculateProbability(self):
        """
        Fills probability data set using integrated real and imaginary data sets.
        Note that the probability data array does not have half time steps,
        and thus will be half the size of the real and imaginary arrays
        Arguments
        __________
        self : object
            uses parameters set by __init__()
        Returns Nothing
        """
        if self.is3D:
            for t in range(self.totalTime - 1):
                # Ignore half time steps
                if t % 2 == 0:
                    self.probDataSet[int(t / 2), :, :, :] = self.latticeReal[t, :, :, :]**2 + self.latticeImag[t + 1, :, :, :] * self.latticeImag[t - 1, :, :, :]
                #else:
                #    self.probDataSet[int(t / 2), :, :, :] = self.latticeImag[t, :, :, :]**2 + self.latticeReal[t + 1, :, :, :] * self.latticeReal[t - 1, :, :, :]
                print("-[wavepacket.py] [{}%] Calculating probability in 3D...".format(100 * (np.floor(t / self.totalTime))), end = '\r')

        else:
            plt.figure()
            for t in range(self.totalTime - 1):
                # Ignore half time steps
                if t % 2 == 0:
                    self.probDataSet[int(t / 2), :, :, 0] = self.latticeReal[t, :, :, 0]**2 + self.latticeImag[t + 1, :, :, 0] * self.latticeImag[t - 1, :, :, 0]
                #else:
                #    self.probDataSet[int(t / 2), :, :, 0] = self.latticeImag[t, :, :, 0]**2 + self.latticeReal[t + 1, :, :, 0] * self.latticeReal[t - 1, :, :, 0]
                print("-[wavepacket.py] [{}%] Calculating probability in 2D...".format(100 * (np.floor(t / self.totalTime))), end = '\r')
                plt.plot(self.probDataSet[int(t / 2), :, :, 0])

            # Quirk of yt's 2D rendering: Need to set all z values equal to each other
            for z in range(self.latticeSize):
                self.probDataSet[:, :, :, z] = self.probDataSet[:, :, :, 0]

        plt.ylabel(r'Probability P$(r)$')
        plt.xlabel(r'Position $(r)$')
        plt.title(r'$2$D Probability Distribution')
        plt.savefig('2D_Harmonic_Prob_Distribution.png')
        plt.show()

        # Clean up giant array
        del self.latticeReal
        del self.latticeImag
        print("-[wavepacket.py] [100%] Completed probability calculations")

        return
    ################################

################################

##### Implementation of simulation

sim3D = False

wavePacket = Wavepacket(33, 4, "Gaussian", "Harmonic", sim3D)

wavePacket.createLattice()
wavePacket.integrateLattice()
wavePacket.calculateProbability()

"""Must replace '0' with '(0.5 * np.pi)' for 2D simulation in view.animateDataSet below"""
view.animateDataSet("Test3DHarmonic", wavePacket.probDataSet, 0, sim3D) # (0.5 * np.pi)
