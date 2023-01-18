import random
from matplotlib import pyplot as plt
from math import *
import numpy as np
from scipy.stats import norm 

class Particle:

    def __init__(self, weight, position=0, direction='f'):
        
            # direction :the direction of the particle ( defaults to 'f'=forward ) , ('f' = forward or 'b' = backward).
        
        self.weight = weight
        self.position = position
        self.direction = direction
        self.pdf_val = None

    def move_particle(self, steps, start, end):

        if self.direction == 'f':
            if self.position + steps >= end:       # If it hits the end of the path change direction to backward .
                self.direction = 'b'               # the particle one step in its derection .
                self.move_particle(steps - 1, start, end)
            else:
                self.position += steps

        else:
            if self.position <= start:       # If it hits the start of the path change direction to forward .
                self.direction = 'f'
                self.move_particle(steps + 1, start, end)
            else:
                self.position -= steps
                

    def __lessthan__(self, other):
        return self.weight < other.weight

    def __equal__(self, other):
        return self.weight == other.weight

    def __lessORequal__(self, other):
        return self.weight <= other.weight

    def __notequal__(self, other):
        return self.weight != other.weight

    def __graterthan__(self, other):
        return self.weight > other.weight

    def __gretaerORequale__(self, other):
        return self.weight >= other.weight

    def __repr__(self) -> str:
        return '(weight: ' + str(self.weight) + ', postion: ' + str(self.position) + ')\n'


class ParticlesFilter:
    """ A class that represents a particle filter applied on a robot with 1D movement
        The robot moves either forward or backward. The map is a 1D array that has 
        letters instead of actual barriers and walls so the robot can see if the position
        its at is similar to any other particle that has the same letter
    """

    def __init__(self, path_length=100, no_of_particles=500, robot_init_postion=50):
        """ The constructor

        Args:
            path_length (int): The length of the path for the map. Defaults to 100.
            no_of_particles (int): Defaults to 500.
            robot_init_postion (int): The initial location of the robot. Defaults to 50.
        """
        self.path_length = path_length
        self.no_of_particles = no_of_particles
        self.robot = Particle(weight=1, position=robot_init_postion)    # The robot itself is a particle

        self.particles = []
        self.path = []

        plt.ion()
        plt.rcParams["figure.figsize"] = [20.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
    
    
    
    def function_x(self,theta):
        """ The used function for the path
        """
        return np.cos(theta) + 0.5*np.cos(3*theta+0.23) + 0.5*np.cos(5*theta-0.4) + \
                0.5*np.cos(7*theta+2.09) + 0.5*np.cos(9*theta-3)



    def generate_random_particles(self):
        """ Generates particles with random postion in the path and random direction. 
            The weights assigned is equal to 1 / number of particles
        """
        for _ in range(self.no_of_particles):
            w = 1 / self.no_of_particles
            p = random.randint(0, self.path_length - 1)
            """ ran = random.randint(1,980989809)
            char = 'f'
            if ran%2:
                char = 'b'
            d = char # Rondomizing the direction """
            self.particles.append(Particle(weight=w, position=p, direction='f'))



    def generate_path(self):
        self.positions = np.arange(45, 55, 0.001)
        # the function, which is y=cosθ+1/2*cos(3*θ+0.23) +1/2*cos(5*θ−0.4)+1/2*cos(7*θ+2.09)+1/2*cos(9*θ−3)
        self.path = self.function_x(self.positions)



    def move(self, steps):
        robot_prev_dir = self.robot.direction
        self.robot.move_particle(steps, 0, self.path_length - 1)

        # to check if the robot changed its direction
        change_direction = (robot_prev_dir != self.robot.direction)

        for i in range(len(self.particles)):
            if change_direction:
                if self.particles[i].direction == 'f':
                    self.particles[i].direction = 'b'
                else:
                    self.particles[i].direction = 'f'

            self.particles[i].move_particle(steps, 0, self.path_length-1)
            # particle_pos = self.function_x(self.particles[i].position)
    
    
    
    def update(self):
        #Updates weights using gaussian distribution
        measurements_diffs = [] # differences between particles' measurements and robot's measurement 

        for i in range(len(self.particles)):
            measurements_diffs.append(self.function_x(self.particles[i].position) - self.function_x(self.robot.position))
       
        ar = np.array(measurements_diffs)
        stdv = ar.std()
        #print(stdv, measurements_diffs)
        if stdv <= 0:
            stdv = 0.0000000000001

        # for i in range(len(measurements_diffs)):
        #     measurements_diffs[i] /= stdv

        dist = norm(0, stdv) # define a normal distribution with mean = 0 and the calculated standard deviation
        probabilities = [dist.pdf(value) for value in measurements_diffs] # calculate the probability of each particle using pdf
        sums_of_prob = sum(probabilities)
        print(sums_of_prob)
        for i in range(len(probabilities)):
            probabilities[i] /= sums_of_prob
            
        #print("zero=", dist.pdf(0), max(probabilities))
            
        for i in range(len(self.particles)):
            self.particles[i].weight *= probabilities[i] # update the weights



    def normalize(self):
        norm_arr = []
        max_val = max(self.particles).weight
        min_val = min(self.particles).weight

        diff_arr = max_val - min_val

        for i in range(len(self.particles)):
            norm_arr.append(self.particles[i].weight)

        sum_of_weights = sum(norm_arr)
        
        for i in range(self.no_of_particles):
            self.particles[i].weight = norm_arr[i] / sum_of_weights


    def sample(self):
        new_particles = []
        indexes = []

        # extract weights
        weights = []
        for i in range(len(self.particles)):               
            if not weights:
                weights.append(self.particles[i].weight)
            else:
                weights.append(self.particles[i].weight + weights[-1])
        
        for _ in range(self.no_of_particles):
            index = self.get_random_index(weights)
            if not index:
                index = self.particles.index(max(self.particles))
            indexes.append(index)
            particle = Particle(self.particles[index].weight, self.particles[index].position,
                                self.particles[index].direction)
            new_particles.append(particle)

        self.particles = new_particles.copy()
        new_particles.clear()



    def get_random_index(self, weights):
        N = random.uniform(min(weights), max(weights))
        for index, weight in enumerate(weights):
            if N <= weight:
                return index



    def stop(self, threshold):
        for i in range(len(self.particles)):
            if abs(self.particles[i].position - self.robot.position) > threshold:
                return False
        return True



    def draw(self):
        #fig = plt.figure()
        plt.ylim(-3, 5)
        plt.xlim(0, 10)
        plt.margins(x=1, y=0)
        plt.grid()
        #plt.scatter(x, y)
        plt.xticks(np.arange(10, step=1))
        plt.plot(self.positions, self.path)
        
        positions = [(self.particles[i].position/10) for i in range(
            len(self.particles))]
        
        y = []
        for _ in range(self.no_of_particles):
            y.append(random.uniform(3, 4))

        plt.margins(x=1, y=0)
        plt.scatter(positions, y, marker="o")
        
        plt.plot([self.robot.position / 10], [3], marker="o", markersize=10,
                 markeredgecolor="green", markerfacecolor="red")

        plt.show()
        plt.draw()
        plt.pause(0.5)
        plt.clf()

        
        