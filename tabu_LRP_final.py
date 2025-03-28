import numpy as np
import colorsys
import matplotlib.pyplot as plt
import json
import pandas as pd
import copy
import time
import pygad
import tracemalloc

def generate_colors(num_colors):
    hues = [i / num_colors for i in range(num_colors)]
    colors = [colorsys.hls_to_rgb(hue, 0.5, 1.0) for hue in hues]  # Lightness = 0.5, Saturation = 1
    return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]

colors = generate_colors(50)

class tabu_LRP():
    def __init__(self, partition,
                  capacity = 100, C_k = 0.615, nb_neigh = 30, V_mean = 30, OperationalCost = 100,
                    NonCompianceCost = 100, veh_rent = 150, remove_depots = False):
        """
        partition (array): cluster considéré
        capacity (int): capacité de chaque véhicule
        C_k (float): coefficient kilométrique
        nb_neigh (int): nombre de voisins pour la génération dans tabu
        V_mean (float): vitesse moyenne de déplacement du vehicule (km/h)
        OperationalCost (float): coût opérationnel de livraison avant la date de chargement prévue (par jour d'avance)
        NonCompianceCost (float): coût résultant d'une livraison après la date max de livraison (par jour de retard)
        veh_rent (float): coût de location d'un vehicule 
        remove_depot (bool): booléen: definit la version du code (False: orienté OOTW, True: orienté enlèvement de depots)
        """
        self.nodes = partition # needs to be an array
        self.capacity = capacity
        self.C_k = C_k
        self.nb_neigh = nb_neigh
        self.veh_rent = veh_rent
        self.remove_depots = remove_depots
        self.distance_matrix = np.load('data/matrice_distances.npy')

        self.data = np.array(pd.read_csv("./data/Order.csv"))
        self.data[:,:2] = self.data[:,:2].astype(int) # Date Chargement + livraison
        self.weights = self.data[:,2] # Poids
        self.all_coordinates = self.data[:,3:]

        df = pd.read_csv('data/Dataset_Livraisons_clean.csv')
        sites = pd.read_excel('data/Sites éligibles.xlsx')

        self.coords = self.all_coordinates[self.nodes]
        self.nb_orders = len(df)
        self.nb_sites = len(sites)
        self.nb_nodes = self.nb_orders + self.nb_sites
        self.V_mean = V_mean 
        self.OperationalCost = OperationalCost
        self.NonCompianceCost = NonCompianceCost

        
        #sites
        data_sites = np.array(pd.read_csv("./data/datasetSites.csv"))

        # Coefficients C_i in Objective function
        
        v=np.array([-0.04, # * nbr chomeurs 
                    -0.08, # * nbr conducteurs
                    1000 ,#* 12 / 100 , # * prix mètre carré -> loyer mensuel
                    750, # * distance aux ports
                    1000, # * concurrents
                    675 # * taxe TCFE
                    ])
        """
        v=np.array([-0.01,-0.08,1000,750,365000,675*3])
        """
        self.coeff_sites=data_sites.dot(v) 

        self.extract()



    def extract(self):
        """This function extracts the indexes of orders and those of depots in the cluster
        """
        self.sites = [] # liste contenant les indices des depots
        i = len(self.nodes)-1
        while(self.nodes[i] >= self.nb_orders):
            self.sites.append(i)
            i -= 1
        self.orders = list(range(i+1))
        print(f"Les sites sont d'indices {self.sites}")

    def assign_client(self):
        """This function assigns clients to the closest eligible site for delivery
        Output: dictionnary with depots as keys and list of clients as values
        """
        coords_depots = self.coords[self.sites,:]
        partition = { site : [] for site in range(len(self.sites))}
        for i in range(len(self.orders)):
            x = self.coords[i]
            D = np.linalg.norm(coords_depots - x, axis =1)
            partition[np.argmin(D)].append(self.orders[i])
        return partition

    def initialize(self):
        """ This function initializes a feasible solution
        """
        feasible_sol = []

        # Partition clients based on their closest depot
        partition = self.assign_client()

        # Map to track the number of vehicles dispatched per day
        days_per_depot = {part : [] for part in partition.keys()}

        for part in partition.keys() : # Iterate over all partitions
            if partition[part]!=[] : # If the partition isn't empty
                partition[part].sort(key = lambda x : self.data[self.nodes[x],1]) # Sort order by ascending order
                orders = copy.deepcopy(partition[part])
                k = 0
                while( k < len(orders)): # Sequence the orders present in the partition while respecting weight constraint
                    sequence = []
                    while (self.compute_weight(sequence) >= 0) and (k < len(orders)):
                        sequence.append(self.nodes[orders[k]])
                        k+=1
                    if(self.compute_weight(sequence) < 0):
                        sequence.pop()
                        k -= 1
                    delivery = {
                        'day' : self.data[sequence[0],0].astype(int),
                        'depot' : self.nodes[self.sites[part]],
                        'vehicle' : len([i for i in range(len(days_per_depot[part])) 
                                        if (days_per_depot[part][i] == self.data[sequence[0],0]) ] ) + 1,
                        'sequence' : sequence
                    }
                    days_per_depot[part].append(self.data[sequence[0],0])
                    feasible_sol.append(delivery)

        return feasible_sol
                    
    def cost_delivery(self,delivery):
        """This function computes the cost of a delivery
        Input: deliver: { 'day':_ , 'depot':_, 'vehicle':_ , 'sequence':_}
        Output: cost 
        """
        depot = delivery['depot']
        sequence = delivery['sequence']
        distance = self.distance_matrix[depot,sequence[0]]
        time = delivery['day']*24 + distance / self.V_mean
        twCost = 0
        if( time/24 < self.data[sequence[0],0]):
            twCost += self.OperationalCost * (self.data[sequence[0],0]-time/24)
            #twCost += self.OperationalCost
        elif ( time/24 >= self.data[sequence[0],1]+1):
            twCost += self.NonCompianceCost * (time/24-self.data[sequence[0],1])
            #twCost += self.NonCompianceCost
        for i in range(len(sequence)-1):

            # Distance costs
            distance += self.distance_matrix[sequence[i],sequence[i+1]]

            # Time-Window related costs
            time = delivery['day']*24 + distance / self.V_mean # in hours

            if( time/24 < self.data[sequence[i+1],0]):
                twCost += self.OperationalCost * (self.data[sequence[i+1],0]-time/24)
                #twCost += self.OperationalCost
            elif ( time/24 >= self.data[sequence[i+1],1] + 1):
                twCost += self.NonCompianceCost * (time/24-self.data[sequence[i+1],1])
                #twCost += self.NonCompianceCost 

        distance += self.distance_matrix[sequence[len(sequence)-1], depot]
        return self.C_k * distance + twCost
    

    def obj_function(self,sol):

        f=0
        # Depot Placement cost
        depots = set([element['depot'] for element in sol])
        for elmnt in depots:
            f += self.coeff_sites[elmnt - self.nb_nodes]

        # Vehicle Routing cost
        for delivery in sol:
            f += self.cost_delivery(delivery)

        # Rent of vehicles
        f += self.veh_rent * len(sol)

        return f
    
    def alter_depot(self, feasible_sol):
        """This function generates a neighbor by removing a randomly selected depot
            It then replaces each delivery of this depot with the depot that is nearest to both the first and last client
            of this delivery.
        """
        sol = copy.deepcopy(feasible_sol)
        #orders = set(np.array([elmnt['sequence'] for elmnt in sol]).flatten())

        depots = set([element['depot'] for element in sol])
        
        depot = np.random.choice(list(depots),replace=False) # We randomly choose a depot to remove  
        
        # We start by removing the depot from all sequences ( in case it intervenes in vehicle restocking)
        deliveries = [elmnt for elmnt in range(len(sol)) if depot in sol[elmnt]['sequence']]
        l =[]
        for deliv in deliveries:
            seq = sol[deliv]['sequence']
            sol[deliv]['sequence'] = [ x for x in seq if x != depot]
            if not sol[deliv]['sequence'] :
                l.append(deliv)
        
        sol = [sol[i] for i in range(len(sol)) if i not in l]

        indexes = [i for i in range(len(sol)) if sol[i]['depot'] == depot] # Deliveries we need to modify to remove the depot

        if self.remove_depots: # Mode for depot minimisation
            if(len(depots)==1):
                return sol
            S = list(depots - {depot})
        else: # Mode for OOTW minimisation
            S = list(set(self.nodes[self.sites].astype(int)) - {depot})

        for i in indexes: # We iterate over all deliveries departing from the chosen depot for removal
            seq = sol[i]['sequence']
            
            first_client = seq[0]
            last_client = seq[-1]
            nearest_depot = np.argmin(self.distance_matrix[first_client , S]+
                                        self.distance_matrix[last_client , S]) # We choose the nearest depot to the
            # first and last clients in the sequence
            
            nearest_depot = S[nearest_depot]
            #print(f'{depot} -> {nearest_depot}')
            #print(sol[i])
            sol[i]['depot'] = nearest_depot
            #print(sol[i])
            l=[ elmnt for elmnt in range(len(sol)) if (sol[elmnt]['depot'] == nearest_depot)
                                                    and (sol[elmnt]['day'] == sol[i]['day'])]
            for index in range(len(l)):
                sol[l[index]]['vehicle'] = index
            
        return sol

    def pygad_optimize(self, delivery):
        new_delivery = delivery
        nodes = [new_delivery['depot']]
        nodes.extend(delivery['sequence'])
        distance_matrix = self.distance_matrix[nodes,nodes]
        
        # fitness_function = - self.cost_delivery
        def fitness_func (ga_instance , solution, solution_idx):
            deliv = copy.deepcopy(delivery)
            deliv['sequence'] = [nodes[int(x)] for x in list(solution) ]
            return - self.cost_delivery(deliv)
        # GA Parameters
        num_generations = 15  # Number of iterations
        num_parents_mating = 4
        sol_per_pop = 6  # Population size
        num_genes = len(delivery['sequence'])

        gene_space = list(range(1, len(distance_matrix)))  # Possible customer indices

        # Initialize PyGAD
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func= fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,
            mutation_type="random",
            crossover_type="single_point",
            parent_selection_type="sss",  # Stochastic Selection
            keep_parents=2,
            allow_duplicate_genes=False,  # Ensures unique customers in sequence
            parallel_processing=['thread', 4]
        )

        # Run GA
        ga_instance.run()

        # Get best solution
        solution, solution_fitness, _ = ga_instance.best_solution()
        new_delivery['sequence'] = [nodes[int(x)] for x in list(solution) ]

        # Plot results
        #ga_instance.plot_fitness()

        return new_delivery

    def optimize_route(self, feasible_sol):
        """This function optimizes the route in delivery"""
        sol = copy.deepcopy(feasible_sol)
        l = [idx for idx in range(len(sol)) if len(sol[idx]['sequence'])>=2]
        if not l:
            return sol
        # Choose a delivery randomly
        indx_delivery = np.random.choice(l, replace=True)

        # Optimize its sequencing using PyGAD
        new_delivery = self.pygad_optimize(sol[indx_delivery])
        sol[indx_delivery] = new_delivery

        return sol
        
    def move_order(self, feasible_sol):
        """This function moves all OOTW orders in a delivery to other deliveries where they get delivered on time 
        """
        sol = copy.deepcopy(feasible_sol)
        sol.sort(key = lambda d : self.outOfTW(d)[0], reverse = True)
        indx = 0 # Choose the delivery the highest number of OOTWs
        
        delivery1 = copy.deepcopy(sol[indx])
        numberOOT, OOTW_orders = self.outOfTW(delivery1) 
        # OOTW_orders: indexes of orders in delivery['sequence'] that are out of their time windows
        if not OOTW_orders:
            #print('No order OOTW in delivery')
            return sol

        L = []
        sol2 = []
        for indx_order in OOTW_orders: # Iterate over all OOTW order in the delivery
            inTW_deliveries = [ idx for idx in range(len(sol)) 
                            if (sol[idx]['day'] >= self.data[delivery1['sequence'][indx_order],0])
                            and (sol[idx]['day'] <= self.data[delivery1['sequence'][indx_order],1])
                            and (idx != indx)
                            and idx not in L ]
            
            order = delivery1['sequence'][indx_order]

            if inTW_deliveries :
                #print(f'Found a delivery for {indx_order}: {inTW_deliveries}')
                # Move the order to a delivery taking place during its TW and departing from the closest depot
                inTW_deliveries.sort(key= lambda x : self.distance_matrix[order,sol[x]['depot']])
                indx2 = inTW_deliveries[0]
                seq = sol[indx2]['sequence']
                indx_order2 = np.argmin(self.distance_matrix[order,seq])
                new_seq = seq[:indx_order2+1]
                new_seq.append(order)
                new_seq.extend(seq[indx_order2+1:])
                delivery2 = sol[indx2]
                if self.compute_weight(new_seq) < 0: # Weight constraint not verified after adding the order
                    #print('Weight not verified after action')
                    
                    i = 0
                    vehs = [k for k in range(len(sol)) if sol[k]['depot'] == delivery2['depot']
                                                    and sol[k]['day'] == delivery2['day']
                                                    and k != indx2]
                    while i < len(new_seq): # Iterate to segment the sequence into subsequences
                        orders = []
                        while (self.compute_weight(orders) >=0) and (i< len(new_seq)):                 
                            orders.append(new_seq[i])
                            i +=1
                        if self.compute_weight(orders) < 0:
                            orders.pop()
                            i -= 1
                        sol2.append(
                            {
                                'day': int(delivery2['day']),
                                'depot': delivery2['depot'],
                                'vehicle':len(vehs) + 1,
                                'sequence': orders
                            }
                        )
                        vehs.append(i)
                else: # Weight constraint is verified
                    #print('Weight verified')
                    sol2.append({
                                'day': int(delivery2['day']),
                                'depot': delivery2['depot'],
                                'vehicle': delivery2['vehicle'],
                                'sequence': new_seq
                            } )
                L.append(indx2)
                #print(self.isValid(sol))
            else: # No delivery takes place in the corresponding TW

                depot_idx = np.argmin(self.distance_matrix[order,self.sites])
                depot = self.nodes[self.sites[depot_idx]]
                deliv ={
                    'day': int(self.data[order,0]),
                    'depot': depot,
                    'vehicle': 1,
                    'sequence': [order]
                }
                sol2.append(deliv)

        L.append(indx)
        
        sol2.extend([sol[i] for i in range(len(sol)) if i not in L])
        
        delivery1['sequence'] = [delivery1['sequence'][i] for i in range(len(delivery1['sequence'])) if i not in OOTW_orders]
        if delivery1['sequence']:
            sol2.append(delivery1)
        #sol2 = [ d for d in sol2 if d['sequence']]
        return sol2
    

    def concatenate_deliveries(self, feasible_sol):
        """This function chooses a day and concatenates two random deliveries taking place on this day """
        sol = copy.deepcopy(feasible_sol)
        days= set([elmnt['day'] for elmnt in sol ])
        day = np.random.choice(list(days),replace=False)
        deliveries = [i for i in range(len(sol)) if sol[i]['day'] == day]
        if(len(deliveries) == 1):
            return sol
        indx1, indx2 = np.random.choice(deliveries, 2,replace=False)
        #compute_weight(self, seq)
        if sol[indx2]['depot'] != sol[indx1]['depot']: # the selected deliveries depart from different depots
            sol[indx1]['sequence'].append(sol[indx2]['depot'])
            sol[indx1]['sequence'].extend(sol[indx2]['sequence'])
            
        else: # We optimize the index at which we place the depot for restocking
            seq = sol[indx1]['sequence'] + sol[indx2]['sequence']
            new_seq =[]
            if self.compute_weight(seq) < 0:
                i = 0
                while i < len(seq):
                    orders = []
                    while (self.compute_weight(orders) >=0) and (i< len(seq)):                 
                        orders.append(seq[i])
                        i +=1
                    if self.compute_weight(orders) < 0:
                        orders.pop()
                        i -= 1

                    new_seq.extend(orders)
                    if i <len(seq):
                        new_seq.append(sol[indx1]['depot'])
                sol[indx1]['sequence'] = new_seq
            else:
                sol[indx1]['sequence'] = seq
                
        sol.pop(indx2)

        return sol
    
    def move_date(self, feasible_sol):
        """This function changes the date of a delivery"""
        sol = copy.deepcopy(feasible_sol)
        indx = np.random.choice(range(len(sol)),replace=False)
        delivery = sol[indx]
        
        min_DD = self.data[min(delivery['sequence'], key= lambda x: self.data[x,1]),1] #minimal delivery date
        min_LD = self.data[min(delivery['sequence'], key= lambda x: self.data[x,0]),0] #minimal loading date
        day = np.random.randint(min_LD , min_DD +1)
        sol[indx]['day'] = int(day)
        
        return sol

    def generate_neighborhood(self, sol):
        """Function to generate the neighborhood of a feasible solution"""
        nb_neigh = self.nb_neigh
        neighborhood =[]
        nb_functions = 5 # number of neighbor generation function ( add_depot, alter_depot ...)
        for i in range(nb_neigh):
            operator_weights = [0.25, 0.3, 0.3, 0.1, 0.05]  # Weights for each operator
            r = np.random.choice(range(nb_functions))#, p=operator_weights)
            if r == 0 :
                neigh = self.alter_depot(sol)
                #print('alter_depot') 
            elif r == 1:
                neigh = self.move_order(sol)
                for k in range(50):
                    neigh = self.move_order(neigh)
                #print('move_order')
            elif r == 2:
                neigh = self.concatenate_deliveries(sol)
                #print('concatenate_deliveries')
            elif r == 3:
                neigh = self.move_date(sol)
                #print('move_date')
            else:
                neigh = self.optimize_route(sol)
                #print('optimize_route')

            #print(self.isValid(neigh))
            neighborhood.append(neigh)
        return neighborhood
    
    def eps_greedy_policy(self, eps , Q, action, choices):
        """This policy function chooses an action( one function among the neighbor generation functions)"""
        if( np.all(Q == 0) or action == None):
            #new_action = np.random.choice(range(len(Q)), p = np.exp(-choices) / np.sum(np.exp(-choices)))
            new_action = np.random.choice(range(len(Q)))#, p = np.exp(-choices) / np.sum(np.exp(-choices)),replace=False)
        else:
            r = np.random.random()
            if(r < eps):
                new_action = np.random.choice(range(len(Q)))#, p = np.exp(-choices) / np.sum(np.exp(-choices)),replace=False)
            else:
                new_action = np.argmax(Q[action,:])
        return new_action
    
    def take_action(self, action, sol):
        if action == 0 :
            neigh = self.alter_depot(sol)
            #print('alter_depot')
        elif action == 1:
            neigh = self.move_order(sol)
            for k in range(20):
                neigh = self.move_order(neigh)
            #print('move_order')
            
        elif action == 2:
            neigh = self.concatenate_deliveries(sol)
            #print('concatenate_deliveries')
            
        elif action == 3:
            neigh = self.optimize_route(sol)
            #print('optimize_route')
        else:
            neigh = self.move_date(sol)

        reward = 1 / self.obj_function(neigh)

        return neigh, reward
    
    def generate_neighborhood_Q(self, sol, Q, eps = 0.3, gamma = 0.4, alpha = 1, lambd = 0.99):
        """ The Q-Learning neighborhood generation function
        """
        nb_neigh = self.nb_neigh        
        neighborhood =[]
        choices = np.zeros((5,))
        action = self.eps_greedy_policy(eps, Q, None, choices)
        choices[action] += 1
        for i in range(nb_neigh):
            # eps = eps * lambd
            # Take action
            neigh, reward = self.take_action(action, sol)
            new_action = self.eps_greedy_policy(eps , Q, action, choices)

            # Update the Q-matrix
            Q[action, new_action] += alpha * (reward + gamma *np.max(Q[new_action]) - Q[action, new_action]) #+ 10*np.sqrt(1/(choices[action]+1e-2))
            action = new_action

            # Increment the number of times the action has been chosen
            choices[action] += 1
            neighborhood.append(neigh)
            
        return neighborhood, Q

    def compute_weight(self, seq):
        """ This function computes the available weight in a vehicle"""
        weight = self.capacity

        for order in seq:
            if order < self.nb_orders: # If the vehicle passes by a client
                weight -= self.data[order,2] # Available weight decreases by the client's order's weight
            else:# If the vehicle passes by a depot
                weight = self.capacity # Available weight becomes maximum( vehicle restocked in depot)

        return weight
        
    def isValid(self, sol):
        # Weight Constraint
        validity = True
        for elmnt in sol :
            weight = self.compute_weight(elmnt['sequence'])
            if(weight < 0):
                print("Weight not valid")
                return False

        # Recurrent of missing orders
        orders = []
        for elmnt in sol:
            l = [ x for x in elmnt['sequence'] if x < self.nb_orders ]
            orders.extend(l)
        
        if(len(orders) < len(self.orders)):
            print('Missing Orders')
            print(len(self.orders) - len(orders))
            return False
        if(len(orders) > len(self.orders)):
            print('Recurrent Orders')
            print(len(set(orders)) - len(self.orders) )
            return False
        if(len(set(orders)) < len(orders)):
            print('Recurrent Orders 2')
            return False
            
        return validity 
    def compute_duration(self, delivery):
        """This function tests the validity of a delivery when it comes to the legal duration of driving a vehicle
        in a day.
        Input: a delivery
        Output: the total duration of the delivery"""
        depot = delivery['depot']
        sequence = delivery['sequence']
        distance = self.distance_matrix[depot,sequence[0]]
        for i in range(len(sequence)-1):
            # Distance
            distance += self.distance_matrix[sequence[i],sequence[i+1]]
        distance += self.distance_matrix[sequence[len(sequence)-1], depot]  
        return distance / self.V_mean 

    def outOfTW(self, delivery):
        """ This function computes the number of instances where a client was delivered his/her order out of the expected
        Time-Window
        Input: deliveru: a certain delivery
        Output: the number of OOTWs orders in the delivery and 
                OOTW_orders: list of the positions of these OOTWs orders in the sequence of the delivery"""
        depot = delivery['depot']
        sequence = delivery['sequence']
        distance = self.distance_matrix[depot,sequence[0]]
        time = delivery['day']*24 + distance / self.V_mean
        OOTW_orders = []
        if ( time/24 < self.data[sequence[0],0]) or ( time/24 >= self.data[sequence[0],1]+1):
                OOTW_orders.append(0)
        for i in range(len(sequence)-1):

            # Distance costs
            distance += self.distance_matrix[sequence[i],sequence[i+1]]

            # Time-Window related costs
            time = delivery['day']*24 + distance / self.V_mean # in hours

            if ( time/24 < self.data[sequence[i+1],0]) or ( time/24 >= self.data[sequence[i+1],1]+1):
                OOTW_orders.append(i+1)

        return len(OOTW_orders), OOTW_orders
    
    def totalOOTW(self, sol):
        """This function returns the total number of instances when a client gets delivered oit of the deignated TW
        for all the solution"""
        total_number = 0
        for deliv in sol:
            numberOOT, OOTW_orders = self.outOfTW(deliv)
            total_number += numberOOT
        return total_number

    def meanNbVeh(self, sol):
        """This function returns the mean number of vehicles sent for delivery daily"""
        days = set([elmnt['day'] for elmnt in sol])
        vehs = []
        for day in days:
            deliveries = [deliv for deliv in sol if deliv['day']==day]
            vehs.append(len(deliveries))
        return np.sum(vehs)/369

    def solve(self, max_iter=100, sol = None, Q = np.zeros((5,5)), QL = False,  eps = 0.4, gamma = 0.7, alpha = 1):
        """
        Implements the Tabu Search algorithm to solve the problem.
        
        Args:
            max_iter (int): Maximum number of iterations to perform.
            sol (list): initial list to optimise, if None the algorithm initializes its own solution.
            Q (array): Q-matrix for Q learning
            QL (boolean): indicates if the algorithm should use Q-Learning
            eps (float): exloration-exploitation rate for QL
            gamma (float): decay rate for QL
            alpha (float): learning rate for QL

        Returns:
            best_sol: (json) the final optimized solution
            Costs: (list) Objective funtion for each iteration
            Q: Final Q matrix ( If QL = False, Q is all zeros)
            OTW: List of the number of orders out of time window for each iteration
            meanNbVeh: List of the mean number of vehicles sent for delivery per day for each iteration
        """
        
        # Initialize solution
        if sol == None :
            current_sol = self.initialize()
        else:
            current_sol = sol

        best_sol = current_sol
        best_cost = self.obj_function(best_sol)
        current_cost = best_cost
        initial_value = self.obj_function(best_sol)
        Costs = [current_cost]
        # Tabu list and its size
        tabu_list = []
        tabu_size = 5  
        OTW = [self.totalOOTW(best_sol)] # Number of instances of orders delivered out of corresponding time-window
        meanNbVeh = [self.meanNbVeh(best_sol)]
        current = 0
        window = 10
        for iteration in range(max_iter):
            print(f"Iteration {iteration + 1}/{max_iter}, "
                  f'Validity of current solution: {self.isValid(current_sol)}')
            depots = [element['depot'] - self.nb_orders for element in current_sol]
            print(f'Chosen depots: {set(depots)}')
            print(f'Number of OOTW: {OTW[-1]}')
            #if OTW[-1] <= len(self.orders) / 100:
            #    print(self.nb_orders)
            #    break
            # Generate the neighborhood of the current solution
            if not QL :
                neighborhood = self.generate_neighborhood(current_sol)
            else:
                neighborhood, Q = self.generate_neighborhood_Q(current_sol, Q,eps , gamma , alpha)
            # obj_function of all neighbors
            neighbor_costs = [(self.obj_function(neigh), neigh) for neigh in neighborhood]
            # Filter out neighbors in the tabu list
            valid_neighbors = [
                (cost, neigh) for cost, neigh in neighbor_costs if neigh not in tabu_list
                                                                and self.isValid(neigh)
            ]

            if not valid_neighbors:
                print("No valid neighbors found. Exiting...")
                Q = np.zeros((5,5))
                continue

            # Find the best valid neighbor
            valid_neighbors.sort(key=lambda x: x[0])  # Sort by cost
            best_neighbor_cost, best_neighbor = valid_neighbors[0]

            # Update tabu list
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

            # Move to the best neighbor
            current_sol = best_neighbor
            current_cost = best_neighbor_cost

            # Update the global best solution
            if current_cost < best_cost:
                best_sol = current_sol
                best_cost = current_cost
                current = 0
            else:
                current +=1

            # Logging for debugging
            print(
                f"Current Cost: {current_cost}, Best Cost: {best_cost}"
            )
            print(" ")
            Costs.append(best_cost)
            OTW.append(self.totalOOTW(best_sol))
            meanNbVeh.append(self.meanNbVeh(best_sol))
            if current > window:
                current_sol = best_sol

        print("Optimization completed.")
        print(f"From {initial_value} to {best_cost}")
        return best_sol, Costs, Q, OTW, meanNbVeh
    
    def plot_delivery(self, delivery):
        """This method plots the sequence of orders in a certain delivery
        Input: delivery: a map variable describing a delivery
        """
        seq = [delivery['depot']]
        seq.extend(delivery['sequence'])
        seq.append(delivery['depot'])

        
        #gdf=gpd.read_file('Data/fr.shp')
        #gdf.plot(ax=ax)
        # Depot
        plt.scatter(self.all_coordinates[seq[0],1],self.all_coordinates[seq[0],0], c= colors[seq[0] - self.nb_orders], label=f"Depot {seq[0]- self.nb_orders}") 
        # Orders
        plt.scatter(self.all_coordinates[seq[1:len(seq)-1],1],self.all_coordinates[seq[1:len(seq)-1],0], c='blue')
        # Routes
        for i in range(len(seq) - 1):
            start = self.all_coordinates[seq[i],:]
            end = self.all_coordinates[seq[i+1],:]
            
            # Draw an arrow between consecutive locations
            plt.arrow(
                start[1], start[0],        # Start point
                end[1] - start[1],         # Change in x
                end[0] - start[0],         # Change in y
                head_width=0.02,            # Arrowhead width
                length_includes_head=True, # Ensure the arrowhead is part of the line
                color='green'              # Arrow color
            )
    
    def plot_day(self, sol, d):
        """This method plots all order sequences taking place in a day
        Input: d: day to consider for the plots"""
        deliveries = [elmnt for elmnt in sol if elmnt['day'] == d]
        print(deliveries)
        if(len(deliveries)==0):
            print('No delivery on this day')
        else:
            fig,ax = plt.subplots(figsize = (7,7))
            for delivery in deliveries:
                self.plot_delivery(delivery)
            # Add labels, legend, and title
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f"Routing of delivery vehicles on day {d}")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot(self, sol):
        """This method plots all the deliveries proposed by a solution"""
        days = {elmnt['day'] for elmnt in sol}
        for day in days:
            self.plot_day(sol, day)
    
    def generate_geojson(self, deliveries):
        NbOrders = 6074
        # GeoJSON structure
        clients_depots = {"type": "FeatureCollection", "features": []}
        routes = {"type": "FeatureCollection", "features": []}

        depots = list(set([d['depot'] for d in deliveries]))
        print(depots)

        # Add depot points
        for dep in depots:
            clients_depots["features"].append({
                    "type": "Feature",
                    "properties": {"type": "depot", "day": 369},
                    "geometry": {"type": "Point", "coordinates": [self.all_coordinates[dep,1], self.all_coordinates[dep,0]]}
                })
        orders =[]
        for delivery in deliveries:
            day = delivery["day"]
            depot = delivery["depot"]
            clients = delivery["sequence"]
            
            # Add client points
            for client in clients:
                if(client < NbOrders):
                    orders.append(client)
                    clients_depots["features"].append({
                        "type": "Feature",
                        "properties": {"type": "client", "day": day},
                        "geometry": {"type": "Point", "coordinates": [self.all_coordinates[client,1], self.all_coordinates[client,0]]}
                    })
            
            # Create the route as a LineString
            route_coordinates = [[self.all_coordinates[depot,1], self.all_coordinates[depot,0]]] + [[self.all_coordinates[c,1], self.all_coordinates[c,0]] for c in clients] + [[self.all_coordinates[depot,1], self.all_coordinates[depot,0]]] 
            routes["features"].append({
                "type": "Feature",
                "properties": {"day": day, "depot":depot},
                "geometry": {"type": "LineString", "coordinates": route_coordinates}
            })
        
        # Save files
        with open("data/clients_depots.geojson", "w") as f:
            json.dump(clients_depots, f, indent=4)
        
        with open("data/routes.geojson", "w") as f:
            json.dump(routes, f, indent=4)

        print("GeoJSON files saved: clients_depots.geojson & routes.geojson")

def multiStep_solve(nb_iter = 2000, max_iter = 300):
    """This function performs the multi-step Tabu Search Algorithm
    Input: number of iterations during the final step nb_iter,
            number of iterations during preliminary steps max_iter
    Output: optimised solution best_sol
    """
    np.random.seed(10)

    # Step 1
    tabu = tabu_LRP(np.array([i for i in range(6115)]), nb_neigh = 10, OperationalCost=1e6, NonCompianceCost=1e6)
    best_sol, Costs, Q, OTW, meanNbVeh = tabu.solve(max_iter = max_iter  , QL = False)
    depots = list(set([d['depot'] for d in best_sol])) # Extrcat depots chosen during the first step
    fig = plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.plot(Costs)
    plt.xlabel('Iteration number ')
    plt.ylabel('Obj Func value')
    plt.title('Convergence curve of OF step 1')

    plt.subplot(1,3,2)
    plt.plot(OTW)
    plt.xlabel('Iteration number')
    plt.ylabel('Number of instances')
    plt.title('Out Of Time-Window instances step 1')

    plt.subplot(1,3,3)
    plt.plot(meanNbVeh)
    plt.xlabel('Iteration number')
    plt.ylabel('Number of vehicles')
    plt.title('Mean number of vehicles per day step 1')
    fig.tight_layout()

    # Extract Out Of Time Window Orders in solution found after step 1
    OOT_orders = []
    for d in best_sol:
        OOT_orders.extend([d['sequence'][i] for i in tabu.outOfTW(d)[1]])


    global_OTW = OTW # List to display the global convergence of the number of OOTW orders 
    #after the multi-step algorithm
    final_sol = [] # The final solution resulting from OOTW elimination steps (prliminary steps, just before the last step)
    OF = [tabu.obj_function(best_sol)] # List to track the objective function during the preliminary steps
    k = 1
    # Loop to perform the OOTW elimination steps
    while len(OOT_orders) >= 1 :
        k += 1
        # Extract all orders that are in time window
        best_sol1 = [d for d in best_sol]
        l = []
        for i in range(len(best_sol1)):
            seq = best_sol1[i]['sequence'][:]
            best_sol1[i]['sequence'] = [x for x in seq if x not in OOT_orders]
            if not best_sol1[i]['sequence'] :
                l.append(i)
        best_sol1 = [best_sol1[i] for i in range(len(best_sol1)) if i not in l]

        # Feed the OOTWs detected to the tabu algorithme with parameters favoring OOTW minimisation
        tabu_i = tabu_LRP(np.array(OOT_orders + depots),  nb_neigh = 15, OperationalCost=1e6, NonCompianceCost=1e6)
        best_sol2, Costs, Q, OTW, meanNbVeh = tabu_i.solve(max_iter, QL = False)


        global_OTW.extend(OTW)
        final_sol.extend(best_sol1)
        
        OF.append(tabu.obj_function(final_sol + best_sol2))

        # Extract OOTWs
        best_sol = [d for d in best_sol2]
        OOT_orders = []
        for d in best_sol:
            OOT_orders.extend([d['sequence'][i] for i in tabu.outOfTW(d)[1]])
        
        assert(tabu.isValid(final_sol + best_sol2))

        if len(OOT_orders) == 0:
            final_sol = final_sol + best_sol2

        fig = plt.figure(figsize=(12,5))
        plt.subplot(1,3,1)
        plt.plot(Costs)
        plt.xlabel('Iteration number ')
        plt.ylabel('Obj Func value')
        plt.title(f'Convergence curve of OF step {k}')

        plt.subplot(1,3,2)
        plt.plot(OTW)
        plt.xlabel('Iteration number')
        plt.ylabel('Number of instances')
        plt.title(f'Out Of Time-Window instances step {k}')

        plt.subplot(1,3,3)
        plt.plot(meanNbVeh)
        plt.xlabel('Iteration number')
        plt.ylabel('Number of vehicles')
        plt.title(f'Mean number of vehicles per day step {k}')
        fig.tight_layout()


    with open("feasible_sol_france.json", "w") as file:
        json.dump(final_sol, file,  default=convert_to_serializable, indent=4)
    
    fig = plt.figure(figsize=(12,5))
    X = [i for i in range(k)]
    plt.subplot(1,2,1)
    plt.bar(X,OF)
    plt.title("Objective Function for each step")
    plt.xlabel('Step number')
    plt.ylabel('Value of Objective Function')

    plt.subplot(1,2,2)
    plt.plot(global_OTW)
    plt.title("Convergence of number of OTW instances")
    plt.xlabel('Step number')
    plt.ylabel('Number of OTW instances')
    fig.tight_layout()

    # Final Step: Depot and vehicle elimination
    tabu_final = tabu_LRP(np.array([i for i in range(6115)]), nb_neigh = 10, OperationalCost=1e4, NonCompianceCost=1e4, veh_rent=1500, remove_depots=True)
    best_sol, Costs, Q, OTW, meanNbVeh = tabu_final.solve(max_iter = nb_iter, sol = final_sol , QL = False)
    with open("best_sol_france.json", "w") as file:
        json.dump(best_sol, file,  default=convert_to_serializable, indent=4)

    fig = plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.plot(Costs)
    plt.xlabel('Iteration number ')
    plt.ylabel('Obj Func value')
    plt.title(f'Convergence curve of OF step {k+1}')

    plt.subplot(1,3,2)
    plt.plot(OTW)
    plt.xlabel('Iteration number')
    plt.ylabel('Number of instances')
    plt.title(f'Out Of Time-Window instances step {k+1}')

    plt.subplot(1,3,3)
    plt.plot(meanNbVeh)
    plt.xlabel('Iteration number')
    plt.ylabel('Number of vehicles')
    plt.title(f'Mean number of vehicles per day step {k+1}')
    fig.tight_layout()
    plt.show()

    return best_sol



def scalability(QL):
    """This function performs the scalability analysis of the tabu algorithm
        Input: QL: boolean, if True the function tests with Tabu + QL, else it tests with Tabu pure
        Output: curves showing the evolution of memory allocation and computation time as functions of the instance size
    """
    nb_orders = 6074
    nb_nodes = 6115
    time_results = {}
    memory_results = {}
    problem_sizes = [100, 1000, 3000, 6000]

    for size in problem_sizes:
        print(size)
        orders = list(range(size))
        depots = list(np.random.choice(range(nb_orders+1,nb_nodes),4,replace=False))
        orders.extend(depots)
        tabu = tabu_LRP(np.array([i for i in range(size)] + [i for  i in range(6075,6115)]), nb_neigh = 10)
        t0 = time.time()
        #mem_usage = memory_usage(tabu.solve , interval=0.1)
        tracemalloc.start()
        tabu.solve(QL=QL)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Peak memory usage: {peak / 10**6} MB")
        tracemalloc.stop()
        time_results[size] = time.time() - t0
        #memory_results[size] = max(mem_usage)
        memory_results[size] = peak / 10**6
        print(f"Problem size {size}: Peak Memory Usage = {peak / 10**6} MB")

    # Compute Memory Performance Index (MPI)
    mpi_values = {}
    for size in problem_sizes:
        mpi = memory_results[size] / memory_results[problem_sizes[0]]
        mpi_values[size] = mpi

    print("\nMemory Performance Index (MPI):")
    for size, mpi in mpi_values.items():
        print(f"Problem size {size}: MPI = {mpi:.2f}")

    fig = plt.figure(figsize=(15,5))

    # Plot memory usage
    plt.subplot(1,3,1)
    sizes = list(memory_results.keys())
    usage = list(memory_results.values())

    plt.plot(sizes, usage, marker='o', linestyle='-', label='Memory Usage')
    plt.grid()
    plt.legend()
    plt.xlabel("Problem Size")
    plt.ylabel("Peak Memory Usage (MB)")
    plt.title("Memory Usage Scalability of Tabu Search")

    # Plot MPI KPI
    plt.subplot(1,3,2)
    mpi = list(mpi_values.values())
    plt.plot(sizes, mpi, marker='o', linestyle='-', label='Memory Performance Index')
    plt.grid()
    plt.legend()
    plt.xlabel("Problem Size")
    plt.ylabel("MPI value", labelpad=15)
    plt.title("Memory Performance Index")

    # Plot time KPI
    plt.subplot(1,3,3)
    t = list(time_results.values())
    plt.plot(sizes, t, marker='o', linestyle='-', label='Time')
    plt.xlabel("Problem Size")
    plt.ylabel("Time value", labelpad=15)
    plt.title("Convergence Time Index")

    plt.legend()
    plt.grid()

    fig.tight_layout()
    plt.show()

def compare_algorithms(nb_iter = 50):
    """This function compares the Tabu Search and the Tabu + QL algorithms. It analyses their mean Objective values 
        and variance.
        Input: nb_iter: number of iteration for each algorithm
    """
    tabu = tabu_LRP(np.array([i for i in range(100)] + [i for  i in range(6075,6085)]), nb_neigh = 15, OperationalCost=1e4, NonCompianceCost=1e4)
    obs = np.zeros((nb_iter,))
    f0 = time.time()
    for i in range(nb_iter):
        np.random.seed(i)
        best_sol,Costs, Q, OTW, meanNbVeh = tabu.solve(max_iter = 200, QL = False)
        obs[i] = tabu.obj_function(best_sol)
        if i==0:
            plt.plot(Costs, label ="QL = False", color = 'red')
        else:
            plt.plot(Costs, color = 'red')
    f1 = time.time()
    mean_f = np.mean(obs)
    std_f = np.std(obs)

    t0 = time.time()
    for i in range(nb_iter):
        np.random.seed(i)
        best_sol,Costs, Q, OTW, meanNbVeh = tabu.solve( max_iter = 200,  QL = True, eps = 0.4, gamma = 0.9, alpha = 4)
        obs[i] = tabu.obj_function(best_sol)
        if i == 0:
            plt.plot(Costs, label ="QL = True ", color = 'green')
        else:
            plt.plot(Costs, color = 'green')
    t1 = time.time()
    mean_t = np.mean(obs)
    std_t = np.std(obs)
    print(f'Time : Tabu pure: {f1-f0} , Tabu + QL: {t1-t0}')
    print(f'means: Tabu pure: {mean_f}, Tabu + QL: {mean_t}')
    print(f'std: Tabu pure: {std_f}, Tabu + QLt: {std_t}')
    plt.legend()
    plt.show()

def evaluate_OOTW(sol):
    """This funtion evaluates the OOTW orders present in a solution wheter they are delivered far from their TW or
    not. 
    Input: sol: a certain feasible solution
    """
    distance_matrix = np.load('data/matrice_distances.npy')
    data = np.array(pd.read_csv("./data/Order.csv"))
    V_mean = 30
    OOTW_orders = {} # Map of the different OOTW orders as keys and the gap from their correponding TW as values
    for delivery in sol: # iterate over all deliveries
        depot = delivery['depot']
        sequence = delivery['sequence']
        distance = distance_matrix[depot,sequence[0]]
        time = delivery['day']*24 + distance / V_mean
        
        if ( time/24 < data[sequence[0],0]) :
            OOTW_orders[sequence[0]] = time/24 -  data[sequence[0],0]
        elif ( time/24 >= data[sequence[0],1]+1):
            OOTW_orders[sequence[0]] = time/24 - data[sequence[0],1]
        for i in range(len(sequence)-1):

            # Distance costs
            distance += distance_matrix[sequence[i],sequence[i+1]]

            # Time-Window related costs
            time = delivery['day']*24 + distance / V_mean # in hours

            if ( time/24 < data[sequence[i+1],0]) :
                OOTW_orders[sequence[i+1]] = time/24 -  data[sequence[i+1],0]
            elif ( time/24 >= data[sequence[i+1],1]+1):
                OOTW_orders[sequence[i+1]] = time/24 - data[sequence[i+1],1]

    #plt.bar(list(OOTW_orders.keys()),list(OOTW_orders.values()))
    plt.bar(range(len(list(OOTW_orders.values()))),list(OOTW_orders.values()))
    plt.title('Non compliance with Time Window for OOTW orders')
    plt.ylabel('Duration in days')
    plt.xlabel('OOTW orders')
    plt.grid()
    plt.show()


# Convert all numpy types to native Python types
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):  # For numpy integers
        return int(obj)
    elif isinstance(obj, np.floating):  # For numpy floats
        return float(obj)
    elif isinstance(obj, np.ndarray):  # For numpy arrays
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

if __name__ =="__main__":
    
    # TW and number of vehicles sent visualization
    
    # np.random.seed(10) # For reproducibility

    # Solve using Tabu
    """tabu = tabu_LRP(np.array([i for i in range(6115)]), nb_neigh = 10, OperationalCost=1e4, NonCompianceCost=1e4, remove_depots=True)
    best_sol, Costs, Q, OTW, meanNbVeh = tabu.solve(max_iter = 200, QL = False)"""

    # Use Multi-step solve for better results
    """best_sol = multiStep_solve()"""

    # Generate geojson files to use for visualization in QGIS
    """tabu.generate_geojson(best_sol)"""

    # Visualize deliveries 
    """tabu.plot(best_sol)"""
    
    # Evaluate OOTWs
    """evaluate_OOTW(best_sol)"""

    # Scalability test
    scalability(QL=False)

    # Compare Tabu Pure and Tabu+QL
    """compare_algorithms()"""
