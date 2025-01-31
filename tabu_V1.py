import cluster 
import numpy as np
from milp import milp_model
from milp_data import milp_data
import colorsys
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import json
import pandas as pd
import copy

def generate_colors(num_colors):
    hues = [i / num_colors for i in range(num_colors)]
    colors = [colorsys.hls_to_rgb(hue, 0.5, 1.0) for hue in hues]  # Lightness = 0.5, Saturation = 1
    return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]
colors = generate_colors(50)
class tabu_LRP():
    def __init__(self, cluster, nb_orders, nb_sites, nb_veh = 2 ,
                  capacity = 100, C_k = 0.615, nb_neigh = 20, V_mean = 30, OperationalCost = 2,
                    NonCompianceCost = 2, veh_rent = 10, mode = 0):
        """
        cluster: cluster considéré
        nb_orders : nombre total(sur toute la France) de commandes
        nb_sites: nombre total(sur toute la France) de sites éligibles
        nb_veh : nombre de vehicules de livraison pour chaque site
        capacity : capacité de chaque véhicule
        C_k : coefficient kilométrique
        nb_neigh : nombre de voisins pour la génération dans tabu
        V_mean: vitesse moyenne de déplacement du vehicule (km/h)
        OperationalCost : coût opérationnel de livraison avant la date de chargement prévue (par jour d'avance)
        NonCompianceCost : coût résultant d'une livraison après la date max de livraison (par jour de retard)
        veh_rent : coût de location d'un vehicule au cas de nécessité
        mode : mode 0: les depots on un nombre illimité de vehicules, mode 1: on prend en considération la 
                    location de vehicules
        """
        self.nodes = cluster[:,0]
        self.coords = cluster[:,1:3]
        self.weights = cluster[:,3]
        self.nb_orders = nb_orders
        self.nb_sites = nb_sites
        self.nb_veh = nb_veh
        self.nb_nodes = nb_orders + nb_sites
        self.capacity = capacity
        self.C_k = C_k
        self.nb_neigh = nb_neigh
        self.veh_rent = veh_rent
        self.mode = mode
        self.distance_matrix = np.load('data/matrice_distances.npy')

        df = pd.read_csv('data/Dataset_Livraisons_clean.csv')
        sites = pd.read_excel('data/Sites éligibles.xlsx')
        coords_sites = pd.DataFrame({'Latitude':list(sites.Coordonnées.str.split(',').str[0]+'.'+sites.Coordonnées.str.split(',').str[1]),
                  'Longitude':list(sites.Coordonnées.str.split(',').str[2]+'.'+sites.Coordonnées.str.split(',').str[3])
                  }).astype({'Latitude':'float64','Longitude':'float64'})
        data = pd.concat([df[['Latitude','Longitude']],coords_sites] , ignore_index=True)
        self.all_coordinates = np.array(data)

        self.V_mean = V_mean 
        self.OperationalCost = OperationalCost
        self.NonCompianceCost = NonCompianceCost

        self.data = np.array(pd.read_csv("./data/Order.csv"))

        #sites
        data_sites = np.array(pd.read_csv("./data/datasetSites.csv"))

        # Coefficients C_i in Objective function
        
        v=np.array([-0.01, # * nbr chomeurs 
                    -0.08, # * nbr conducteurs
                    1000 * 12 / 100 , # * prix mètre carré -> loyer mensuel
                    750, # * distance aux ports
                    1000, # * concurrents
                    675 # * taxe TCFE
                    ])
        """
        v=np.array([-0.01,-0.08,1000,750,365000,675*3])
        """
        self.coeff_sites=data_sites.dot(v) 



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
            partition[np.argmin(D)].append(i)
        return partition
    
    
    def initialize(self):
        feasible_sol = []
        partition = tabu.assign_client()
        for part in partition.keys() :
            if partition[part]!=[] :
                for i in range(0,len(partition[part]),10):

                    if(i+15 < len(partition[part])):
                        l = tabu.nodes[partition[part]].astype(int)[i:i+10]
                    else:
                        l = tabu.nodes[partition[part]].astype(int)[i:]
                        
                    lrp = milp_model([tabu.nodes[tabu.sites[part]].astype(int)], l, 2, capacity=100, solver="SCIP")
                    lrp.solve()
                    feasible_sol.extend(lrp.reconstruct())
        return feasible_sol
    
    def cost_delivery(self,delivery):
        """This function computes the cost of a delivery
        Input: deliver: { 'day':_ , 'depot':_, 'vehicle':_ , 'sequence':_}
        Output: cost 
        """
        depot = delivery['depot']
        sequence = delivery['sequence']
        distance = self.distance_matrix[depot,sequence[0]]
        twCost = 0 
        for i in range(len(sequence)-1):
            # Distance costs
            distance += self.distance_matrix[sequence[i],sequence[i+1]]

            # Time-Window related costs
            time = delivery['day']*24 + distance / self.V_mean # in hours

            if( time/24 < self.data[sequence[i],0]):
                twCost += self.OperationalCost * (self.data[sequence[i],0]-time/24)
            elif ( time/24 > self.data[sequence[i],1]):
                twCost += self.NonCompianceCost * (time/24-self.data[sequence[i],1])
            
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
        if (self.mode == 1):
            days = set([elmnt['day'] for elmnt in sol])
            
            for day in days:
                for depot in depots:
                    vehicles = {elmnt['vehicle'] for elmnt in sol if elmnt['day']==day
                                                                    and elmnt['depot'] == depot}
                    if (len(vehicles) > self.nb_veh):
                        f += self.veh_rent * (len(vehicles) - self.nb_veh)

        return f
    
    def remove_depot(self, feasible_sol):
        """This function generates a neighbor by removing a randomly selected depot
            It then replaces each delivery of this depot with the depot that is nearest to both the first and last client
            of this delivery.
        """
        sol = copy.deepcopy(feasible_sol)
        #orders = set(np.array([elmnt['sequence'] for elmnt in sol]).flatten())
        orders = set(el for elmnt in sol for el in elmnt['sequence'])
        in_sequence_depots = [elmnt for elmnt in orders if elmnt >= self.nb_orders]
        deliveries = [elmnt for elmnt in range(len(sol)) if set(in_sequence_depots) & set(sol[elmnt]['sequence'])]
        for deliv in deliveries:
            seq = sol[deliv]['sequence']
            sol[deliv]['sequence'] = [ x for x in seq if x not in in_sequence_depots]
        """
        if in_sequence_depots :
            for depot in in_sequence_depots:
                deliveries = [deliv for deliv in range(len(sol)) if depot in sol[deliv]['sequence']]
                for deliv in deliveries:
                    sol[deliv]['sequence'] = 
        """

        depots = set([element['depot'] for element in sol])
        if(len(depots)>=2):
            depot = np.random.choice(list(depots)) # We randomly choose a depot to remove  
            indexes = [i for i in range(len(sol)) if sol[i]['depot'] == depot] # Deliveries we need to modify
            # to remove the depot
            S = list(depots - {depot}) 
            for i in indexes:
                seq = sol[i]['sequence']
                first_client = seq[0]
                last_client = seq[-1]
                nearest_depot = np.argmin(self.distance_matrix[first_client , S]+
                                          self.distance_matrix[last_client , S])
                
                nearest_depot = S[nearest_depot]
                #print(f'{depot} -> {nearest_depot}')
                #print(sol[i])
                sol[i]['depot'] = nearest_depot
                #print(sol[i])
                l=[ elmnt['vehicle'] for elmnt in sol if (elmnt['depot'] == nearest_depot)
                                                        and (elmnt['day'] == sol[i]['day'])]
                if (len(l)>1):
                    for k in range(1,self.nb_veh+1):
                        if k not in l:
                            sol[i]['vehicle'] = k
                            k += 1 
                            break
                    if (k == self.nb_veh):
                        sol[i]['vehicle'] = max(l)+1 
                
            return sol
        else:
            return sol

    def add_depot(self, feasible_sol):
        """This function adds a depot that wasn't chosen for placement
        """
        sol = copy.deepcopy(feasible_sol)
        depots = [element['depot'] for element in sol]
        sites = self.nodes[self.sites].astype(int)
        candidate_depots = list(set(sites)-set(depots))
        if(len(candidate_depots)==0):
            return sol
        else:
            depot = np.random.choice(candidate_depots)
            deliv_indx = np.random.choice(range(len(sol)))
            sol[deliv_indx]['depot'] = depot
            return sol

    def replace_depot(self, feasible_sol):
        """This function replaces a depot in a delivery by another one"""
        sol = copy.deepcopy(feasible_sol)
        indx1 = np.random.choice(range(len(sol)))
        depot1 = sol[indx1]['depot']
        depots = {elmnt['depot'] for elmnt in sol if elmnt['depot'] != depot1 }
        if len(depots) == 0:
            return sol
        depot2 = np.random.choice(list(depots))

        sol[indx1]['depot'] = depot2

        return sol
    
    def change_depot(self, feasible_sol):
        """This function changes all occurences of a depot in the solution by another depot that wasn't chosen 
        in this solution"""
        sol = copy.deepcopy(feasible_sol)
        depots = set([element['depot'] for element in sol])
        depot = np.random.choice(list(depots)) # We randomly choose a depot to remove  
        indexes = [i for i in range(len(sol)) if sol[i]['depot'] == depot] # Deliveries we need to modify

        sites = self.nodes[self.sites].astype(int)
        S = list(set(sites) - {depot}) 
        if ( len(S) != 0):
            new_depot = np.random.choice(S)
            for i in indexes:
                sol[i]['depot'] = new_depot
        
        return sol

    def permute_delivery(self, feasible_sol):
        """This function permutes the sequences of two distinct deliveries"""
        sol = copy.deepcopy(feasible_sol)
        indx1, indx2 = np.random.choice(range(len(sol)), 2)
        deliv1 = sol[indx1]
        sol[indx1]['sequence'] = sol[indx2]['sequence'].copy()
        sol[indx2]['sequence'] = deliv1['sequence'].copy()

        return sol
    
    def inter_route_swap(self, feasible_sol):
        """This function swaps two orders in the same delivery"""
        sol = copy.deepcopy(feasible_sol)
        indx_delivery = np.random.randint(len(sol))
        seq = sol[indx_delivery]['sequence'].copy()

        indx1, indx2 = np.random.choice(range(len(seq)),2)
        seq[indx1] = seq[indx2]
        seq[indx2] = sol[indx_delivery]['sequence'][indx1]

        sol[indx_delivery]['sequence'] = seq

        return sol
    
    def intra_route_swap(self, feasible_sol):
        """ This function performs an intra route swap of orders"""

        sol = copy.deepcopy(feasible_sol)
        indx_delivery1, indx_delivery2 = np.random.randint(0, len(sol), 2)
        deliv1 = copy.deepcopy(sol[indx_delivery1])
        deliv2 = copy.deepcopy(sol[indx_delivery2])

        indx1 = np.random.choice(range(len(deliv1['sequence'])))
        indx2 = np.random.choice(range(len(deliv2['sequence'])))

        deliv1['sequence'][indx1] = deliv2['sequence'][indx2]
        deliv2['sequence'][indx2] = sol[indx_delivery1]['sequence'][indx1]

        sol[indx_delivery1] = deliv1
        sol[indx_delivery2] = deliv2

        return sol
    
    def move_order(self, feasible_sol):
        """This function moves an order from a delivery to another"""
        sol = copy.deepcopy(feasible_sol)
        indx_delivery1, indx_delivery2 = np.random.randint(0, len(sol), 2)
        deliv1 = copy.deepcopy(sol[indx_delivery1])
        deliv2 = copy.deepcopy(sol[indx_delivery2])

        indx1 = np.random.choice(range(len(deliv1['sequence'])))
        indx2 = np.random.choice(range(len(deliv2['sequence'])))

        order = deliv1['sequence'].pop(indx1)
        deliv2['sequence'].insert(indx2,order)
        sol[indx_delivery2]['sequence'] = deliv2['sequence']

        if not deliv1['sequence'] :
            sol.pop(indx_delivery1)

        return sol

    def concatenate_deliveries(self, feasible_sol):
        """This function chooses a day and concatenates two random deliveries taking place on this day """
        sol = copy.deepcopy(feasible_sol)
        days= set([elmnt['day'] for elmnt in sol ])#if len([i for i in range(len(sol)) if sol[i]['day']==elmnt['day']])>=2])
        day = np.random.choice(list(days))
        deliveries = [i for i in range(len(sol)) if sol[i]['day'] == day]
        if(len(deliveries) == 1):
            return sol
        indx1, indx2 = np.random.choice(deliveries, 2)
        seq1 = sol[indx1]['sequence'].copy()
        seq1.append(sol[indx2]['depot'])
        seq1.extend(sol[indx2]['sequence'])
        sol[indx1]['sequence'] = seq1
        sol.pop(indx2)
        return sol
    
    def move_date(self, feasible_sol):
        sol = copy.deepcopy(feasible_sol)
        indx = np.random.choice(range(len(sol)))
        delivery = sol[indx]
        min_DD = self.data[min(delivery['sequence'], key= lambda x: self.data[x,1]),1] #minimal delivery date
        min_LD = self.data[min(delivery['sequence'], key= lambda x: self.data[x,0]),0] #minimal loading date
        day = np.random.randint(min_LD , min_DD +1)
        sol[indx]['day'] = day
        return sol


    def generate_neighborhood(self, sol):
        nb_neigh = self.nb_neigh
        neighborhood =[]
        nb_functions = 10 # number of neighbor generation function ( add_depot, remove_depot ...)
        for i in range(nb_neigh):
            r = np.random.randint(0,nb_functions)
            if r == 0 :
                neigh = self.remove_depot(sol)
                #print('remove_depot')
            elif r == 1:
                neigh = self.add_depot(sol)
                #print('add_depot')
            elif r == 2:
                neigh = self.replace_depot(sol)
                #print('replace_depot')
            elif r == 3:
                neigh = self.change_depot(sol)
            elif r == 4:
                neigh = self.permute_delivery(sol)
                #print('permute_delivery')
            elif r == 5:
                neigh = self.inter_route_swap(sol)
                #print('inter_route_swap')
            elif r == 6:
                neigh = self.intra_route_swap(sol)
                #print('intra_route_swap')
            elif r == 7:
                neigh = self.move_order(sol)
                #print('move_order')
            elif r == 8:
                neigh = self.concatenate_deliveries(sol)
                #print('concatenate_deliveries')
            else:
                neigh = self.move_date(sol)
                #print('move_date')
            #print(self.isValid(neigh))
            neighborhood.append(neigh)
        return neighborhood
    
    def eps_greedy_policy(self, eps , Q, action, choices):
        """This policy function chooses an action( one function among the neighbor generation functions)"""
        if( np.all(Q == 0) or action == None):
            new_action = np.random.choice(range(len(Q)), p = np.exp(-choices) / np.sum(np.exp(-choices)))
        else:
            r = np.random.random()
            if(r < eps):
                new_action = np.random.choice(range(len(Q)))
            else:
                new_action = np.argmax(Q[action,:])
        return new_action
    
    def take_action(self, action, sol):
        if action == 0 :
            neigh = self.remove_depot(sol)
            #print('remove_depot')
        elif action == 1:
            neigh = self.add_depot(sol)
            #print('add_depot')
        elif action == 2:
            neigh = self.replace_depot(sol)
            #print('replace_depot')
        elif action == 3:
            neigh = self.change_depot(sol)
        elif action == 4:
            neigh = self.permute_delivery(sol)
            #print('permute_delivery')
        elif action == 5:
            neigh = self.inter_route_swap(sol)
            #print('inter_route_swap')
        elif action == 6:
            neigh = self.intra_route_swap(sol)
            #print('intra_route_swap')
        elif action == 7:
            neigh = self.move_order(sol)
            #print('move_order')
        elif action == 8:
            neigh = self.concatenate_deliveries(sol)
            #print('concatenate_deliveries')
        else:
            neigh = self.move_date(sol)

        reward = 1 / self.obj_function(neigh)

        return neigh, reward
    
    def generate_neighborhood_Q(self, sol, Q, eps = 0.3, gamma = 0.4, alpha = 1, lambd = 0.99):
        nb_neigh = self.nb_neigh        
        neighborhood =[]
        choices = np.zeros((10,))
        #Q = np.zeros((10,10))
        nb_functions = 10 # number of neighbor generation function ( add_depot, remove_depot ...)
        action = self.eps_greedy_policy(eps, Q, None, choices)
        choices[action] += 1
        for i in range(nb_neigh):
            eps = eps * lambd
            neigh, reward = self.take_action(action, sol)
            new_action = self.eps_greedy_policy(eps , Q, action, choices)
            Q[action, new_action] += alpha * (reward + gamma *np.max(Q[new_action]) - Q[action, new_action]) #+ 10*np.sqrt(1/(choices[action]+1e-2))
            action = new_action
            choices[action] += 1
            neighborhood.append(neigh)
            
        return neighborhood, Q

    def compute_weight(self, delivery):
        seq = delivery['sequence']
        weight = self.capacity

        for order in seq:
            if order < self.nb_orders:
                weight -= self.data[order,2]
            else:
                weight = self.capacity

        return weight
        
    def isValid(self, sol):
        days = {elmnt['day'] for elmnt in sol}
        depots = {elmnt['depot'] for elmnt in sol}

        for day in days:
            for depot in depots:
                for veh in range(1, self.nb_veh+1):
                    deliveries = [elmnt for elmnt in sol if elmnt['day'] == day
                                                            and elmnt['depot'] == depot
                                                            and elmnt['vehicle'] == veh]
                    if len(deliveries) > 1 :
                        print("Vehicle sent multiple times in one day")
                        return False
        """           
        vehicles = {elmnt['vehicle'] for elmnt in sol if elmnt['vehicle'] not in range(1,self.nb_veh+1)}
        if(len(vehicles) != 0):
            print("Vehicles not in range")
            return False
        """
        # Weight Constraint
        for elmnt in sol :
            weight = self.compute_weight(elmnt)
            if(weight > self.capacity):
                print("Weight not valid")
                return False
            
        return True
    
    def solve(self, max_iter=100, Q=None, QL = True, eps = 0.3, gamma = 0.4, alpha = 1, lambd = 0.9):
        """
        Implements the Tabu Search algorithm to solve the problem.
        
        Args:
            max_iter (int): Maximum number of iterations to perform.

        Returns:
            dict: The best solution found.
        """
        # Initialize solution
        #current_sol = self.initialize_sol()
        with open("feasible_sol_cluster_1.json", "r") as file:
            current_sol = json.load(file)
        best_sol = current_sol
        best_cost = self.obj_function(best_sol)
        current_cost = best_cost
        initial_value = self.obj_function(best_sol)
        Costs = [current_cost]
        # Tabu list and its size
        tabu_list = []
        tabu_size = 10  # You can tune this parameter

        if(Q == None):
            Q = np.zeros((10,10))

        for iteration in range(max_iter):
            print(f"Iteration {iteration + 1}/{max_iter}, "
                  f'Validity of current solution: {self.isValid(current_sol)}')
            depots = [self.nb_nodes - element['depot'] for element in current_sol]
            print(f'Chosen depots: {set(depots)}')
            # Generate the neighborhood of the current solution
            if not QL :
                neighborhood = self.generate_neighborhood(current_sol)
            else:
                neighborhood, Q = self.generate_neighborhood_Q(current_sol, Q,eps , gamma , alpha, lambd)
            # obj_function of all neighbors
            neighbor_costs = [(self.obj_function(neigh), neigh) for neigh in neighborhood]
            # Filter out neighbors in the tabu list
            valid_neighbors = [
                (cost, neigh) for cost, neigh in neighbor_costs if neigh not in tabu_list
                                                                and self.isValid(neigh)
            ]

            if not valid_neighbors:
                print("No valid neighbors found. Exiting...")
                Q = np.zeros((10,10))
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

            # Logging for debugging
            print(
                f"Current Cost: {current_cost}, Best Cost: {best_cost}"
            )
            print(" ")
            Costs.append(current_cost)

        print("Optimization completed.")
        print(f"From {initial_value} to {best_cost}")
        return best_sol, Costs
    
    def plot_delivery(self, delivery):
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
        days = {elmnt['day'] for elmnt in sol}
        for day in days:
            self.plot_day(sol, day)


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
    #np.random.seed(10)
    
    # Clustering
    model = cluster.cluster()
    model.partition("KMEANS")
    clusters = model.cluster_list()

    # Tabu
    tabu = tabu_LRP(clusters[1], model.len_order, model.len_depot, nb_neigh = 30)
    tabu.extract()
    partition = tabu.assign_client()

    
    with open("feasible_sol_cluster_1.json", "r") as file:
        sol = json.load(file)
    obs = np.zeros((5,))
    """for i in range(5):
        best_sol, Costs = tabu.solve(max_iter = 100, QL = False, eps = 0.3, gamma = 0.85, alpha = 1.5, lambd = 0.99)
        obs[i] = tabu.obj_function(best_sol)
        plt.plot(Costs, label ="QL = False", color = 'red')"""
    mean_f = np.mean(obs)
    std_f = np.std(obs)
    for i in range(5):
        best_sol, Costs = tabu.solve(max_iter = 100, QL = True, eps = 0.3, gamma = 0.6, alpha = 3, lambd = 0.99)
        obs[i] = tabu.obj_function(best_sol)
        plt.plot(Costs, label ="QL = True ", color = 'green')
    mean_t = np.mean(obs)
    std_t = np.std(obs)
    print(f'means: f: {mean_f}, t: {mean_t}')
    print(f'stq: f: {std_f}, t: {std_t}')

    """
    with open("feasible_sol_.json", "w") as file:
        json.dump(best_sol, file,  default=convert_to_serializable, indent=4)
    """
    #plt.plot(Costs)
    plt.xlabel('Iteration number')
    plt.ylabel('Obj Func value')
    plt.title('Convergence curve of OF')
    plt.xlim(90,100)
    plt.ylim(300000,500000)
    plt.show()
    #tabu.plot(best_sol)
    
    
    

    #colors = generate_colors(200)
    #np.random.shuffle(colors)
    """colors = ['blue','green','yellow', 'brown', 'purple', 'orange']
    for part in partition.keys():
        
        plt.scatter(tabu.coords[tabu.sites[part],1],tabu.coords[tabu.sites[part],0], color='black')

        if(partition[part] != []):
            
            #plt.scatter(tabu.coords[partition[part],0],tabu.coords[partition[part],1], color=colors[part])
            lo = list(tabu.coords[partition[part],1])
            la = list(tabu.coords[partition[part],0])
            geometry = [Point(xy) for xy in zip(lo,la)]
            geo_df = gpd.GeoDataFrame(geometry = geometry)
            geo_df.plot(ax = ax,color=colors[part],marker='*')
            #plt.scatter(tabu.coords[tabu.sites[part],1],tabu.coords[tabu.sites[part],0], color='red')
            plt.text(tabu.coords[tabu.sites[part],1],tabu.coords[tabu.sites[part],0], f'depot {part}', c='red',fontsize=8)
    plt.show()
    
    # Save to JSON file

    sol = tabu.initialize()
    with open("feasible_sol_new.json", "w") as file:
        json.dump(sol, file,  default=convert_to_serializable, indent=4)

    with open("feasible_sol_2.json", "w") as file:
        json.dump(feasible_sol, file,  default=convert_to_serializable, indent=4)

    with open("feasible_sol_2.json", "r") as file:
        loaded_data = json.load(file)
        print(loaded_data)
    #for d in range(int(lrp.data.day_charge), int(lrp.data.day_deliver)):
    #   lrp.visualise(d)
    """
