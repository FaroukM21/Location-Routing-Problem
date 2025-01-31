from ortools.linear_solver import pywraplp 
from milp_data import *
import numpy as np
import time
import matplotlib.pyplot as plt
from  tabu_V1 import *
import cluster 

class milp_model():

    """This class defines the MILP model using OR-Tools.
        It also defines visualisation function self.visualise(k,d,s) 
        and client sequencing function self.coord_seq(k,d,s)
        Input:
            number of eligible sites to take into consideration: num_depot
            number of orders: num_commande
            number of vehicles per depot: num_veh
            number of days to consider stating from the 31/12/2020: horison
            Capacity of each vehicle: capacity
            Mean speed of a delivery vehicle: V_mean (default=30)
            Kilometric coefficient (Formule du trinôme): C_k (default = 0.615)
            Solver for OR-Tools: solver (default="SCIP")
    """

    def __init__(self, depots, orders, num_veh, capacity, V_mean=30, C_k = 0.615, solver="SCIP"):

        self.solver = solver
        self.data = milp_data(depots, orders, num_veh, capacity)
        self.solved = False


    def solve(self):
        # Create the solver 
        lrp = pywraplp.Solver.CreateSolver(self.solver)
        if not lrp:
            print(f"{self.solver} solver not available.")
            return
        
        # Define decision variables

        u = {} # u_i_j_k_d_s :  binary decision variable indication that vehicle k of depot s 
        # delivers order i and then j on day d

        st = {} # st_{i}_{k}_{d}_{s} : continuois decision variable indicating the time of delivery of order i 
        # by vehicle k of depot s on day d, we ignore these variables when order i isn't delivered on day d
        # by this vehicle

        gamma={} # gamma_{s} : binary decision variable indicating if eligible depot s should or should not be placed

        delta={} # delta_{k}_{d}_{s} : binary decision variable indication the departure time window 1 ( 6 A.M to 2 P.M)
        # Or 2 ( 2 P.M to 10 P.M)

        q={} # q_{i}_{k}_{d}_{s} :  continuous decision variable that indicates the quantity available in 
        # vehicle k of depot s just before delivering order i on day d. We ignore the value given by this variable
        # if order i isn't delivered by vehicle k of depot s on day d.
        t0=time.time()
        for s in range(self.data.num_depot):

            gamma[s] = lrp.BoolVar(f"gamma_{s}")

            for d in range(int(self.data.day_charge), int(self.data.day_deliver)):
                for k in range(self.data.num_veh):

                    delta[k,d,s]=lrp.BoolVar(f"delta_{k}_{d}_{s}")

                    for i in range(self.data.num_nodes):
                        for j in range(self.data.num_nodes):
                            
                            if (i != j) and ( i<self.data.num_nodes-self.data.num_depot or
                                              j<self.data.num_nodes-self.data.num_depot): 
                                # We only take into consideration deliveries (arcs) of different orders (i !=j) and 
                                # and arcs in which veritces are of different types( one is an order and the other is an eligible site)
                                                    
                                u[i, j, k,d,s] = lrp.BoolVar(f"u_{i}_{j}_{k}_{d}_{s}")
                
                    for i in range(self.data.num_nodes):
                        st[i, k,d,s] = lrp.NumVar(0.0,lrp.infinity(),f"st_{i}_{k}_{d}_{s}")
                        q[i, k,d,s] = lrp.NumVar(0.0,lrp.infinity(),f"q_{i}_{k}_{d}_{s}")
        
        print(f'Number of variables = {lrp.NumVariables()}')

        # Define Objective function

        f = 0
        for s in range(self.data.num_depot):

            f += self.data.coeff_sites[self.data.coeff_sites.shape[0] - self.data.num_depot - s ] * gamma[s]

            for d in range(int(self.data.day_charge), int(self.data.day_deliver)):
                for k in range(self.data.num_veh):
                    for i in range(self.data.num_nodes):
                        for j in range(self.data.num_nodes):
                            if (i != j) and (i < self.data.num_nodes - self.data.num_depot or j < self.data.num_nodes - self.data.num_depot):
                                f += self.data.distance_matrix[i][j] * self.data.C_k * u[i, j, k, d, s]

        lrp.Minimize(f)

        # Define Constraints
        l=list(u.items())

        # At least one depot needs to be placed
        lrp.Add(sum(list(gamma.values()))>=1,f"Place_Depot{s}")

        
        for s in range(self.data.num_depot):
            lrp.Add(sum([l[j][1] for j in range(len(l)) if (l[j][0][0]==self.data.num_nodes-self.data.num_depot+s)
                                                         or (l[j][0][1]==self.data.num_nodes-self.data.num_depot+s) 
                                                         or (l[j][0][4]==s)]) <= gamma[s] * 1e6 ,f"StartDepot_{s}")
            for d in range(int(self.data.day_charge), int(self.data.day_deliver)):
                for k in range(self.data.num_veh):
                    # Vehicle k of depot s always departs from depot s
                    lrp.Add(sum([l[j][1] for j in range(len(l)) 
                                 if (l[j][0][4]==s) and (l[j][0][3]==d) and (l[j][0][2]==k)])
                            <=
                            sum([l[j][1] for j in range(len(l)) 
                                 if (l[j][0][0]==self.data.num_nodes-self.data.num_depot+s) and (l[j][0][4]==s) and (l[j][0][3]==d) and (l[j][0][2]==k)])*1e6,f"Depart1_{s}_{d}_{k}")
                    
                    
                    for i in range(self.data.num_nodes):
                        # A vehicle that delivers an order i needs to pass by the node i
                        lrp.Add(sum([l[j][1] for j in range(len(l)) if (l[j][0][0]==i) and (l[j][0][4]==s) and (l[j][0][3]==d) and (l[j][0][2]==k)])==
                                sum([l[j][1] for j in range(len(l)) if (l[j][0][1]==i) and (l[j][0][4]==s) and (l[j][0][3]==d) and (l[j][0][2]==k)]),f"Permute_{s}_{d}_{k}_{i}")
                        
                        """for j in range(self.data.num_nodes):
                            # We can't deliver starting from a depot that isn't placed
                            if (i != j) and (i < self.data.num_nodes - self.data.num_depot or j < self.data.num_nodes - self.data.num_depot):
                                lrp.Add(u[i,j,k,d,s]<=gamma[s])
                        """
                        # Capacity constraints
                        if(i>=self.data.num_nodes-self.data.num_depot):
                            lrp.Add(q[i,k,d,s]==self.data.capacity)
                        else:
                            for j in range(self.data.num_nodes):
                                if(i!=j):
                                    lrp.Add(q[i,k,d,s]-self.data.order_matrix[i,2]-q[j,k,d,s] <= (1-u[i,j,k,d,s])*1e6)
                                    #lrp.Add(q[j,k,d,s]-(q[i,k,d,s]-Matrix_orders[i,2]) <= (1-u[i,j,k,d,s])*1e6)
                            lrp.Add(q[i,k,d,s]<=self.data.capacity)

                        # Time-Window Constraints
                        if(i==self.data.num_nodes-self.data.num_depot+s):
                            lrp.Add( st[i,k,d,s]==24*d+6*(1-delta[k,d,s])+14*delta[k,d,s], f"SetDepart_{s}_{d}_{k}_{i}")
                        else:
                            for j in range(self.data.num_nodes):
                                if( (i!=j )and (i < self.data.num_nodes - self.data.num_depot or j < self.data.num_nodes - self.data.num_depot) ):
                                    lrp.Add(st[i,k,d,s]+self.data.distance_matrix[i,j]/self.data.V_mean-st[j,k,d,s] <= (1-u[i,j,k,d,s])*1e6 , f"CumulativeTime_{s}_{d}_{k}_{i}_{j}")
                                    #lrp.Add(st[j,k,d,s]-(st[i,k,d,s]+self.data.distance_matrix[i,j]/self.data.V_mean) <= (1-u[i,j,k,d,s])*1e6)
                            if(i<self.data.num_nodes-self.data.num_depot):
                                lrp.Add( st[i,k,d,s] <= self.data.order_matrix[i,1]*24  ,f"TW_{i}_{k}_{d}_{s}")

        # Each order needs to be delivered in the chosen time horizon
        for i in range(self.data.num_nodes-self.data.num_depot):
            lrp.Add(sum([l[j][1] for j in range(len(l)) if l[j][0][0]==i])==1 ,f"Deliver_{i}")

        # A delivery takes place only after the loading date

        for d in range(int(self.data.day_charge), int(self.data.day_deliver)):
            for i in range(self.data.num_nodes):
                if (i<self.data.num_nodes-self.data.num_depot):
                    lrp.Add(int(self.data.order_matrix[i,0]) <= d + (1-sum([l[j][1] for j in range(len(l)) 
                                 if (l[j][0][0]==i) and (l[j][0][3]==d)]))*1e6,f"Load_{d}_{i}")
                    
                    
        print(f"Number of constraints = {lrp.NumConstraints()}")
        # Solve the problem
        status = lrp.Solve()
        t1=time.time()

        # Save .lp model
        lp_model = lrp.ExportModelAsLpFormat(False)
        # Save the string to a file
        file_path = "model.lp"
        with open(file_path, "w") as file:
            file.write(lp_model)

        # Print computation time
        print(f"Computation time = {t1-t0}")
        
        # Check the result
        if status == pywraplp.Solver.OPTIMAL:
            print('Solution:')
            print(f'Objective value = {lrp.Objective().Value()}')
            print(f'gamma = {[y.solution_value() for y in list(gamma.values())]}')
            self.u = u
            self.gamma = gamma
            self.st = st
            self.delta = delta
            self.q = q
            self.solved = True
            self.solution = lrp

        elif status == pywraplp.Solver.INFEASIBLE:
            print('No feasible solution found.')
            
        elif status == pywraplp.Solver.UNBOUNDED:
            print('The problem is unbounded.')
        else:
            print('Solver ended with status:', status)


    def coord_seq(self, k, d, s):
        """Function to visualize the delivery route of vehicle k of depot s on day t
        output: sequence of coordinates
        """

        assert self.solved==True, "The optimal solution needs to be found before visualisation"
        assert k < self.data.num_veh, f"Choose a number of vehicles less than {self.data.num_veh}"
        assert d < self.data.horison, f"Choose a date before {self.data.horison}"
        assert s < self.data.num_depot, f"Choose a depot number less than {self.data.num_depot}"

        l=list(self.u.items())
        # Check wheter depot s sent k for delivery on day d
        if sum(l[j][1].solution_value() for j in range(len(l)) if (l[j][0][2]==k) and (l[j][0][3]==d) and (l[j][0][4]==s)):    
            
            i = self.data.num_nodes-self.data.num_depot+s # starting vertex
            coordinates=[(self.data.coords[i,0],self.data.coords[i,1])] #list of coordinates (latitude,longitude)

            for key in list(self.u.keys()):                
                if (self.u[key].solution_value()==1) and (key[0]==i) and (key[2]==k) and (key[3]==d) and (key[4]==s):
                    j=key[1]
                    break
            #print(f"from {i} to {j}")

            # Loop to build the sequence of coordinates
            while (j!=self.data.num_nodes-self.data.num_depot+s):
                i=j
                coordinates.append((self.data.coords[i,0],self.data.coords[i,1]))
                for key in list(self.u.keys()):                
                    if (self.u[key].solution_value()==1) and (key[0]==i):
                        j=key[1]
                        break
                #print(f"from {i} to {j}")
            coordinates.append((self.data.coords[j,0],self.data.coords[j,1]))
            return coordinates
        
        else:
            print(f"Vehicle {k} of depot {s} isn't sent for delivery on day {d}")
            return []

    def visualise(self, k, d, s):
        coordinates = self.coord_seq(k, d, s)
        if len(coordinates) != 0:
            """route_x,route_y = zip(*coordinates)
            plt.plot(route_x, route_y, label="Vehicle Route", color='green', marker='o')
            """
            plt.scatter(self.data.coords[self.data.coords.shape[0]-self.data.num_depot:,0], self.data.coords[self.data.coords.shape[0]-self.data.num_depot:,1] , c='yellow', s=100, label=f'Other depots')
            plt.scatter(coordinates[0][0], coordinates[0][1], c='red', s=100, label=f'Depot {s}')  # Larger size for depot
            
            plt.scatter(self.data.coordinates[:self.data.num_commande,0],self.data.coordinates[:self.data.num_commande,1])
            for i in range(len(coordinates) - 1):
                start = coordinates[i]
                end = coordinates[i+1]
                
                # Draw an arrow between consecutive locations
                plt.arrow(
                    start[0], start[1],        # Start point
                    end[0] - start[0],         # Change in x
                    end[1] - start[1],         # Change in y
                    head_width=0.2,            # Arrowhead width
                    length_includes_head=True, # Ensure the arrowhead is part of the line
                    color='green'              # Arrow color
                )
            # Add labels, legend, and title
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f"Routing of delivery vehicle {k} of depot {s} on day {d}")
            plt.legend()
            plt.grid(True)
            plt.show()

            
if __name__ == "__main__":
    np.random.seed(10)
    model = cluster.cluster()
    model.partition("KMEANS")
    clusters = model.cluster_list()
    tabu = tabu_LRP(clusters[1], model.len_order, model.len_depot, nb_neigh = 30)
    tabu.extract()
    num_veh = 2
    lrp = milp_model(tabu.nodes[tabu.sites].astype(int), tabu.nodes[tabu.orders].astype(int), num_veh, capacity=100, solver="CP-SAT")
    lrp.solve()

    #lrp.visualise(0,4,1)
    """for s in range(num_depot):
        for k in range(num_veh):
            for d in range(horison):
                lrp.visualise(k,d,s)"""
    
    