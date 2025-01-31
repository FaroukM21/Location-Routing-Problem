from distance import *
import pandas as pd
import numpy as np

class milp_data:
    def __init__(self, depots, orders, num_veh,  capacity, V_mean=30, C_k = 0.615):
        """ This class defines the data necessary for the MILP program.
        Input:
            eligible sites to take into consideration: depots
            orders: list of indexes of desired orders : orders
            number of vehicles per depot: num_veh
            Capacity of each vehicle: capacity
            Mean speed of a delivery vehicle: V_mean (default=30)
            Kilometric coefficient (Formule du trinôme): C_k (default = 0.615)
        """
        self.num_depot = len(depots)
        self.depots = depots
        self.num_commande = len(orders)
        self.orders = orders
        self.num_nodes = self.num_commande + self.num_depot 
        self.nodes = [*self.orders , *self.depots ]
        self.num_veh = num_veh
        self.capacity = capacity
        self.V_mean = V_mean # km/h
        self.C_k = C_k # terme kilométrique de la formule du trinôme
        

        # Import data
        order = pd.read_csv("./data/Order.csv")
        coordinates = np.load('./data/coordinates.npy')
        data_sites = np.array(pd.read_csv("./data/datasetSites.csv"))

        self.coordinates = coordinates

        # Check parameters' validity

        if( self.num_commande > len(order)-len(data_sites)):
            print("The number of orders chosen exceeds the forcasted deliveries in the dataset")
            self.num_commande = len(order)-len(data_sites)
        if( self.num_depot > len(data_sites)):
            print("The number of depots chosen exceeds the available number of eligible sites")
            self.num_depot = len(data_sites)

        # Compute Orders matrix
        self.order_matrix = np.concatenate([order.iloc[self.orders]
                                            , order.iloc[self.depots]])
        
        self.order_matrix[:,0:2] = self.order_matrix[:,0:2].astype(int) # Date de chargement et date de livraison 
        # doivent être en jours et commencent par le 31/12/2020.
        # Dans le dataset des commandes, on simplifie l'étude en prenant en compte uniquement les poids des commandes


        # Compute Distance Matrix
        coords_horizon = np.concatenate([coordinates[self.orders]
                                       , coordinates[self.depots]]
                                       , axis=0)
        self.coords = coords_horizon
        distance_vector = pdist(coords_horizon, metric=haversine)
        self.distance_matrix = squareform(distance_vector)

        # Coefficients C_i in Objective function
        v=np.array([-0.01,-0.08,1000,750,365000,675*3])
        self.coeff_sites=data_sites.dot(v)

        if(self.num_commande<100):
            self.coeff_sites=data_sites.dot(v)/100 # On diminue le scale des coefficients durant les tests avec un 
            # nombre restreint de commandes

        # Loading date
        self.charge = order.iloc[self.orders]["DateChargement_jour"]
        self.day_charge = self.charge.min()
        print(f"minimum date of loading { self.day_charge}")
        
        # Delivery date
        self.deliver = order.iloc[self.orders]["DateLivraison_jour"]
        self.day_deliver = self.deliver.max()
        print(f"maximum date of delivery { self.day_deliver}")

        

if __name__=='__main__':
    np.random.seed(10)
    M = np.load('data/matrice_distances.npy')
    print(M.shape)
    


    