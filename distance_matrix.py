import numpy as np
from scipy.spatial.distance import squareform, pdist
import pandas as pd

def haversine(coord1, coord2):
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Extract latitudes and longitudes, converting to radians
    
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

if __name__ =="__main__":

    # Prepare GPS coordinates
    df = pd.read_csv('data/Dataset_Livraisons_clean.csv')
    sites = pd.read_excel('data/Sites éligibles.xlsx')
    coords_sites = pd.DataFrame({'Latitude':list(sites.Coordonnées.str.split(',').str[0]+'.'+sites.Coordonnées.str.split(',').str[1]),
                'Longitude':list(sites.Coordonnées.str.split(',').str[2]+'.'+sites.Coordonnées.str.split(',').str[3])
                }).astype({'Latitude':'float64','Longitude':'float64'})
    combined_df = pd.concat([df[['Latitude','Longitude']],coords_sites] , ignore_index=True)

    latitudes = combined_df['Latitude']
    longitudes = combined_df['Longitude']
    coordinates = np.column_stack((latitudes, longitudes))

    # Compute the pairwise distances using pdist
    distance_vector = pdist(coordinates, metric=haversine)

    # Convert to a symmetric distance matrix
    distance_matrix = squareform(distance_vector)

    # Save the distance matrix
    np.save("data/matrice_distances.npy", distance_matrix)