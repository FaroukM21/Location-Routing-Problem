Le seul dataset manquant dans ce git est celui des valeurs foncières vu sa taille importante. 
Si nécessaire, vous pouvez le trouver sur le site data.gouv.fr

---------------------------------------------------------------------------------
MILP: Model fonctionnel et exact. Cependant, le temps de calcul rend son utilisation dans la résolution de ce problème impossible.

---------------------------------------------------------------------------------
Métaheuristique: l'idée consiste à divise nos données en des partitions et d'appliquer une Métaheuristiquesur chaque cluster considéré
comme un sous problème LRP.


Clustering: le code cluster.py effectue le clustering des données. J'ai essayé de faire le clustering en prenant en compte
les occurences de chaque clients. Le résultat donné par cette approche ( code cluster_occurences.py ) montre que les 
clusters ne sont pas géographiquement raisonnables. les clients de chaque cluster sont éparpillés partout en France.

La deuxième approche consiste en simplement prendre en compte les emplacements GPS des clients/depots. La seule décision à prendre
est soit on inclut les sites éligibles dans le dataset de clustering, soit on ne les inclut pas et on les affecte à des clusters
par un algorithme séparé du clustering.

Dans le cas où on inclut les depots dans le dataset, on remarque DBSCAN propose un partitionnement en plusieurs clusters (19 partitions)
ce qui n'est pas adapté à notre problème. KMeans par contre permet d'avoir un très bon clustering. 


