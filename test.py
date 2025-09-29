from mesh_simlification_to_graph import SingleCentroidGraph
print("SingleCentroidGraph", " loaded")
#import pymesh

mesh_path =  "./data/SHREC2021-main/dataset/test_set/test_OFFs/1.off"                   #"./data/airplane_0627.off"
#mesh = pymesh.load_mesh(mesh_path)
#print(mesh)


builder = SingleCentroidGraph(
    mesh_path=mesh_path,
    properties_dir="./data/SHREC2021-main/dataset/test_set/properties"
)

G = builder.graph         # networkx.Graph
C = builder.centroids     # np.ndarray (C, D)

# sauvegarde optionnelle
builder.save_pickle("./output_graphs/1_test.pkl")
