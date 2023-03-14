# %%
from clustering import preprocess

subjects = [f'subject_{i:02d}' for i in range(5, 6)]
sets     = [f'set{i:03d}'      for i in range(1, 2)]

a = preprocess(subjects, sets, True, True)
a.remove_static_bins()
#a.rescaling('min_max')
#a.rescaling('log')
a.rescaling()
a.filter_mDoppler(size=(15, 15), sigma=10)
# %%
#a.Plot_Gif_rdn(60)
# %%
#a.plot_rdn_map()
# %%
#a.plot_rdn_map_3d()
# %%
a.Plot_Gif_mDoppler(100)
# %%
a.plot_mDoppler_map()
# %%
#a.plot_mDoppler_map_3d()
# %%
#a.dbscan()
# %%
a.kmeans()
# %%
a.hierarchical_clustering()
# %%
a.GMM()
# %%
a.hdbscan(min_cluster_size=10, min_samples=10, alpha=1.0, metric='euclidean')

# %%
a.gridsearch()
# %%
