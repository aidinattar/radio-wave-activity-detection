# %%
from clustering import preprocess

subjects = [f'subject_{i:02d}' for i in range(5, 6)]
sets     = [f'set{i:03d}'      for i in range(1, 2)]

a = preprocess(subjects, sets, True, True)
a.remove_static_bins()
a.rescaling('norm')
# %%
a.Plot_Gif_rdn(60)
# %%
a.plot_rdn_map()
# %%
a.plot_rdn_map_3d()
# %%
a.Plot_Gif_mDoppler(30)
# %%
a.plot_mDoppler_map()
# %%
a.plot_mDoppler_map_3d()