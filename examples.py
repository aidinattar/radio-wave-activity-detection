# %%
from DataReader import DataReader

subjects = [f'subject_{i:02d}' for i in range(5, 6)]
sets     = [f'set{i:03d}'      for i in range(1, 2)]

a = DataReader(subjects=subjects, sets=sets, do_rdn=True, do_mDoppler=True)
a.remove_static_bins()
a.rescaling()
a.filter_mDoppler(size=(15, 15), sigma=10)
# %%
#a.Plot_Gif_rdn(60)
# %%
a.plot_rdn_map()
# %%
#a.plot_rdn_map_3d()
# %%
a.Plot_Gif_mDoppler(100)
# %%
a.plot_mDoppler_map(start=161, stop=208)
# %%
#a.plot_mDoppler_map_3d()
# %%
a.divide_actions(11)
# %%
