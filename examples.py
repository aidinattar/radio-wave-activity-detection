# %%
from preprocessing.DataReader import DataReader
from preprocessing.DataCutter import DataCutter
from preprocessing.DataProcess import DataProcess
from preprocessing.dataset     import Dataset

subjects = [f'subject_{i:02d}' for i in range(5, 6)]
sets     = [f'set{i:03d}'      for i in range(1, 2)]

a = DataReader(subjects=subjects, sets=sets, do_rdn=True, do_mDoppler=True)
a.remove_static_bins()
a.crop_mDoppler()
a.crop_rdn()
a.rescaling()
a.filter_mDoppler(size=(15, 15), sigma=10)

# %%
#a.Plot_Gif_rdn(60)
# %%
a.plot_rdn_map()
# %%
a.plot_rdn_map_3d(k=0, range_length=6000)
# %%
a.Plot_Gif_mDoppler(100)
# %%
a.plot_mDoppler_map(start=161, stop=208)
# %%
#a.plot_mDoppler_map_3d()
# %%
a.timestamp_to_bins(11)
# %%
a.plot_divided_actions()

# %%
b = DataCutter(a)
b.cut(11)
# %%
b.create_labels_list()
# %%
b.labels_to_int()

# %%
b.save()
# %%
c = DataProcess(b)
# %%
c.cut_time()
# %%
c.padding(padding=40)
# %%
c.save()
# %%
d = Dataset(path='DATA_preprocessed', file='data_processed.npz')
# %%
