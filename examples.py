# %%
from preprocessing.DataReader import DataReader
from preprocessing.DataCutter import DataCutter
from preprocessing.DataProcess import DataProcess
from preprocessing.dataset     import Dataset

subjects = [f'subject_{i:02d}' for i in range(5, 6)]
sets     = [f'set{i:03d}'      for i in range(1, 2)]

dr = DataReader(subjects=subjects, sets=sets, do_rdn=False, do_mDoppler=True)
dr.remove_static_bins()
dr.crop_mDoppler()
#dr.crop_rdn()
dr.rescaling()
dr.filter_mDoppler(size=(21, 11), sigma=15)

# %%
#dr.Plot_Gif_rdn(60)
# %%
dr.plot_rdn_map()
# %%
dr.plot_rdn_map_3d(k=0, range_length=6000)
# %%
dr.Plot_Gif_mDoppler(100)
# %%
dr.plot_mDoppler_map(start=161, stop=208)
# %%
#dr.plot_mDoppler_map_3d()
# %%
dr.timestamp_to_bins(11)
# %%
dr.plot_divided_actions()

# %%
dc = DataCutter(dr)
dc.cut(11)
# %%
dc.create_labels_list()
# %%
dc.labels_to_int()

# %%
dc.save()
# %%
dp = DataProcess(dc)
# %%
dp.cut_time(loc='normal')
# %%
dp.padding(padding=40)
# %%
#dp.rotate()
# %%
dp.save()
# %%
d = Dataset(path='DATA_preprocessed', file='data_processed.npz', type='mDoppler')
# %%
