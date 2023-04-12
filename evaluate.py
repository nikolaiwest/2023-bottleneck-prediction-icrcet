# %%
import pickle 
from prediction import BottleneckPrediction

params = {
    "mdl_scenario": "100-100-10-10_100e_train_000-500_test_900-1000_inkl-bn",
    "n_steps_in" : 100,
    "n_steps_out" : 100,
    "aggreg_lvl" : 10, 
    "stride_lvl" : 10, 
    "range_train" : range(0, 500), 
    "range_test" : range(900, 1000), 
    "training_epochs" : 100, 
    "include_bottleneck_in_training_data" : True, 
    "include_one_hot_encoding_of_y_label" : False
    }

bottleneck_prediction = BottleneckPrediction(**params)
bottleneck_prediction.run()


#%%
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)
# sample usage
save_object(bottleneck_prediction, f'models/{bottleneck_prediction.mdl_scenario}.pkl')

#%%



#%%

bottleneck_prediction.plot_some_predictions(stride=50)

#%%

