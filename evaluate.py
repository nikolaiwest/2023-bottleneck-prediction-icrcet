# project
from prediction import BottleneckPrediction, BottleneckBenchmark

# other
import tensorflow as tf

# set default values for the validation parameters 
n_steps_in = 60 # input window
n_steps_out = 30 # prediction horizon
aggreg_lvl = 1 # set aggregation value for buffer level
stride_lvl = 10 # set stride for sequence split

num_of_folds = 10 # ten fold cross validation 
num_of_exper = 1000 # total number of simulation runs (each with 10k steps)
len_of_folds = int(num_of_exper/num_of_folds) 

# check GPU configuration 
print(tf.config.list_physical_devices('GPU'))


# perform a ten fold validation (= 900 train and 100 test simulations with 10k observations each)
for i in range(num_of_folds):
    # get range values for fold 
    range_train = [x for x in range(num_of_exper) if x not in range(i*len_of_folds, (i+1)*len_of_folds)]
    range_test = list(range(i*len_of_folds, (i+1)*len_of_folds))
    # describe 
    print(f"Running fold {i}:")
    print(f"- Test =  'data between {min(range_test)} to {max(range_test)} (n={len(range_test)})'")
    print(f"- Train = 'all remaining data (n={len(range_train)})'")
    # set parameter for fold 
    params = {
        "mdl_scenario": f"in{n_steps_in}-out{n_steps_out}-agg{aggreg_lvl}-std{stride_lvl}_fold{i}_bn_oh",
        "n_steps_in" : n_steps_in, 
        "n_steps_out" : n_steps_out, 
        "aggreg_lvl" : aggreg_lvl, 
        "stride_lvl" : stride_lvl, 
        "range_train" : range_train, 
        "range_test" : range_test, 
        "training_epochs" : 20, 
        "include_bottleneck_in_training_data" : True, 
        "include_one_hot_encoding_of_y_label" : True
        }

    # get prediction and run 
    bottleneck_prediction = BottleneckPrediction(**params)
    bottleneck_prediction.run()


# set default parameter to run the benchmark
params = {
    "mdl_scenario": f"in{n_steps_in}-out{n_steps_out}-agg{aggreg_lvl}-std{stride_lvl}_benchmarking",
    "n_steps_in" : n_steps_in, 
    "n_steps_out" : n_steps_out, 
    "aggreg_lvl" : aggreg_lvl, 
    "stride_lvl" : stride_lvl, 
    "range_train" : range(0), # no training needed for benchmarking
    "range_test" : range(1000), # include all data
    "training_epochs" : 20, 
    "include_bottleneck_in_training_data" : True, 
    "include_one_hot_encoding_of_y_label" : False
    }

# random benchmark predicts a random value for each bottleneck in y_pred
benchmark1 = BottleneckBenchmark("random", params)
benchmark1.run()
benchmark1.describe_pred()

# naive benchmark predicts always the last bottleneck from x_test for y_pred
benchmark2 = BottleneckBenchmark("naive", params)
benchmark2.run()
benchmark2.describe_pred()