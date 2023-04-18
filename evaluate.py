# project
from auxiliaries import save_data
from prediction import BottleneckPrediction, BottleneckBenchmark

# set default values for the modeling scenario 
n_steps_in = 60 # input window
n_steps_out = 30 # prediction horizon
aggreg_lvl = 1 # set aggregation value for buffer level
stride_lvl = 10 # set stride for sequence split

# set additional parameter for the evaluation 
num_of_folds = 10 # ten fold cross validation 
num_of_exper = 1000 # total number of simulation runs (each with 10k steps)
len_of_folds = int(num_of_exper/num_of_folds) 

# optional: check GPU configuration 
# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))


# Experiment 1: 
# Bottleneck Prediction WITHOUT including the bottleneck in the training data, called 'input=[4x60]'
# in60-out30-agg1-std10_fold{i}_oh

# get emtpy list to store some results during cross validation 
results_experiment_1 = []

# perform a ten fold cross validation 
# 900 train and 100 test simulations with 10k observations each
for i in range(num_of_folds):
    print(i)
    # get range values for fold 
    range_train = [x for x in range(num_of_exper) if x not in range(i*len_of_folds, (i+1)*len_of_folds)]
    range_test = list(range(i*len_of_folds, (i+1)*len_of_folds))
    
    # describe 
    print(f"Running fold {i}:")
    print(f"- Test =  'data between {min(range_test)} to {max(range_test)} (n={len(range_test)})'")
    print(f"- Train = 'all remaining data (n={len(range_train)})'")
    
    # set parameter for fold 
    params = {
        "mdl_scenario": 
            f"in{n_steps_in}-out{n_steps_out}-agg{aggreg_lvl}-std{stride_lvl}_fold{i}_oh",
        "n_steps_in" : n_steps_in, 
        "n_steps_out" : n_steps_out, 
        "aggreg_lvl" : aggreg_lvl, 
        "stride_lvl" : stride_lvl, 
        "range_train" : range_train, 
        "range_test" : range_test, 
        "training_epochs" : 20, 
        "include_bottleneck_in_training_data" : False, 
        "include_one_hot_encoding_of_y_label" : True
        }

    # make a new bottleneck prediction
    bottleneck_prediction = BottleneckPrediction(**params)
    # run prediction and get results 
    result_from_fold = bottleneck_prediction.run()
    # add results from fold to list 
    results_experiment_1.append(result_from_fold)
    bottleneck_prediction.describe_data()

# save the results from experiment 1 for later plotting
save_data(results_experiment_1, 'results_experiment_1')


#%% Experiment 2: 
# Bottleneck Prediction that INCLUDES the bottleneck in the training data, called 'input=[5x60]'
# in60-out30-agg1-std10_fold{i}_oh_bn

# get emtpy list to store some results during cross validation 
results_experiment_2 = []

# perform a ten fold cross validation 
# 900 train and 100 test simulations with 10k observations each
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
        "mdl_scenario": 
            f"in{n_steps_in}-out{n_steps_out}-agg{aggreg_lvl}-std{stride_lvl}_fold{i}_oh_bn",
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

    # make a new bottleneck prediction
    bottleneck_prediction = BottleneckPrediction(**params)
    # run prediction and get results 
    result_from_fold = bottleneck_prediction.run()
    # add results from fold to list 
    results_experiment_2.append(result_from_fold)

# save the results from experiment 2 for later plotting
#save_data(results_experiment_2, 'results_experiment_2')


# Benchmarking: 
# Bottleneck Prediction using some simplified approaches as comparative benchmark 

# set the default parameter to run the benchmark
params = {
    "mdl_scenario": 
        f"in{n_steps_in}-out{n_steps_out}-agg{aggreg_lvl}-std{stride_lvl}_benchmarking",
    "n_steps_in" : n_steps_in, 
    "n_steps_out" : n_steps_out, 
    "aggreg_lvl" : aggreg_lvl, 
    "stride_lvl" : stride_lvl, 
    "range_train" : range(0), # no training needed for benchmarking
    "range_test" : range(1000), # include all data
    "training_epochs" : 20, 
    "include_bottleneck_in_training_data" : False, 
    "include_one_hot_encoding_of_y_label" : False
    }


# random: predicts a random value for each bottleneck in y_pred
# naiveM2: predicts always the second machine for y_pred
# naiveM4: predicts always the fourth machine for y_pred
# last: predicts always the last bottleneck from x_test for y_pred

# get empty list to save benchmarking results 
results_benchmarking = []
# get a new benchmark and load data 
benchmark = BottleneckBenchmark(params)
benchmark.run()
# iterate all benchmarkings and save them to list 
for how in ["random", "naiveM2", "naiveM4", "last"]:
    print(f"Now running '{how}' benchmark.")
    results_benchmarking.append(benchmark.get_benchmarks(how=how))

# save the results from the benchmarking for later plotting
save_data(results_benchmarking, 'results_benchmarking')