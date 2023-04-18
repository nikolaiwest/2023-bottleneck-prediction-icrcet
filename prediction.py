# project 
from auxiliaries import get_buffer, get_active_periods, load_data, save_data

# other 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from datetime import datetime as dt 

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# parameter selected according to the simulation scenario
SIM_LENGTH = 10000 
SIM_STATIONS = ["B1", "B2", "B3", "B4"] # stations under consideration
SIM_SCENARIO = "10k_S2-S4+25%"


class ResultEvaluation():

    def __init__(self, name:str, y_test, y_pred, evaluation):
        self.name = name
        self.y_test = y_test
        self.y_pred = y_pred
        self.eval = evaluation

    def get_name(self) -> str:
        return self.name

    def get_prediction(self) -> np.ndarray:
        return self.y_pred
    
    def get_test_data(self) -> np.ndarray:
        return self.y_test
    
    def get_evaluation(self) -> list:
        return self.eval


class BottleneckPrediction():

    def __init__(
        self,
        mdl_scenario : str, 
        n_steps_out : int,
        n_steps_in : int,
        aggreg_lvl : int,
        stride_lvl : int,
        range_train : list,
        range_test : list, 
        training_epochs : int, 
        include_bottleneck_in_training_data : bool, 
        include_one_hot_encoding_of_y_label : bool, 
        ): 
        # shape of the data 
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.aggreg_lvl = aggreg_lvl
        self.stride_lvl = stride_lvl

        # experiments to include 
        self.range_train = range_train
        self.range_test = range_test

        # training duration (check val loss to see overfitting)
        self.training_epochs = training_epochs

        # modelling variants 
        self.ibitd = include_bottleneck_in_training_data
        self.iohey = include_one_hot_encoding_of_y_label

        # custom name of the simulation scenario
        self.sim_scenario = SIM_SCENARIO
        # custom list of stations
        self.sim_stations = SIM_STATIONS
        self.num_stations = len(self.sim_stations)
        # custom name to store the best model 
        self.mdl_scenario = mdl_scenario

    def run(self, verbose=False) -> ResultEvaluation: 

        # check if data can be loaded from drive
        if not os.path.exists(f"data_prepared/data_{self.mdl_scenario}.pkl"):
            # run data preparation
            self.x_train, self.y_train = self.prepare_data(file_range=self.range_train)
            self.x_test, self.y_test = self.prepare_data(file_range=self.range_test)
            # save data
            save_data((self.x_train, self.y_train, self.x_test, self.y_test), f"data_prepared/data_{self.mdl_scenario}")
        else: # load
            self.x_train, self.y_train, self.x_test, self.y_test = load_data(path="data_prepared/", name=f"data_{self.mdl_scenario}")
            if verbose: 
                print(f"{self.mdl_scenario}: Data loaded from drive")
        if verbose:
            self.describe_data()

        # modelling
        self.model = self.get_model()
        if verbose: 
            self.describe_model()

        # check if model weights are available from previous training
        if not os.path.exists(f"models/{self.mdl_scenario}.hdf5"):
            
            # start training
            now = dt.now()
            self.model_hist = self.fit_model()
            self.time_training = dt.now() - now
            print(f"training complete in: {self.time_training}")
            if verbose:
                self.describe_hist()
        
        else:
            if verbose: 
                print(f"{self.mdl_scenario}: Model loading from drive")

        # evaluation 
        self.best_model = load_model(f'models/{self.mdl_scenario}.hdf5') 
        self.y_pred = self.best_model.predict(self.x_test)
        self.eval = self.describe_pred(verbose=verbose)

        # 
        if verbose: 
            self.plot_some_predictions(stride=int(len(self.y_pred)/10))

        return ResultEvaluation(
            name = self.mdl_scenario,
            y_test = self.y_test,
            y_pred = self.y_pred,
            evaluation = self.eval,
        )

    def load_sim_data(
        self,
        path: str, 
        scenario_name: str, 
    ) -> pd.DataFrame: 
        ''' Import and aggregation the prepared simulation data. 

        Parameters
        ----------
        path : str
            Folder path to the simulation data (buffer level and active periods)

        s_name : str
            Scenario name (currently _0 to _999 simulations available)

        Return
        -------
            Returns a single  dataframe with buffer levels and an attribute for the 
            bottleneck station. Due to the unlimited system boundaries in the 
            simulation, the first and the last buffer are omitted. The first buffer 
            is always approx. at max capacity while the last buffer will fill up to 
            an unlimited capacity.'''
        
        # import from raw simulation data from path using the scenario name
        data_buffer = get_buffer(f"{path}buffer{scenario_name}.csv")
        data_actper = get_active_periods(f"{path}active_periods{scenario_name}.csv")

        # get the number of observations after aggregation
        num_of_splits = int(SIM_LENGTH/self.aggreg_lvl)

        # make data_buffer same length with simulation length 
        if len(data_buffer)!=SIM_LENGTH: 
            # shorten df (just as backup)
            if len(data_buffer) > SIM_LENGTH:
                data_buffer = data_buffer[:SIM_LENGTH]
            else: 
                # fill df with last rows
                for i in range(len(data_buffer), SIM_LENGTH):
                    # copy last row and append 
                    data_buffer.loc[i] = data_buffer.loc[i-1]
                    # increase "t"
                    data_buffer.loc[i][0] = data_buffer.loc[i-1][0]+1

        # make df_actper same length with simulation length 
        if len(data_actper)!=SIM_LENGTH: 
            # shorten df (again, just a fallback)
            if len(data_actper) > SIM_LENGTH: 
                data_actper = data_actper[:SIM_LENGTH]
            else: 
                # fill df with last rows
                for i in range(len(data_actper), SIM_LENGTH):
                    # copy last row and append 
                    data_actper.loc[i] = data_actper.loc[i-1]

        # aggregate buffer to every n observation
        data_buffer = data_buffer.groupby(np.arange(len(data_buffer))//self.aggreg_lvl).mean()
        data_buffer = data_buffer[:num_of_splits]

        # aggregate bottleneck info for every n observation
        assert len(data_buffer)==num_of_splits, "Aggregation will result in unequal lengths"
        data_actper = data_actper["bottleneck"][:self.aggreg_lvl*num_of_splits]
        data_actper = np.split(data_actper, num_of_splits, axis=0)
        data_actper = [self._most_common(x.tolist()) for x in data_actper]

        # merge both dfs and convert "bottleneck" all columns to float
        data_return = data_buffer
        data_return["bottleneck"] = [v[1:] for v in data_actper]
        data_return =  data_return.astype(float)

        # select and return only the relevant columns
        data_return = data_return.loc[:, self.sim_stations + ["bottleneck"]] 
        # omitting "B0" for it is almost constant (close to max buffer)
        # omitting "B_last" for it increased indefinitely
        data_return =  data_return[data_return.columns].to_numpy() 
        return data_return

    def prepare_data(
        self, 
        file_range : list
    ):
        '''Load and prepare data from multiple simulations for training and testing.

        Parameters
        ----------
        file_range : range
            Range object to determine the simulations to include in the dataset
        
        Return
        -------
            Returns a tuple of x and y data. While the x values include the buffer 
            levels, the y values include the designated bottleneck station (by number).
        '''

        # get number of attributes in the x values 
        num_attr_for_x = self.num_stations+1 if self.ibitd else self.num_stations
        # get empty ndarrays for X and y 
        x = np.empty([0,self.n_steps_in, num_attr_for_x])
        y = np.empty([0,self.n_steps_out])

        # iter over training range and add scenario data to X and y 
        for i in file_range: 
            # load new scenario from path
            s_i = self.load_sim_data(
                path = "data/",
                scenario_name = f"_{self.sim_scenario}_{self._to_str(i)}",
                )
            # split according to parameter selection
            x_i, y_i = self._split_sequences(s_i)

            # add to result arrays
            x = np.concatenate((x, x_i))
            y = np.concatenate((y, y_i))

        # shuffle all observations 
        x_train, y_train = self._to_unison_shuffled_copies(x, y)
        
        if self.iohey: 
            # include_one_hot_encoding_of_y_label 
            y_train = self._to_one_hot(y_train)

        return x_train, y_train


    def get_model(self) -> Sequential:
        '''Returns a custom LSTM with a single hidden layer.'''
                
        # get custom optimizer with gardient clipping
        self.optimizer = Adam(
            learning_rate = 0.0001,
            beta_1= 0.9,
            beta_2 = 0.999,
            epsilon = 1e-7,
            amsgrad = False,
            clipvalue=1.0)

        # get the number of features from the training data 
        n_features = self.x_train.shape[2]

        # update size of output layer
        self.n_steps_out = self.n_steps_out*(self.num_stations+1)

        # define the model
        model = Sequential()
        # input layer 
        model.add(
            LSTM(
                units=64, 
                activation='tanh', 
                return_sequences=True, 
                input_shape=(self.n_steps_in, n_features),
                recurrent_dropout=0
                )
            )
        # hidden layer 
        model.add(
            LSTM(
                units=64, 
                activation='tanh'
                )
            )
        # output layer
        model.add(
            Dense(
                units = self.n_steps_out,
                use_bias = True,
                kernel_initializer = "glorot_uniform",
                bias_initializer = "zeros",
                )
            )
        # compile and return 
        model.compile(
            optimizer=self.optimizer, 
            loss="mse")
        return model


    def fit_model(self):

        # get callback for early stopping
        self.cp_stopper = EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=50,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True,
            #start_from_epoch=0
        )

        # get callback to save models
        self.cp_saver = ModelCheckpoint(
            filepath=f'models/{self.mdl_scenario}.hdf5', 
            save_best_only=True, 
            monitor='loss', 
            mode='min')
        
        # fit model and return modelling history 
        return self.model.fit(
            x = self.x_train, 
            y = self.y_train, 
            validation_data = (self.x_test, self.y_test),
            epochs = self.training_epochs, 
            verbose = 1, 
            shuffle = True,
            batch_size = 256, 
            callbacks = [self.cp_stopper, self.cp_saver]) 


    def describe_data(self) -> None:
        # describe data
        print(f"X_train: {self.x_train.shape}")
        print(f"y_train: {self.y_train.shape}")
        print(f"X_test: {self.x_test.shape}")
        print(f"y_test: {self.y_test.shape}")


    def describe_model(self) -> None:
        # describe model 
        print(self.model.summary())


    def describe_hist(self) -> None: 
        # describe training history 
        plt.plot(self.model_hist.history['loss'], label="loss")
        plt.plot(self.model_hist.history['val_loss'], label="val_loss")
        plt.title(f"{self.mdl_scenario}_loss")
        plt.ylim([0,1.5])
        plt.savefig(f"training/{self.mdl_scenario}_loss.png", format="png")
        plt.show()


    def describe_pred(self, verbose:bool) -> list: 

        # revert one hot encoding
        if self.iohey:
            self.y_test = self._reverse_one_hot(self.y_test)
            self.y_pred = self._reverse_one_hot(self.y_pred)

        # eval (no one-hot-enc)
        # assert self.y_test.shape == self.y_pred.shape

        # check if the prediction was correct 
        eval_dict = {}
        for i in range(self.y_test.shape[0]): 
            y_t = self.y_test[i]
            p_t = np.rint(self.y_pred[i])
            res = [y==p for y, p in zip(y_t, p_t)]
            eval_dict[i] = res

        # count number of correct predictions (by prediction horizon)
        eval_list = [0] * self.y_test.shape[1]
        for val in eval_dict.values():
            _add = [1 if v else 0 for v in val]
            eval_list = [a+b for a, b in zip(eval_list, _add )]

        # evaluation list 
        eval_list = [e/self.y_test.shape[0] for e in eval_list]

        # make plot of evaluation
        if verbose: 
            # plot the evaluation 
            plt.plot(eval_list)
            plt.title(self.mdl_scenario)
            plt.ylim(0, 1)
            #plt.xlim(0, 25)
            plt.savefig(f"training/{self.mdl_scenario}_eval.png", format="png")
            plt.show()

        # return evaluation as list 
        return eval_list


    def plot_some_predictions(self, stride): 

        for t in range(0, len(self.x_test), stride):

            fig, axes = plt.subplots(1,2)

            # plot buffer before prediction
            axes[0].plot(self.x_test[t])

            # plot prediction and y_test
            axes[1].plot(self.y_pred[t], label="prediction", color="red", linestyle=":", linewidth=2)
            axes[1].plot(self.y_test[t], label="truth", color="green")

            # formating
            axes[1].set_ylim([0,5])
            axes[1].legend()
            fig.suptitle(t)
            plt.show()


    def _most_common(self, lst: list) -> object:
        return max(set(lst), key=lst.count)


    def _split_sequences(
            self,
            sequences,
            ):
        
        # create two empty result lists
        X, y = list(), list()

        # iter over the given sequence 
        for i in range(0, len(sequences), self.stride_lvl):
            # find the end of this pattern
            end_ix = i + self.n_steps_in
            out_end_ix = end_ix + self.n_steps_out-1
        
            # check if we are beyond the seq length
            if out_end_ix > len(sequences):
                break
        
            # gather input and output parts of the pattern
            if self.ibitd: # include_bottleneck_in_training_data
                seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1:out_end_ix, -1]
            else: # exclude last column ('bottleneck') in training data 
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]

            # add to predefined lists 
            X.append(seq_x)
            y.append(seq_y)
        
        # convert lists to array and return
        return np.array(X), np.array(y)


    def _to_unison_shuffled_copies(self,a, b):
        # check for lengths
        assert len(a) == len(b)
        # shuffle and return
        np.random.seed(42) # defined, to repeat the experiment
        p = np.random.permutation(len(a))
        return a[p], b[p]


    def _to_str(self, i: int) -> str: 
        '''Simple aux. function to load files.'''
        if i < 10:
            return f"00{i}"
        elif i < 100:
            return f"0{i}"
        else:
            return f"{i}"


    def _to_one_hot(self, data, nb_classes=5):
        # convert y to int
        data = data.astype(int)
        # get dimensions
        nb_inputs = data.shape[0] 
        nb_outputs = data.shape[1] # before one-hot-encoding
        # one hot encoding for each predicted station
        res = np.eye(nb_classes)[np.array(data).reshape(-1)]
        # reshape and return 
        return res.reshape([nb_inputs, nb_outputs*nb_classes])


    def _reverse_one_hot(self, data, nb_classes=5):
        # convert y to int
        data = np.rint(data)
        # get dimensions
        nb_inputs = data.shape[0] 
        nb_outputs = data.shape[1] # before one-hot-encoding
        # reshape and return 
        data = data.reshape([nb_inputs, int(nb_outputs/nb_classes), nb_classes])
        return np.array([d.argmax(axis=1).tolist() for d in data])


class BottleneckBenchmark(BottleneckPrediction):
    
    def __init__(self, params: dict):
        super().__init__(**params)

    def run(self, verbose=False) -> ResultEvaluation:  

        # check if data can be loaded from drive
        if not os.path.exists(f"data_prepared/data_benchmarking.pkl"):
            # run data preparation
            self.x_train, self.y_train = self.prepare_data(file_range=self.range_train)
            self.x_test, self.y_test = self.prepare_data(file_range=self.range_test)
            # save data
            save_data((self.x_train, self.y_train, self.x_test, self.y_test), "data_prepared/data_benchmarking")
        else: # load
            self.x_train, self.y_train, self.x_test, self.y_test = load_data(path="data_prepared/", name=f"data_benchmarking")
            if verbose:
                print(f"Benchmarking: Data loaded from drive")
        
        if verbose:
            self.describe_data()

        # ensure one hot encoding is disabled (not required for benchmarking)
        self.iohey = False


    def get_benchmarks(self, how: str ) -> ResultEvaluation: 
        # predict the last bottleneck from the training data for y_test 
        if how == "last":
            y_pred = []
            for i in range(self.x_test.shape[0]):
                # get list of the last value of y_test
                y = self.x_test[i][-1][-1] 
                y_pred.append([y] * self.n_steps_out)
            self.y_pred=np.array(y_pred)

        # predict a random value for the next bottleneck 
        elif how =="random":
            y_pred = []
            for i in range(self.x_test.shape[0]):
                # get list of the random values for y_pred
                y = list(np.random.randint(low = self.num_stations+1, size=self.n_steps_out))
                y_pred.append(y)
            self.y_pred=np.array(y_pred)

        # predict only the second station as bottleneck 
        elif how =="naiveM2":
            y_pred = []
            for i in range(self.x_test.shape[0]):
                # get 'n_steps_out' values of [1] for y_pred (corresponds to M2)
                y = [1] * self.n_steps_out
                y_pred.append(y)
            self.y_pred=np.array(y_pred)

        # predict only the fourth station as bottleneck 
        elif how =="naiveM4":
            y_pred = []
            for i in range(self.x_test.shape[0]):
                # get 'n_steps_out' values of [23] for y_pred (corresponds to M4)
                y = [3] * self.n_steps_out
                y_pred.append(y)
            self.y_pred=np.array(y_pred)
        
        else:
            raise ValueError(f"{how} is no valid benchmark.")
        
        # get evaluation for benchmark
        self.eval = self.describe_pred(verbose=False)

        return ResultEvaluation(
            name = f"benchmark_{how}",
            y_test = self.y_test,
            y_pred = self.y_pred,
            evaluation = self.eval,
        )