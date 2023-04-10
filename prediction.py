# project 
from auxiliaries import get_buffer, get_active_periods

# other 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from datetime import datetime as dt 

from typing import Tuple
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


# parameter selected according to the simulation scenario
SIM_LENGTH = 10000 
SIM_STATIONS = ["B1", "B2", "B3", "B4"]
SIM_SCENARIO = "10k_S2-S4+25%"


class BottleneckPrediction():

    def __init__(
        self,
        mdl_scenario : str, 
        n_steps_out : int,
        n_steps_in : int,
        aggreg_lvl : int,
        stride_lvl : int,
        range_train : range,
        range_test : range, 
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
        # custom name to store the best model 
        self.mdl_scenario = mdl_scenario

    def run(self): 

        # data preparation
        self.x_train, self.y_train = self.prepare_data(file_range=self.range_train)
        self.x_test, self.y_test = self.prepare_data(file_range=self.range_test)
        self.describe_data()

        # modelling
        # TODO: add timer 
        self.model = self.get_model()
        self.describe_model()

        # training
        now = dt.now()
        self.model_hist = self.fit_model()
        self.time_training = dt.now() - now
        print(f"training complete in: {self.time_training}")
        self.describe_hist()

        # evaluation 
        self.best_model = load_model(f'models/{self.mdl_scenario}.hdf5') 
        self.y_pred = self.best_model.predict(self.x_test)
        self.describe_pred()

        # 
        self.plot_some_predictions(stride=25)

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
        df_buffer = get_buffer(f"{path}buffer{scenario_name}.csv")
        df_actper = get_active_periods(f"{path}active_periods{scenario_name}.csv")

        # aggregate buffer to every n observations
        df_buffer = df_buffer.groupby(np.arange(len(df_buffer))//self.aggreg_lvl).mean() # aggregate

        # add final bottleneck states "0" and split before aggregation

        num_of_splits = int(SIM_LENGTH/self.aggreg_lvl)
        ar_actper = np.split(np.concatenate((df_actper["bottleneck"], ["S0"]*(SIM_LENGTH - len(df_actper)))), num_of_splits, axis=0)

        # aggregate by group and get most frequent bottleneck state 
        ar_actper = [self._most_common(x.tolist()) for x in ar_actper]

        # merge both dfs and convert "bottleneck" all columns to float
        df = df_buffer
        df["bottleneck"] = [v[1:] for v in ar_actper]
        df =  df.astype(float)

        # select and return only the relevant columns
        df = df.loc[:, SIM_STATIONS + ["bottleneck"]] 
        # omitting "B0" for it is almost constant (close to max buffer)
        # omitting "B_last" for it increased indefinitely
        return df[df.columns].to_numpy() 


    def prepare_data(
        self, 
        file_range : range
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
        num_attr_for_x = len(SIM_STATIONS) + 1 if self.ibitd else len(SIM_STATIONS)
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
        
        if self.iohey: # include_one_hot_encoding_of_y_label 
            y_train = self._to_one_hot(y_train)

        return x_train, y_train


    def get_model(self) -> Sequential:
        '''Returns a custom LSTM with a single hidden layer.'''
                
        # get custom optimizer with gardient clipping
        self.optimizer = Adam(
            learning_rate = 0.001,
            beta_1= 0.9,
            beta_2 = 0.999,
            epsilon = 1e-7,
            amsgrad = False,
            clipvalue=1.0)

        # get the number of features from the training data 
        n_features = self.x_train.shape[2]*len(SIM_STATIONS+1) if self.iohey else self.x_train.shape[2]

        # define the model
        model = Sequential()
        # input layer 
        model.add(
            LSTM(
                units=256, 
                activation='relu', 
                return_sequences=True, 
                input_shape=(self.n_steps_in, n_features)
                )
            )
        # hidden layer 
        model.add(
            LSTM(
                units=256, 
                activation='relu'
                )
            )
        # output layer
        model.add(
            Dense(
                units = self.n_steps_out,
                activation = None,
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
            patience=30,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0
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
            batch_size = 64, 
            callbacks = [self.cp_stopper, self.cp_saver]) 


    def describe(self):
        # TODO: add print() to describe BottleneckPrediction
        pass


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
        plt.title("loss")
        plt.ylim([0,2.5])
        plt.show()


    def describe_pred(self) -> None: 

        # eval (no one-hot-enc)
        assert self.y_test.shape == self.y_pred.shape

        # check if the prediction was correct 
        eval_dict = {}
        for i in range(self.y_test.shape[0]): 
            y_t = self.y_test[i]
            p_t = self.y_pred[i].astype(int)
            res = [y==p for y, p in zip(y_t, p_t)]
            eval_dict[i] = res

        # count number of correct predictions (by prediction horizon)
        eval_list = [0] * self.y_test.shape[1]
        for val in eval_dict.values():
            _add = [1 if v else 0 for v in val]
            eval_list = [a+b for a, b in zip(eval_list, _add )]

        # 
        eval_list = [e/self.y_test.shape[0] for e in eval_list]

        # plot the evaluation 
        plt.plot(eval_list)
        plt.ylim(0, 1)
        plt.xlim(0, 25)


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
        data = data.astype(int)
        # get dimensions
        nb_inputs = data.shape[0] 
        nb_outputs = data.shape[1] # before one-hot-encoding
        # reshape and return 
        data = data.reshape([nb_inputs, int(nb_outputs/nb_classes), nb_classes])
        return np.array([d.argmax(axis=1).tolist() for d in data])
