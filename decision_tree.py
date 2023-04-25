import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(self, X: pd.DataFrame,
                 y: list,
                 min_samples_split=20,
                 max_depth=5,
                 depth=0,
                 node_type="root",
                 rule=""
                ):
        # saving data
        self.X = X
        self.y = y
        
        # saving hyperparameters
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        # default current depth of the node
        self.depth = depth
        
        # extracting all the features
        self.features = list(self.X.columns)
        
        # type of node
        self.node_type = node_type
        
        # rule for spliting
        self.rule = rule
        
        # computing counts of y in the node
        self.counts = Counter(y)
        
        # getting the GINI impurity based on the y distribution
        self.gini_impurity = self.get_gini()
        
        # sorting the counts and saving the final prediction of the node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))
        
        # getting the last item
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        # saving to object attribute, this node will predict the class with the most frequent class
        self.yhat = yhat 

        # saving the number of observations in the node 
        self.n = len(y)

        # initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # default values for splits
        self.best_feature = None 
        self.best_value = None 
    
    @staticmethod
    def GINI_impurity(arr: np.array) -> float:
        n = np.sum(arr)
        gini = 1 - np.sum((arr / n) ** 2)
        return gini
    
    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list. 
        """
        return np.convolve(x, np.ones(window), 'valid') / window
    
    def get_gini(self):
        """
        Function to calculate the GINI impurity of a node 
        """
        counts = np.array(list(self.counts.values()))
        return self.GINI_impurity(counts)
    
    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df["Y"] = self.y
        
        # Getting the GINI impurity for the base input 
        gini_base = self.get_gini()
        
        # Finding which split yields the best GINI gain 
        max_gain = 0
        
        # Default best feature and split
        best_feature = None
        best_value = None
        
        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)
            
            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)
            
            for value in xmeans:
                # Spliting the dataset 
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])
                
                # Getting the Y distribution from the dicts
                left_dists = list(sorted(left_counts.items(), key=lambda item: item[0]))
                left_dists = np.array([tup[1] for tup in left_dists])
                right_dists = list(sorted(right_counts.items(), key=lambda item: item[0]))
                right_dists = np.array([tup[1] for tup in right_dists])
                
                # Getting the left and right gini impurities
                gini_left = self.GINI_impurity(left_dists)
                gini_right = self.GINI_impurity(right_dists)
                
                # Getting the obs count from the left and the right data splits
                n_left = np.sum(left_dists)
                n_right = np.sum(right_dists)
                
                # Calculating the weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)
                
                # Calculating the weighted GINI impurity
                wGINI = w_left * gini_left + w_right * gini_right
                
                # Calculating the GINI gain 
                GINIgain = gini_base - wGINI
                
                # Checking if this is the best split so far 
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value
                    
                    # Setting the best gain to the current one 
                    max_gain = GINIgain
            
        return (best_feature, best_value)
    
    def build_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data 
        df = self.X.copy()
        df['Y'] = self.y
        
        # If there is GINI to be gained, we split further 
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
            # Getting the best split 
            best_feature, best_value = self.best_split()
            
            if best_feature is not None:
                # saving the best split to curr node
                self.best_feature = best_feature
                self.best_value = best_value
                
                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()
                
                # creating left and right nodes
                left = Node( 
                    left_df[self.features],
                    left_df['Y'].values.tolist(),
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                    )
                
                self.left = left 
                self.left.build_tree()

                right = Node(
                    right_df[self.features],
                    right_df['Y'].values.tolist(), 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                    )

                self.right = right
                self.right.build_tree()
                
    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []
        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
            predictions.append(self.predict_obs(values))     
        predictions = np.array(predictions)
        
        return predictions
    
    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value
            
            if cur_node.n < cur_node.min_samples_split:
                break
                
            if values.get(best_feature) < best_value:
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
                         
        return cur_node.yhat