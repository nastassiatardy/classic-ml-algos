from collections import Counter
import numpy as np

class TreeNode:
  def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain):
    self.data = data
    self.feature_idx = feature_idx
    self.feature_val = feature_val
    self.prediction_probs = prediction_probs
    self.information_gain = information_gain
    self.left = None
    self.right = None

  def node_def(self):
    if self.left or self.right:
      return f"NODE | Information Gain = {self.information_gain} | Split IF X[{self.feature_idx}] < {self.feature_val} THEN left O/W right"
    else:
      unique_values, value_counts = np.unique(self.data[:,-1], return_counts=True)
      output = ", ".join([f"{value}->{count}" for value, count in zip(unique_values, value_counts)])
      return f"LEAF | Label Counts = {output} | Pred Probs = {self.prediction_probs}"


class DecisionTree:
  """
  Implementation with criterion = 'entropy'
  """
  def __init__(
      self,
      max_depth=4,
      min_samples_leaf=1,
      min_information_gain=0.0
      ):
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf
    self.min_information_gain = min_information_gain

  def entropy(self, class_probabilities):
    # Low entropy: the splitted data group have a dominant class (i.e. good split)
    # Otherwise high entropy
    # => we search for splits that have the lowest entropy
    return sum([-p * np.log2(p) for p in class_probabilities if p>0])

  def class_probabilities(self, labels):
    total_count = len(labels)
    return [label_count / total_count for label_count in Counter(labels).values()]

  def data_entropy(self, labels):
    return self.entropy(self.class_probabilities(labels))

  def partition_entropy(self, subsets):
    total_count = sum([len(subset) for subset in subsets])
    return sum([
        self.data_entropy(subset) * (len(subset) / total_count)
        for subset in subsets])

  def split(self, data, feature_idx, feature_val):
    mask = data[:,feature_idx] <= feature_val
    group1 = data[mask]
    group2 = data[~mask]
    return group1, group2

  def find_best_split(self, data):
    min_part_entropy = 1e9

    for idx in range(data.shape[-1] - 1):
      # feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25))
      #       for feature_val in feature_vals:
      feature_val = np.median(data[:,idx])
      X1, X2 = self.split(data, idx, feature_val)
      current_entropy = self.partition_entropy([X1[:,-1], X2[:,-1]])
      if current_entropy < min_part_entropy:
        min_part_entropy = current_entropy
        min_entropy_feature_idx, min_entropy_feature_val = idx, feature_val
        X1_min, X2_min = X1, X2
    return (
        X1_min, X2_min,
        min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy
        )

  def find_label_probs(self, data):
    labels_as_integers = data[:,-1].astype(int)
    total_labels = len(labels_as_integers)

    label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

    for i, label in enumerate(self.labels_in_train):
      label_index = np.where(labels_as_integers == i)[0]
      if len(label_index) > 0:
        label_probabilities[i] = len(label_index) / total_labels

    return label_probabilities


  def create_tree(self, data, current_depth):
    if current_depth > self.max_depth: # Check if max_depth is reached (stopping criteria)
      return None

    # Find best split
    X1, X2, split_feature_idx, split_feature_val, split_entropy = self.find_best_split(data)

    # Find label probs for the node
    label_probabilities = self.find_label_probs(data)

    # Compute information gain
    node_entropy = self.entropy(label_probabilities)
    information_gain = node_entropy - split_entropy

    # Create node
    node = TreeNode(
        data, split_feature_idx, split_feature_val,
        label_probabilities, information_gain
        )

    # Check if min_samples_leaf has been satisfied (stopping criteria)
    if self.min_samples_leaf > X1.shape[0] or self.min_samples_leaf > X2.shape[0]:
      return node
    # Check if the min_information_gain has been satisfied (stopping criteria)
    elif information_gain < self.min_information_gain:
      return node

    node.left = self.create_tree(X1, current_depth + 1)
    node.right = self.create_tree(X2, current_depth + 1)

    return node

  def train(self, X_train, Y_train):
    # Concat features and labels
    self.labels_in_train = np.unique(Y_train)
    train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

    # Start creating the tree
    self.tree = self.create_tree(train_data, current_depth=0)


  def predict_one_sample(self, X):
    """Returns prediction for 1 dim array"""
    node = self.tree

    while node:
      pred_probs = node.prediction_probs
      if X[node.feature_idx] <= node.feature_val:
        node = node.left
      else:
        node = node.right
    return pred_probs

  def predict_proba(self, X_set):
    """Returns the predicted probs for a given dataset"""
    pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
    return pred_probs

  def predict(self, X_set):
    """Returns the predicted class for a given dataset"""
    pred_probs = self.predict_proba(X_set)
    preds = np.argmax(pred_probs, axis=1)
    return preds

  def print_recursive(self, node, level=0):
    if node is not None:
      self.print_recursive(node.left, level + 1)
      print('    ' * 4 * level + '-> ' + node.node_def())
      self.print_recursive(node.right, level + 1)

  def print_tree(self):
    self.print_recursive(node=self.tree)
