# Adam Wilson
# CS4342
# Project 2
# 5/1/2020

import random
import math

training_set = [
    {"i": 1, "A": 0, "B": 0, "C": 0, "Class": "+"},
    {"i": 2, "A": 0, "B": 0, "C": 1, "Class": "+"},
    {"i": 3, "A": 0, "B": 1, "C": 0, "Class": "+"},
    {"i": 4, "A": 0, "B": 1, "C": 1, "Class": "-"},
    {"i": 5, "A": 1, "B": 0, "C": 0, "Class": "+"},
    {"i": 6, "A": 1, "B": 0, "C": 0, "Class": "+"},
    {"i": 7, "A": 1, "B": 1, "C": 0, "Class": "-"},
    {"i": 8, "A": 1, "B": 0, "C": 1, "Class": "+"},
    {"i": 9, "A": 1, "B": 1, "C": 0, "Class": "-"},
    {"i": 10, "A": 1, "B": 1, "C": 0, "Class": "-"}
]

validation_set = [
    {"i": 11, "A": 0, "B": 0, "C": 0, "Class": "+"},
    {"i": 12, "A": 0, "B": 1, "C": 1, "Class": "+"},
    {"i": 13, "A": 1, "B": 1, "C": 0, "Class": "+"},
    {"i": 14, "A": 1, "B": 0, "C": 1, "Class": "-"},
    {"i": 15, "A": 1, "B": 0, "C": 0, "Class": "+"}
]

attributes = ("A", "B", "C")


# builds decision tree from input records
class DecisionTreeNode:
    # initializes node with its parent, a list of records, and a
    # set of d attributes, then decides whether to split
    def __init__(self, records, d, parent):
        self.zero = None
        self.one = None
        self.parent = parent
        self.records = records
        self.d = d
        self.split = None
        self.label = None

        self.split_check()

    # decides whether to split a node or label it and leave as a leaf
    def split_check(self):
        if self.records:
            # if all records at node are of same class, node becomes a leaf
            # labelled with that class
            same = True
            for record in self.records:
                if record["Class"] != self.records[0]["Class"]:
                    same = False
            if same:
                self.label = self.records[0]["Class"]

            # if there are no attributes remaining to split the node, it becomes
            # a leaf whose label is the majority class of its records
            elif self.d == ():
                self.label = majority(self.records)

            # otherwise the node is split by whichever attribute yields the highest
            # info gain: any records at this node with a 0-value of said attribute
            # are sent to one child, while the other child contains those with a 1-value
            else:
                self.split = best_split(self.records, self.d)
                if self.split is None:
                    self.label = majority(self.records)
                    return
                list(self.d).remove(self.split)
                zero, one = [], []
                for record in self.records:
                    if record[self.split] == 0:
                        zero.append(record)
                    else:
                        one.append(record)
                self.zero = DecisionTreeNode(zero, self.d, self)
                self.one = DecisionTreeNode(one, self.d, self)

        else:
            self.label = self.parent.label

    # graphically displays decision tree
    def display(self):
        lines, _, _, _ = self.display_aux()
        for line in lines:
            print(line)

    # Returns list of strings, width, height, and horizontal coordinate of the root.
    def display_aux(self):
        # No child.
        if self.one is None and self.zero is None:
            line = "{}({}) ".format(self.label, len(self.records))
            return [line], len(line), 1, 0

        # Two children.
        left, n, p, x = self.zero.display_aux()
        right, m, q, y = self.one.display_aux()
        s = '%s' % self.split
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    # inserts test instance into tree until reaching a leaf, then returns the leaf's label
    def classify_test_instance(self, test_instance):
        if self.split is None:
            return self.label
        else:
            if test_instance[self.split] == 0:
                return self.zero.classify_test_instance(test_instance)
            else:
                return self.one.classify_test_instance(test_instance)


# finds the info gain for each of the remaining attributes on which the node could potentially
# be split, returning the attribute with the highest info gain or None if there is no attribute
# which yields a positive info gain
def best_split(records, d):
    pos, info_gain = 0, 0
    n = len(records)
    best = None
    for record in records:
        if record["Class"] == "+":
            pos += 1
    neg = n - pos
    # calculates entropy of current node
    entropy = -pos / n * math.log(pos / n, 2) - neg / n * math.log(neg / n, 2)

    for attribute in d:
        neg_0, pos_0, neg_1, pos_1 = 0, 0, 0, 0
        # counts impurity of potential children
        for record in records:
            if record[attribute] == 0 and record["Class"] == "+":
                pos_0 += 1
            elif record[attribute] == 0 and record["Class"] == "-":
                neg_0 += 1
            elif record[attribute] == 1 and record["Class"] == "+":
                pos_1 += 1
            else:
                neg_1 += 1

        # calculates weighted entropy of children and corresponding info gain, storing best info gain
        children_entropy = weighted_entropy(pos_1, neg_1, n) + weighted_entropy(pos_0, neg_0, n)
        if entropy - children_entropy > info_gain:
            info_gain = entropy - children_entropy
            best = attribute

    return best


# calculates weighted entropy of a child
def weighted_entropy(pos, neg, n):
    n1 = pos + neg
    if pos == 0 or neg == 0:
        return 0
    else:
        return n1 / n * -pos / n1 * math.log(pos / n1, 2) - neg / n1 * math.log(neg / n1, 2)


# returns majority class of a group of records, or a random class if there is a tie
def majority(records):
    pos, neg = 0, 0
    for record in records:
        if record["Class"] == "+":
            pos += 1
        else:
            neg += 1
    if pos > neg:
        return "+"
    elif pos < neg:
        return "-"
    else:
        return random.choice(["+", "-"])


# creates and displays base classifier from training set
print("Decision tree without bagging:\n")
root = DecisionTreeNode(training_set, attributes, None)
root.display()

# takes 10 samples of 10 (with replacement) from training set, creating and displaying
# a decision tree for each sample, then adding each tree to the ensemble model
ensemble = []
for i in range(10):
    bootstrap_sample = []
    for j in range(len(training_set)):
        bootstrap_sample.append(random.choice(training_set))

    print("\n\nBagging Round {}\nSample:\n".format(i + 1))
    for data in bootstrap_sample:
        print(data)

    print("\nDecision Tree:\n")
    root = DecisionTreeNode(bootstrap_sample, attributes, None)
    root.display()
    ensemble.append(root)

# tests each instance in validation set on each tree in the ensemble, uses the results to take
# a majority vote for each instance, then displays the predicted vs. actual classes
print("\n\nTesting validation set:")
for instance in validation_set:
    results = []
    for tree in ensemble:
        results.append({"Class": tree.classify_test_instance(instance)})
    print("\nTest instance {}\t\t Predicted Class: {}\t\t Actual Class: {}"
          .format(instance["i"] - 10, majority(results), instance["Class"]))
