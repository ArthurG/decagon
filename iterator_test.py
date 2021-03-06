#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

from polypharmacy.utility import *

# Train on CPU (hide GPU) due to memory constraints
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################

def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc

def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders
# In[2]:


###########################################################
#
# Load and preprocess data (This is a dummy toy example!)
#
###########################################################

####
# The following code uses artificially generated and very small networks.
# Expect less than excellent performance as these random networks do not have any interesting structure.
# The purpose of main.py is to show how to use the code!
#
# All preprocessed datasets used in the drug combination study are at: http://snap.stanford.edu/decagon:
# (1) Download datasets from http://snap.stanford.edu/decagon to your local machine.
# (2) Replace dummy toy datasets used here with the actual datasets you just downloaded.
# (3) Train & test the model.
####

val_test_size = 0.10


gene_net, node2idx = load_ppi_v2("data/bio-decagon-ppi.csv") # protein protein interactions
stitch2proteins = load_targets("data/bio-decagon-targets.csv") #drug protein interations
combo2stitch, combo2se, se2name = load_combo_se('data/bio-decagon-combo.csv') #Drug drug interactions


# ## Clean drug data

# In[3]:


from collections import Counter

def get_se_counter(se_map):
    side_effects = []
    for drug in se_map:
        side_effects += list(set(se_map[drug]))
    return Counter(side_effects)

combo_counter = get_se_counter(combo2se)

common_se = []
for se, count in combo_counter.most_common(964):
    common_se += [se]
common_se = set(common_se)


# In[4]:


# Do some pre-processing of drug data

#count up all the unique drugs, and give them a unique id
drug2idx = {}
idx = 0
for combo, se_set in combo2se.items():     
    if len(combo2se[combo].intersection(common_se)) == 0:
        continue
    drug_0 = combo.split("_")[0]
    drug_1 = combo.split("_")[1]
    if drug_0 not in drug2idx:
        drug2idx[drug_0] = idx
        idx+=1
    if drug_1 not in drug2idx:
        drug2idx[drug_1] = idx
        idx+=1
n_drugs = len(drug2idx)
print(idx)
#count up all the unique side effects, give them a unique id
idx = 0
se2idx = {}
for _, se_set in combo2se.items(): 
    for se in se_set:
        if se not in se2idx and se in common_se:
            se2idx[se] = idx
            idx+=1
        
print(idx)


# In[5]:


# Need to create a drug - drug interaction matrix
drug_drug_adj_list = [np.zeros((n_drugs, n_drugs)) for _ in se2idx]

count= 0
for drug_drug, se_set in combo2se.items():
    drug0 = drug_drug.split("_")[0]
    drug1 = drug_drug.split("_")[1]
    for se in se_set:
        if drug0 in drug2idx and  drug1 in drug2idx and se in se2idx:
            se_index = se2idx[se]
            drug0_index = drug2idx[drug0]
            drug1_index = drug2idx[drug1]
            drug_drug_adj_list[se_index][drug0_index][drug1_index] = 1        
            drug_drug_adj_list[se_index][drug1_index][drug0_index] = 1
            count+=1
        
drug_drug_adj_list = [ sp.csr_matrix(item) for item in drug_drug_adj_list]
drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]
        
print(count)


# ## Work with genes (aka proteins)

# In[6]:


gene_adj = nx.adjacency_matrix(gene_net)
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
n_genes = gene_adj.shape[0]


# ## Work with drug - protein interactions

# In[7]:


missing = 0 
present = 0
gene_drug_matrix =  np.zeros((n_genes, n_drugs))
for (drug_id, protein_set) in stitch2proteins.items():
    drug_idx = drug2idx[drug_id]
    #print(drug_id)
    #print(drug2idx[drug_id])
    #print(protein_set)
    for item in protein_set:
        if item not in node2idx:
            missing+=1
        else:
            protein_idx = node2idx[item]
        #print(node2idx[item])
            gene_drug_matrix[protein_idx][drug_idx] = 1
            present+=1
    #break
print("Unable to find {} out of {} proteins ({})".format(missing, present, missing/present))


gene_drug_adj = sp.csr_matrix(gene_drug_matrix)
drug_gene_adj = gene_drug_adj.transpose(copy=True)


# In[8]:


print(n_genes, n_drugs)


# In[9]:


# data representation
adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_drug_adj],
    (1, 0): [drug_gene_adj],
    (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
    #(1, 1): drug_drug_adj_list
}
degrees = {
    0: [gene_degrees, gene_degrees],
    #1: drug_degrees_list
    1: drug_degrees_list + drug_degrees_list,
}

# featureless (genes)
gene_feat = sp.identity(n_genes)
gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# features (drugs)
drug_feat = sp.identity(n_drugs)
drug_nonzero_feat, drug_num_feat = drug_feat.shape
drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

# data representation
num_feat = {
    0: gene_num_feat,
    1: drug_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: drug_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: drug_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'dedicom',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)


# In[10]:


###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)



# In[11]:


###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)


# In[12]:


import pickle

pickle_out = open("minibatch_v2.pickle","wb")
pickle.dump(minibatch, pickle_out)
pickle_out.close()

#fileObject = open("minibatch.pickle",'rb')  
#minibatch = pickle.load(fileObject)  
#fileObject.close()




# In[ ]:



###########################################################
#
# Train model
#
###########################################################

print("Train model")
for epoch in range(50):

    minibatch.shuffle()
    itr = 0
    while not minibatch.end():
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=0.1,
            placeholders=placeholders)

        t = time.time()
        print("Currently on epoch, iter", epoch, itr)

        itr += 1

print("Optimization finished!")

for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
print()
