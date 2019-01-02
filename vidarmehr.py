#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

#import pixiedust

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing


# In[2]:


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

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


# In[3]:


protein_protein_data = pd.read_csv("data/bio-decagon-ppi.csv", sep=',',header = 0)
polypharmacy_side_effect_data = pd.read_csv("data/bio-decagon-combo.csv", sep=',',header = 0)
drug_target_protein_data = pd.read_csv("data/bio-decagon-targets.csv", sep=',',header = 0)


# In[4]:


print("size of the protein-protein network:", protein_protein_data.shape)
print("size of the drug-target protein associations:", drug_target_protein_data.shape)
#print("size of the Drug-target protein associations culled from several curated databases :",drug_target_protein_all_data.shape)
print("size of the polypharmacy side effects:", polypharmacy_side_effect_data.shape)
#print("size of the Side effect categories:",side_effect_categories_data.shape)
#print("size of the Side effects of individual drugs:",side_effect_individuale_data.shape)
#protein_protein_data.head()


# # PPI graph

# In[5]:


df_gene1_gene2 = pd.crosstab(protein_protein_data['Gene 1'], protein_protein_data['Gene 2'])
gene_idx = df_gene1_gene2.columns.union(df_gene1_gene2.index)
df_gene1_gene2 = df_gene1_gene2.reindex(index = gene_idx, columns = gene_idx,fill_value=0)#upper triangle of the adjacency matrix

df_gene2_gene1 = pd.crosstab(protein_protein_data['Gene 2'], protein_protein_data['Gene 1'])
df_gene2_gene1 = df_gene2_gene1.reindex(index = gene_idx, columns = gene_idx, fill_value=0)#lower triangle of the adjacency matrix

gene_adj = df_gene2_gene1.add(df_gene1_gene2, fill_value=0)#creates a symmetric adjacency matrix of Gene 1 and Gene 2 by adding upper triangle and lower triangle
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()#get the degrees of genes
print("size of the gene-gene adjacecny matrix:",gene_adj.shape, ", number of genes:",len(gene_degrees))


# In[6]:


n_genes = len(gene_degrees)
n_genes


# In[7]:


indices_genes = list(range(0,len(gene_idx)))
dict_genes = dict(zip(gene_idx, indices_genes))#creating a dictionary to map genes to indices


# In[8]:


gene_adj_mat = np.zeros((n_genes,n_genes)) # adj. matrix of size n_genes * n_genes
for i in range(0,protein_protein_data.shape[0]):#read from the protein-protein file
    gene1 = protein_protein_data.loc[i][0]#read gene1
    gene2 = protein_protein_data.loc[i][1]#read gene2
    #print(gene1,gene2)
    gene1_index = dict_genes.get(gene1)#get the index of gene1 in dictionary
    gene2_index = dict_genes.get(gene2)#get the index of gene2 in dictionary
    #print(gene1_index,gene2_index)
    gene_adj_mat[gene1_index][gene2_index] = 1.0 
    gene_adj_mat[gene2_index][gene1_index] = 1.0


# In[9]:


G=nx.from_numpy_matrix(gene_adj_mat)
edges = G.edges(data=True)


# In[10]:


def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500)
    plt.show()


# In[11]:


gene_adj_mat_t = gene_adj_mat.transpose() #transpose matrix of the adj. matrix


# # Creating a list of symmetric adjacency matrices for each side effect from the drug-drug network (combo)
# 

# In[12]:


list_side_effects = []
for i in range(0,polypharmacy_side_effect_data.shape[0]):#get the list of side effects
    if polypharmacy_side_effect_data.loc[i][2] not in list_side_effects:
        list_side_effects.append(polypharmacy_side_effect_data.loc[i][2])
print("number of side effects:", len(list_side_effects))


# In[13]:


df_drug1_drug2 = pd.crosstab(polypharmacy_side_effect_data['STITCH 1'], polypharmacy_side_effect_data['STITCH 2'])
drug_idx = df_drug1_drug2.columns.union(df_drug1_drug2.index)#get names of all drugs


# In[14]:


indices_drugs = list(range(0,len(drug_idx)))
dict_drugs = dict(zip(drug_idx, indices_drugs))#create a dictionary to map each drug to its index.
n_drugs = len(drug_idx)
print("number of drugs:",len(drug_idx))


# In[15]:


drug_drug_adj_list = [np.zeros((n_drugs, n_drugs)) for _ in list_side_effects]
side_effects_mapping = {list_side_effects[i]: i for i in range(len(list_side_effects))}

print(polypharmacy_side_effect_data.shape[0])
for row_index in range(polypharmacy_side_effect_data.shape[0]):
    if row_index % 100000 == 0:
        print(row_index)
    se = polypharmacy_side_effect_data.loc[row_index][2]
    seidx = side_effects_mapping[se]
    
    drug_1 = polypharmacy_side_effect_data.loc[row_index][0]
    drug_2 = polypharmacy_side_effect_data.loc[row_index][1] 
    drug_1_index = dict_drugs.get(drug_1)
    drug_2_index = dict_drugs.get(drug_2)
    
    drug_drug_adj_list[seidx][drug_1_index,drug_2_index] = drug_drug_adj_list[seidx][drug_2_index,drug_1_index] = 1.
drug_drug_adj_list = [sp.csr_matrix(drug_drug_mat) for drug_drug_mat in drug_drug_adj_list]# add the adjacency matrix for the side effect to the list of adj. matrices
drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]


# In[16]:


(n_drugs,n_genes)


# # Creating adjacency matrices for the drug-protein and protein-drug network.
# 

# In[17]:


drug_gene_adj = np.zeros((n_drugs,n_genes))


# In[19]:


for i in range(0,drug_target_protein_data.shape[0]):
    drug = drug_target_protein_data.loc[i][0]#read drug from the drug-gene data
    gene = drug_target_protein_data.loc[i][1]#read gene from the drug-gene data
    gene_index = dict_genes.get(gene)#get the index of gene from the gene dictionary
    drug_index = dict_drugs.get(drug)#ge the indx of drug from the drug dictionary
    drug_gene_adj[drug_index][gene_index] = 1.0
gene_drug_adj = drug_gene_adj.transpose()


# In[ ]:





# In[20]:


drug_target_protein_data.shape[0]


# In[21]:


adj_mats_orig = {
    (0, 0): [sp.csr_matrix(gene_adj_mat), sp.csr_matrix(gene_adj_mat_t)],
    (0, 1): [sp.csr_matrix(gene_drug_adj)],
    (1, 0): [sp.csr_matrix(drug_gene_adj)],
    (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
}


# In[22]:


degrees = {
    0: [gene_degrees, gene_degrees],
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


# In[23]:


edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.app.flags
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 1, 'minibatch size.') # was 512
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)

tf.app.flags.DEFINE_string('f', '', 'kernel')
FLAGS = flags.FLAGS


# In[24]:


val_test_size = 0.1


# In[27]:


print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)


# In[28]:


import pickle                               
                                                                           
pickle_out = open("minibatch_vida.pickle","wb")
pickle.dump(minibatch, pickle_out)
pickle_out.close()


# In[ ]:


print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)


# In[ ]:


print("Create optimizer")
with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )



print("Initialize session")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}


# In[19]:




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

        # Training step: run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
        train_cost = outs[1]
        batch_edge_type = outs[2]

        if itr % PRINT_PROGRESS_EVERY == 0:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])

            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

    if epoch % 5 == 0:
        f = open("{}_iter_{}.txt".format(modelname, epoch), "w")
        for et in range(num_edge_types):
            roc_score, auprc_score, apk_score = get_accuracy_scores(
                minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
            print 'Test AUROC score %d: %5.3f' % (et, roc_score)
            print 'Test AUPRC score %d: %5.3f' % (et, auprc_score)
            print 'Test AP@k score %d: %5.3f' % (et, apk_score)
            print
            f.write("{},{},{},{},{},{}\n".format(
                minibatch.idx2edge_type[et][0],minibatch.idx2edge_type[et][1],minibatch.idx2edge_type[et][2], 
                roc_score, auprc_score, apk_score))
        f.close()

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


f = open("{}_iter_{}.txt".format(modelname, "final"), "w")
for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print 'Test AUROC score %d: %5.3f' % (et, roc_score)
    print 'Test AUPRC score %d: %5.3f' % (et, auprc_score)
    print 'Test AP@k score %d: %5.3f' % (et, apk_score)
    print
    f.write("{},{},{},{},{},{}\n".format(
        minibatch.idx2edge_type[et][0],minibatch.idx2edge_type[et][1],minibatch.idx2edge_type[et][2], 
        roc_score, auprc_score, apk_score))
f.close()
