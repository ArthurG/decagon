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

from polypharmacy import utility


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
    predicted = zip(*sorted(predicted, reverse=True, key=itemgetter(0)))[1]

    roc_score = metrics.roc_auc_score(labels_all, preds_all)
    aupr_score = metrics.average_precision_score(labels_all, preds_all)
    apk_score = rank_metrics.apk(actual, predicted, k=50)

    return roc_score, aupr_score, apk_score


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

###########################################################
#
# Load and preprocess data (This is a dummy toy example!) #
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

"""
val_test_size = 0.05
n_genes = 500
n_drugs = 400
n_drugdrug_rel_types = 3
gene_net = nx.planted_partition_graph(50, 10, 0.2, 0.05, seed=42)

gene_adj = nx.adjacency_matrix(gene_net)
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

gene_drug_adj = sp.csr_matrix((10 * np.random.randn(n_genes, n_drugs) > 15).astype(int))
drug_gene_adj = gene_drug_adj.transpose(copy=True)

drug_drug_adj_list = []
tmp = np.dot(drug_gene_adj, gene_drug_adj)
for i in range(n_drugdrug_rel_types):
    mat = np.zeros((n_drugs, n_drugs))
    for d1, d2 in combinations(list(range(n_drugs)), 2):
        if tmp[d1, d2] == i+4:
            mat[d1, d2] = mat[d2, d1] = 1.
    drug_drug_adj_list.append(sp.csr_matrix(mat))
drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]
"""


val_test_size = 0.10


gene_net, node2idx = utility.load_ppi_v2("data/bio-decagon-ppi.csv") # protein protein interactions
stitch2proteins = utility.load_targets("data/bio-decagon-targets.csv") #drug protein interations
combo2stitch, combo2se, se2name = utility.load_combo_se('data/bio-decagon-combo.csv') #Drug drug interactions


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

#count up all the unique side effects, give them a unique id
idx = 0
se2idx = {}
for _, se_set in combo2se.items(): 
    for se in se_set:
        if se not in se2idx and se in common_se:
            se2idx[se] = idx
            idx+=1
        


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




# data representation
adj_mats_orig = {
    (0,0): [gene_adj],
    (0,1): [gene_drug_adj],
    (1,0): [drug_gene_adj],
    (1,1): drug_drug_adj_list,
}
degrees = {
    0: [gene_degrees],
    1: drug_degrees_list,
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

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.iteritems()}
edge_type2directed = {
    (0,0): [False],
    (0,1): [True],
    (1,0): [True],
    (1,1): [False]*len(drug_drug_adj_list),
}
edge_type2decoder = {
    (0,0): 'bilinear',
    (0,1): 'bilinear',
    (1,0): 'bilinear',
    (1,1): 'dedicom',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.iteritems()}
num_edge_types = sum(edge_types.values())
print 'Edge types: %d' % num_edge_types

###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
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

print 'Defining placeholders'
placeholders = construct_placeholders(edge_types)

###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

"""
print 'Create minibatch iterator'
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    directed=edge_type2directed,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)
"""

import pickle

"""
pickle_out = open("minibatch.pickle","wb")
pickle.dump(minibatch, pickle_out)
pickle_out.close()
"""

fileObject = open("minibatch.pickle",'rb')  
minibatch = pickle.load(fileObject)  
fileObject.close()




print 'Create model'
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)

print 'Create optimizer'
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

print 'Initialize session'
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}

###########################################################
#
# Train model
#
###########################################################

curr_val_acc = -1
prev_val_acc = -1
early_stop_count = 0

print("Train model")
for epoch in range(FLAGS.epochs):

    minibatch.shuffle()
    itr = 0
    while not minibatch.end():
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
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

        itr += 1

    roc_list = []
    prc_list = []
    apk_list = []
    for et in range(num_edge_types):
        roc_score, auprc_score, apk_score = get_accuracy_scores(
            minibatch.val_edges, minibatch.val_edges_false, minibatch.idx2edge_type[et])
        roc_list.append(roc_score)
        prc_list.append(auprc_score)
        apk_list.append(apk_score)
    print 'Val AUROC score: %5.3f' % (np.mean(roc_list))
    print 'Val AUPRC score: %5.3f' % (np.mean( auprc_score))
    print 'Val AP@k score: %5.3f' % (np.mean( apk_score))

    curr_val_acc = np.mean(roc_list)
    if curr_val_acc < prev_val_acc:
        early_stop_count += 1
        if early_stop_count == 2:
            print("Stopping early on epoch %d".format(epoch))
            break
        print("Early stop warning on epoch %d".format(epoch))
    else:
        early_stop_count = 0
    prev_val_acc = curr_val_acc


print("Optimization finished!")

for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print 'Test AUROC score %d: %5.3f' % (et, roc_score)
    print 'Test AUPRC score %d: %5.3f' % (et, auprc_score)
    print 'Test AP@k score %d: %5.3f' % (et, apk_score)
    print

f.close()
