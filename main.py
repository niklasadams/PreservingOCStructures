import ocpa.algo.predictive_monitoring.factory
from gnn_utils import *
from ocpa.objects.log.importer.csv import factory as csv_import_factory
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
from ocpa.objects.log.util import misc as log_util
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
import pandas as pd
from graph_embedding import convert_to_nx_graphs, embed
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from tqdm import tqdm
params = {"sap": {"batch_size": 4, "lr": 0.001, "epochs": 15}}

NEXT_ACTIVITY = "next_activity"

def next_activity(node, ocel, params):
    act = params[0]
    e_id = node.event_id
    out_edges = ocel.graph.eog.out_edges(e_id)
    next_act = 0
    for (source, target) in out_edges:
        if ocel.get_value(target, "event_activity") == act:
            next_act = 1
            for (source_, target_) in out_edges:
                if ocel.get_value(target_, "event_timestamp") < ocel.get_value(target, "event_timestamp"):
                    next_act = 0
    return next_act


NEXT_TIMESTAMP = "next_timestamp"


def next_timestamp(node, ocel, params):
    e_id = node.event_id
    out_edges = ocel.graph.eog.out_edges(e_id)
    if len(out_edges) == 0:
        # placeholder, will not be used for prediction
        return 0
    return min([ocel.get_value(target, "event_timestamp") for (source, target) in out_edges]).to_pydatetime().timestamp() - ocel.get_value(e_id, "event_timestamp").to_pydatetime().timestamp()


REL_ACTIVITY_OCCURRENCE_PER_TYPE ="act_occurrence_p_type"
def relative_activity_occurence(node,ocel,params):
    ots = params[0]
    acts = params[1]
    results_dict = {(ot,act):0 for (ot,act) in itertools.product(ots, acts)}
    e_id = node.event_id
    case = ocel.process_executions[node.pexec_id]
    pexec_objects = ocel.process_execution_objects[node.pexec_id]
    oc_counter = {act:{(ot_,o):False for (ot_,o) in pexec_objects} for act in acts}
    this_timestamp = ocel.get_value(e_id,"event_timestamp")
    for e in case:
        if ocel.get_value(e,"event_timestamp") > this_timestamp:
            continue
        else:
            curr_act = ocel.get_value(e,"event_activity")
            ev_ob_dict = {}
            for ot_iter in ots:
                ev_ob_dict[ot_iter] = [o for o in ocel.get_value(e_id,ot_iter)]
            for (ot_,o) in pexec_objects:
                if o in ev_ob_dict[ot_]:
                    oc_counter[curr_act][(ot_,o)] = True
    rel_counter = 0
    for curr_act in acts:
        for curr_ot in ots:
            curr_ot_obs = [(ot_,o) for (ot_,o) in pexec_objects if ot_ == curr_ot]
            for (ot_,o) in curr_ot_obs:
                if oc_counter[curr_act][(ot_,o)]:
                    rel_counter += 1
            results_dict[(curr_ot,curr_act)] = rel_counter/len(curr_ot_obs)

    return results_dict

ocpa.algo.predictive_monitoring.factory.VERSIONS[
    ocpa.algo.predictive_monitoring.factory.EVENT_BASED][REL_ACTIVITY_OCCURRENCE_PER_TYPE] = relative_activity_occurence
ocpa.algo.predictive_monitoring.factory.VERSIONS[
    ocpa.algo.predictive_monitoring.factory.EVENT_BASED][NEXT_ACTIVITY] = next_activity
ocpa.algo.predictive_monitoring.factory.VERSIONS[
    ocpa.algo.predictive_monitoring.factory.EVENT_BASED][NEXT_TIMESTAMP] = next_timestamp

def tf_acc_score(test_target,res):
    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(test_target, res)
    acc_score = m.result().numpy()
    return acc_score

def filter_process_executions(ocel, cases):

    events = [e for case in cases for e in case]
    new_event_df = ocel.log.log.loc[ocel.log.log["event_id"].isin(
        events)].copy()
    new_log = log_util.copy_log_from_df(new_event_df, ocel.parameters)
    return new_log


def cat_target_to_vector(feature_storage, targets, new_name):
    # targets needs to be ordered
    for g in feature_storage.feature_graphs:
        for node in g.nodes:
            vec = [node.attributes[t] for t in targets]
            vec = [1 if e == max(vec) else 0 for e in vec]
            node.attributes[new_name] = vec
            for t in targets:
                del node.attributes[t]
    return feature_storage


def GNN_prediction(layer_size, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=64, lr=0.01, n_output=1):
    # return 0,0,0,0

    train_loader = GraphDataLoader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        add_self_loop=True,
        make_bidirected=False,
        on_gpu=False
    )
    val_loader = GraphDataLoader(
        x_val,
        y_val,
        batch_size=batch_size,
        shuffle=True,
        add_self_loop=True,
        make_bidirected=False,
        on_gpu=False
    )
    test_loader = GraphDataLoader(
        x_test,
        y_test,
        batch_size=128,
        shuffle=False,
        add_self_loop=True,
        make_bidirected=False,
        on_gpu=False
    )

    # define GCN model
    tf.keras.backend.clear_session()
    model = None
    # regression
    if n_output == 1:
        model = GCN(layer_size, layer_size)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
        loss_function = tf.keras.losses.MeanSquaredError()
    else:
        model = ClassificationGCN(layer_size, layer_size, n_output)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
        loss_function = tf.keras.losses.MeanSquaredError()
    # run tensorflow training loop
    epochs = 20
    iter_idx = np.arange(0, train_loader.__len__())
    loss_history = []
    val_loss_history = []
    step_losses = []
    for e in range(epochs):
        print('Running epoch:', e)
        np.random.shuffle(iter_idx)
        current_loss = step = 0
        for batch_id in tqdm(iter_idx):
            step += 1
            dgl_batch, label_batch = train_loader.__getitem__(batch_id)
            # if n_output != 1:
            #    label_batch = tf.reshape(label_batch, (int(label_batch.shape[0] * label_batch.shape[1]),))
            with tf.GradientTape() as tape:
                pred = model(dgl_batch, dgl_batch.ndata['features'])
                # print(pred.shape)
                # if n_output != 1:
                #    pred = tf.reshape(pred, (int(pred.shape[0] * pred.shape[1]),))
                loss = loss_function(label_batch, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            step_losses.append(loss.numpy())
            current_loss += loss.numpy()
            # if (step % 100 == 0): print('Loss: %s'%((current_loss / step)))
            loss_history.append(current_loss / step)
        val_predictions, val_labels = evaluate_gnn(val_loader, model)
        val_loss = 0
        if n_output == 1:
            val_loss = tf.keras.metrics.mean_squared_error(
                np.squeeze(val_labels), np.squeeze(val_predictions)).numpy()
            #val_loss = tf.keras.metrics.mean_absolute_error(
            #    np.squeeze(val_labels), np.squeeze(val_predictions)).numpy()
            print('    Validation MAE GNN:', val_loss)
            if len(val_loss_history) < 1:
                model.save_weights('gnn_checkpoint.tf')
                print('    GNN checkpoint saved.')
            else:
                if val_loss < np.min(val_loss_history):
                    model.save_weights('gnn_checkpoint.tf')
                    print('    GNN checkpoint saved.')
        else:
            m = tf.keras.metrics.CategoricalAccuracy()
            m.update_state(val_labels, val_predictions)
            m.result().numpy()
            # tf.keras.metrics.categorical_accuracy(val_labels, val_predictions).numpy()
            val_loss = m.result().numpy()

            print('    Validation Accuracy GNN:', val_loss)
            if len(val_loss_history) < 1:
                model.save_weights('gnn_checkpoint.tf')
                print('    GNN checkpoint saved.')
            else:
                if val_loss > np.min(val_loss_history):
                    model.save_weights('gnn_checkpoint.tf')
                    print('    GNN checkpoint saved.')
        val_loss_history.append(val_loss)

    # visualize training progress
    pd.DataFrame({'loss': loss_history, 'step_losses': step_losses}).plot(
        subplots=True, layout=(1, 2), sharey=True)

    # restore weights from best epoch
    cp_status = model.load_weights('gnn_checkpoint.tf')
    cp_status.assert_consumed()

    # generate predictions and calculate MAE for train, val & test sets
    train_predictions, train_labels = evaluate_gnn(train_loader, model)
    val_predictions, val_labels = evaluate_gnn(val_loader, model)
    test_predictions, test_labels = evaluate_gnn(test_loader, model)

    # regression
    if n_output == 1:
        mean_prediction = np.mean(np.array(y_train))
        print('MAE baseline: ')
        baseline = mean_absolute_error(
            test_labels, np.repeat(mean_prediction, len(test_labels)))
        print(mean_absolute_error(test_labels, np.repeat(
            mean_prediction, len(test_labels))))
        print('MAE GNN: ')
        test_score = mean_absolute_error(test_predictions, test_labels)
        print(test_score)
        train_score = mean_absolute_error(train_predictions, train_labels)
        val_score = mean_absolute_error(val_predictions, val_labels)

    # classification
    else:
        agg_elems = sum(train_labels)
        majority = list(agg_elems).index(max(list(agg_elems)))
        max_elem_prediction = [
            1 if i == majority else 0 for i in range(0, len(agg_elems))]
        print("Accuracy baseline: ")
        baseline_predictions = [
            max_elem_prediction for i in range(0, len(test_labels))]
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(test_labels, baseline_predictions)
        baseline = m.result().numpy()
        print(baseline)
        print('Accuracy GNN: ')
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(test_labels, test_predictions)
        test_score = m.result().numpy()
        print(test_score)
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(train_labels, train_predictions)
        train_score = m.result().numpy()
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(val_labels, val_predictions)
        val_score = m.result().numpy()

    return baseline, train_score, val_score, test_score

#
filename = "BPI2017-Final.csv"
lr = 0.01
batch_size = 256
object_types = ["application", "offer"]
parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}
ocel = csv_import_factory.apply(file_path=filename, parameters=parameters)
ks = [2, 3, 4, 5, 6, 7, 8]
ks = [i for i in range(8,1,-1)]

#ks = [8]


# This is a small test log
# filename = "p2p-normal.jsonocel"
# ocel = ocel_import_factory.apply(filename)
# lr = 0.01
# batch_size = 32
# ks = [4]


# This is the order management event log
# filename = "orders.jsonocel"
# #parameters = {"execution_extraction": "leading_type",
# #             "leading_type": "items"}
# ocel = ocel_import_factory.apply(filename)#, parameters=parameters)
# lr = 0.01
# batch_size = 256
# ks = [2,3,4,5,6,7,8]
# ks = [i for i in range(8,1,-1)]



print("Number of process executions: "+str(len(ocel.process_executions)))
print("Average lengths: " +
      str(sum([len(e) for e in ocel.process_executions])/len(ocel.process_executions)))
activities = list(set(ocel.log.log["event_activity"].tolist()))
print(str(len(activities))+" actvities")
accuracy_dict = {}

for target in [[(NEXT_ACTIVITY, (act,)) for act in activities]]:
    include_last = False
    F = target+[(predictive_monitoring.EVENT_SYNCHRONIZATION_TIME, ())]
    feature_storage = predictive_monitoring.apply(ocel, F, [])
    # replace synchronization time with 0 placeholder for empty feature
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time', ())] = 1
    feature_storage.extract_normalized_train_test_split(0.3, state=3)
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time', ())] = 1

    # replace categorical features with vector
    new_target_name = (target[0][0], ())
    feature_storage = cat_target_to_vector(
        feature_storage, target, new_target_name)
    for k in ks:
        if True:
            print("___________________________")
            print("Prediction with Graph Structure and GNN")
            print("___________________________")

            layer_size = len(F)-len(target)

            # generate training & test datasets
            train_idx, val_idx = train_test_split(
                feature_storage.training_indices, test_size=0.2)
            x_train, y_train = generate_graph_dataset(
                feature_storage.feature_graphs, train_idx, ocel, k=k, target=new_target_name, include_last=include_last)

            x_val, y_val = generate_graph_dataset(
                feature_storage.feature_graphs, val_idx, ocel, k=k, target=new_target_name, include_last=include_last)
            x_test, y_test = generate_graph_dataset(
                feature_storage.feature_graphs, feature_storage.test_indices, ocel, k=k, target=new_target_name, include_last=include_last)
            start_time = time.time()
            baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(
                layer_size, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=batch_size, lr=lr, n_output=len(activities))
            calc_time = time.time() - start_time
            # record performance of GNN
            accuracy_dict[new_target_name[0]+'_graph_gnn_k_' + str(k)] = {
                'baseline_ACC': baseline_MAE,
                'train_ACC': train_MAE,
                'val_ACC': val_MAE,
                'test_ACC': test_MAE,
                "time":calc_time
            }
            print(pd.DataFrame(accuracy_dict))
        if True:
            print("___________________________")
            print("Prediction with Sequential Structure and GNN")
            print("___________________________")

            layer_size = len(F) - len(target)
            train_idx, val_idx = train_test_split(
                feature_storage.training_indices, test_size=0.2)
            x_train, y_train = generate_sequential_graph_dataset(feature_storage.feature_graphs, train_idx, ocel, k=k,
                                                                 target=new_target_name, include_last=include_last)
            x_val, y_val = generate_sequential_graph_dataset(feature_storage.feature_graphs, val_idx, ocel, k=k,
                                                             target=new_target_name, include_last=include_last)
            x_test, y_test = generate_sequential_graph_dataset(feature_storage.feature_graphs,
                                                               feature_storage.test_indices, ocel, k=k, target=new_target_name,
                                                               include_last=include_last)
            start_time= time.time()
            baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(layer_size, x_train, y_train, x_val, y_val,
                                                                        x_test, y_test, batch_size=batch_size, lr=lr,
                                                                        n_output=len(activities))
            calc_time = time.time() - start_time
            # record performance of GNN
            accuracy_dict[new_target_name[0] + '_flat_gnn_k_' + str(k)] = {
                'baseline_ACC': baseline_MAE,
                'train_ACC': train_MAE,
                'val_ACC': val_MAE,
                'test_ACC': test_MAE,
                "time":calc_time
            }
            print(pd.DataFrame(accuracy_dict))
        if True:
            print("___________________________")
            print("Prediction with Graph Embedding")
            print("___________________________")
            train_nx_feature_graphs = []
            test_nx_feature_graphs = []
            train_target = []
            test_target = []
            print("Constructing Subgraphs ")
            for i in tqdm(feature_storage.training_indices):
                g = feature_storage.feature_graphs[i]
                converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                              target=new_target_name,
                                                                              from_start=False, include_last=include_last)

                train_nx_feature_graphs += converted_subgraphs
                train_target += extracted_targets

            for i in tqdm(feature_storage.test_indices):
                g = feature_storage.feature_graphs[i]
                converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                              target=new_target_name,
                                                                              from_start=False, include_last=include_last)
                test_nx_feature_graphs += converted_subgraphs
                test_target += extracted_targets

            for embedding_technique in [
                'Graph2Vec', 'NetLSD', 'WaveletCharacteristic',
                    'LDP', 'GL2Vec', 'SF', 'FGSD']:
                try:
                    start_time = time.time()
                    X_train, X_test = embed(
                        train_nx_feature_graphs, test_nx_feature_graphs, embedding_technique, size=10*k)
                    calc_time = time.time() -start_time
                    print(X_train.shape)
                    print(X_test.shape)

                    model = LogisticRegression(multi_class='multinomial')
                    model.fit(X_train, [ys.index(1) for ys in train_target])
                    res = model.predict(X_test)
                    acc_score = accuracy_score(
                        [ys.index(1) for ys in test_target], res)
                    print(accuracy_score([ys.index(1)
                          for ys in test_target], res))
                    accuracy_dict[new_target_name[0]+'_embed_reg_' + embedding_technique+'_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score,
                        'time': calc_time
                    }
                    regr = MLPClassifier(random_state=3, max_iter=2000, hidden_layer_sizes=(
                        5, 5)).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[new_target_name[0]+'_embed_nn_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score,
                        'time': calc_time
                    }
                except ValueError:
                    accuracy_dict[new_target_name[0] + '_embed_reg_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': "NA",
                        'time': 0
                    }
                    accuracy_dict[new_target_name[0] + '_embed_nn_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': "NA",
                        'time': 0
                    }

            print(pd.DataFrame(accuracy_dict))


for target in [(NEXT_TIMESTAMP, ()), (predictive_monitoring.EVENT_REMAINING_TIME, ())]:
    include_last = True
    if target == (NEXT_TIMESTAMP, ()):
        include_last = False

    F = [target,
         (predictive_monitoring.EVENT_SYNCHRONIZATION_TIME, ())]
    feature_storage = predictive_monitoring.apply(ocel, F, [])

    # replace synchronization time with 1 as placeholder for empty feature
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time', ())] = 1
    feature_storage.extract_normalized_train_test_split(0.3, state=3)
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time', ())] = 1

    g_set_list_t = []
    g_set_list_te = []
    seq_set_list = []
    seq_set_list_v = []
    seq_set_list_t = []

    for k in ks:
        if True:
            print("___________________________")
            print("Prediction with Graph Structure and GNN")
            print("___________________________")

            layer_size = len(F)-1

            # generate training & test datasets
            train_idx, val_idx = train_test_split(
                feature_storage.training_indices, test_size=0.2)
            x_train, y_train = generate_graph_dataset(
                feature_storage.feature_graphs, train_idx, ocel, k=k, target=target, include_last=include_last)
            x_val, y_val = generate_graph_dataset(
                feature_storage.feature_graphs, val_idx, ocel, k=k, target=target, include_last=include_last)
            x_test, y_test = generate_graph_dataset(
                feature_storage.feature_graphs, feature_storage.test_indices, ocel, k=k, target=target, include_last=include_last)
            start_time = time.time()
            baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(
                layer_size, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=batch_size, lr=lr)
            calc_time = time.time() - start_time
            # record performance of GNN
            accuracy_dict[target[0]+'graph_gnn_k_' + str(k)] = {
                'baseline_MAE': baseline_MAE,
                'train_MAE': train_MAE,
                'val_MAE': val_MAE,
                'test_MAE': test_MAE,
                "time":calc_time
            }
            print(pd.DataFrame(accuracy_dict))

        if True:
            print("___________________________")
            print("Prediction with Sequential Structure and GNN")
            print("___________________________")

            layer_size = len(F)-1

            # generate training & test datasets
            train_idx, val_idx = train_test_split(
                feature_storage.training_indices, test_size=0.2)
            x_train, y_train = generate_sequential_graph_dataset(
                feature_storage.feature_graphs, train_idx, ocel, k=k, target=target, include_last=include_last)
            x_val, y_val = generate_sequential_graph_dataset(
                feature_storage.feature_graphs, val_idx, ocel, k=k, target=target, include_last=include_last)
            x_test, y_test = generate_sequential_graph_dataset(
                feature_storage.feature_graphs, feature_storage.test_indices, ocel, k=k, target=target, include_last=include_last)
            start_time = time.time()
            baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(layer_size, x_train, y_train, x_val, y_val, x_test,
                                                                        y_test, batch_size=batch_size, lr=lr)
            calc_time = time.time() - start_time

            # record performance of GNN
            accuracy_dict[target[0]+'flat_gnn_k_' + str(k)] = {
                'baseline_MAE': baseline_MAE,
                'train_MAE': train_MAE,
                'val_MAE': val_MAE,
                'test_MAE': test_MAE,
                "time": calc_time
            }
            print(pd.DataFrame(accuracy_dict))

        if True:
            print("___________________________")
            print("Prediction with Graph Embedding")
            print("___________________________")
            train_nx_feature_graphs = []
            test_nx_feature_graphs = []
            train_target = []
            test_target = []
            print("Constructing Subgraphs ")
            for i in tqdm(feature_storage.training_indices):
                g = feature_storage.feature_graphs[i]
                converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                              target=target,
                                                                              from_start=False, include_last=include_last)

                train_nx_feature_graphs += converted_subgraphs
                train_target += extracted_targets

            for i in tqdm(feature_storage.test_indices):
                g = feature_storage.feature_graphs[i]
                converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                              target=target,
                                                                              from_start=False, include_last=include_last)
                test_nx_feature_graphs += converted_subgraphs
                test_target += extracted_targets

            # IGE has problems with sparseness
            for embedding_technique in [  'Graph2Vec', 'NetLSD', 'WaveletCharacteristic',
                    'LDP', 'GL2Vec', 'SF', 'FGSD']:
                try:
                    start_time = time.time()
                    X_train, X_test = embed(
                        train_nx_feature_graphs, test_nx_feature_graphs, embedding_technique, size=10*k)
                    calc_time = time.time() - start_time
                    print(X_train.shape)
                    print(X_test.shape)
                    model = LinearRegression()
                    model.fit(X_train, train_target)
                    res = model.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    accuracy_dict[target[0]+'embed_reg_' + embedding_technique+'_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=2000,early_stopping=True, hidden_layer_sizes=(
                        5, 5,)).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    accuracy_dict[target[0]+'embed_nn_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }


                except ValueError:
                    accuracy_dict[target[0] + 'embed_nn_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': "NA",
                        "time":0
                    }
                    accuracy_dict[target[0] + 'embed_reg_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': "NA",
                        "time":0
                    }
            print(pd.DataFrame(accuracy_dict))


#pd.set_option('display.max_columns', None)
print(pd.DataFrame(accuracy_dict))
pd.DataFrame(accuracy_dict).to_csv("metrics.csv")


#############################
#Structural Delta Experiments
#############################
accuracy_dict = {}
if True:
    

    for target in [(predictive_monitoring.EVENT_REMAINING_TIME, ()),(NEXT_TIMESTAMP, ())]:
        include_last = True
        if target == (NEXT_TIMESTAMP, ()):
            include_last = False

        F = [target] + [(predictive_monitoring.EVENT_TYPE_COUNT, (ot,)) for ot in ocel.object_types] #+ [(REL_ACTIVITY_OCCURRENCE_PER_TYPE, (ot,act)) for (ot,act) in itertools.product(ocel.object_types, activities)]
        feature_storage = predictive_monitoring.apply(ocel, F, [], multi_output_event_features=[(REL_ACTIVITY_OCCURRENCE_PER_TYPE, (ocel.object_types,activities))])
        feature_storage.extract_normalized_train_test_split(0.3, state=3)

        g_set_list_t = []
        g_set_list_te = []
        seq_set_list = []
        seq_set_list_v = []
        seq_set_list_t = []

        for k in ks:
            if True:
                print("___________________________")
                print("Prediction with Graph Embedding")
                print("___________________________")
                train_nx_feature_graphs = []
                test_nx_feature_graphs = []
                train_target = []
                test_target = []
                print("Constructing Subgraphs ")
                for i in tqdm(feature_storage.training_indices):
                    g = feature_storage.feature_graphs[i]
                    converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                                  target=target,
                                                                                  from_start=False,
                                                                                  include_last=include_last)

                    train_nx_feature_graphs += converted_subgraphs
                    train_target += extracted_targets

                for i in tqdm(feature_storage.test_indices):
                    g = feature_storage.feature_graphs[i]
                    converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                                  target=target,
                                                                                  from_start=False,
                                                                                  include_last=include_last)
                    test_nx_feature_graphs += converted_subgraphs
                    test_target += extracted_targets


                for embedding_technique in ['baseline','manual','delta_manual']:

                    X_train, X_test = embed(
                        train_nx_feature_graphs, test_nx_feature_graphs, embedding_technique, size=10 * k)

                    print(X_train.shape)
                    print(X_test.shape)
                    print(X_train)
                    model = LinearRegression()
                    model.fit(X_train, train_target)
                    res = model.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    accuracy_dict[target[0] + 'embed_reg_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500, hidden_layer_sizes=(
                        20, 10), early_stopping=True, solver="lbfgs", batch_size=512).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    # print(mean_squared_error(test_target,res))
                    accuracy_dict[target[0] + 'embed_nn_20_10_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500, hidden_layer_sizes=(
                        20,), early_stopping=True, solver="lbfgs", batch_size=512).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    # print(mean_squared_error(test_target,res))
                    accuracy_dict[target[0] + 'embed_nn_20_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500, hidden_layer_sizes=(
                        18,9),early_stopping=True, solver="lbfgs",batch_size=512).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    #print(mean_squared_error(test_target,res))
                    accuracy_dict[target[0] + 'embed_nn_18_9_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500, hidden_layer_sizes=(
                        18,), early_stopping=True, solver="lbfgs", batch_size=512).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    # print(mean_squared_error(test_target,res))
                    accuracy_dict[target[0] + 'embed_nn_18_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500, hidden_layer_sizes=(
                        16, 8), early_stopping=True, solver="lbfgs", batch_size=512).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    # print(mean_squared_error(test_target,res))
                    accuracy_dict[target[0] + 'embed_nn_16_8_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500, hidden_layer_sizes=(
                        16,), early_stopping=True, solver="lbfgs", batch_size=512).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    #print(mean_squared_error(test_target, res))
                    accuracy_dict[target[0] + 'embed_nn_16_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500, hidden_layer_sizes=(
                        14,7), early_stopping=True, solver="lbfgs", batch_size=512).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    #print(mean_squared_error(test_target, res))
                    accuracy_dict[target[0] + 'embed_nn_14_7_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500, hidden_layer_sizes=(
                        14,), early_stopping=True, solver="lbfgs", batch_size=512).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    #print(mean_squared_error(test_target, res))
                    accuracy_dict[target[0] + 'embed_nn_14_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }




                print(pd.DataFrame(accuracy_dict))
    for target in [[(NEXT_ACTIVITY, (act,)) for act in activities]]:
        include_last = False
        #CHANGE FEATURES
        F = target + [(predictive_monitoring.EVENT_TYPE_COUNT, (ot,)) for ot in ocel.object_types] #+ [
            #(REL_ACTIVITY_OCCURRENCE_PER_TYPE, (ot, act)) for (ot, act) in
            #itertools.product(ocel.object_types, activities)]
        feature_storage = predictive_monitoring.apply(ocel, F, [],multi_output_event_features=[(REL_ACTIVITY_OCCURRENCE_PER_TYPE, (ocel.object_types,activities))])
        # replace synchronization time with 0 placeholder for empty feature
        feature_storage.extract_normalized_train_test_split(0.3, state=3)


        # replace categorical features with vector
        new_target_name = (target[0][0], ())
        feature_storage = cat_target_to_vector(
            feature_storage, target, new_target_name)
        for k in ks:
            if True:
                print("___________________________")
                print("Prediction with Graph Embedding")
                print("___________________________")
                train_nx_feature_graphs = []
                test_nx_feature_graphs = []
                train_target = []
                test_target = []
                print("Constructing Subgraphs ")
                for i in tqdm(feature_storage.training_indices):
                    g = feature_storage.feature_graphs[i]
                    converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                                  target=new_target_name,
                                                                                  from_start=False,
                                                                                  include_last=include_last)

                    train_nx_feature_graphs += converted_subgraphs
                    train_target += extracted_targets

                for i in tqdm(feature_storage.test_indices):
                    g = feature_storage.feature_graphs[i]
                    converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                                  target=new_target_name,
                                                                                  from_start=False,
                                                                                  include_last=include_last)
                    test_nx_feature_graphs += converted_subgraphs
                    test_target += extracted_targets

                for embedding_technique in ['baseline','manual','delta_manual']:
                    X_train, X_test = embed(
                        train_nx_feature_graphs, test_nx_feature_graphs, embedding_technique, size=10 * k)
                    print(X_train.shape)
                    print(X_test.shape)
                    model = LogisticRegression(multi_class='multinomial')
                    model.fit(X_train, [ys.index(1) for ys in train_target])
                    res = model.predict(X_test)
                    acc_score = accuracy_score(
                        [ys.index(1) for ys in test_target], res)
                    print(acc_score)
                    accuracy_dict[new_target_name[0] + '_delta_reg_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }
                    regr = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(
                        20, 10), early_stopping=True, batch_size=512).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[
                        new_target_name[0] + '_delta_nn_20_10_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }
                    regr = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(
                        20, ), early_stopping=True, batch_size=512).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[
                        new_target_name[0] + '_delta_nn_20_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }
                    regr = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(
                        18,9),early_stopping=True,batch_size=512).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[new_target_name[0] + '_delta_nn_18_9_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }
                    regr = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(
                        18,), early_stopping=True, batch_size=512).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[new_target_name[0] + '_delta_nn_18_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }
                    regr = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(
                        16,8), early_stopping=True, batch_size=512).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[new_target_name[0] + '_delta_nn_16_8_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }
                    regr = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(
                        16,), early_stopping=True, batch_size=512).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[new_target_name[0] + '_delta_nn_16_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }
                    regr = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(
                        14,7), early_stopping=True, batch_size=512).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[new_target_name[0] + '_delta_nn_14_7_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }
                    regr = MLPClassifier(random_state=3, max_iter=500, hidden_layer_sizes=(
                        14,), early_stopping=True, batch_size=512).fit(X_train, train_target)
                    res = regr.predict_proba(X_test)
                    m = tf.keras.metrics.CategoricalAccuracy()
                    m.update_state(test_target, res)
                    acc_score = m.result().numpy()
                    print(acc_score)
                    accuracy_dict[new_target_name[0] + '_delta_nn_14_FGSD_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_ACC': 0,
                        'train_ACC': 0,
                        'val_ACC': 0,
                        'test_ACC': acc_score
                    }


                print(pd.DataFrame(accuracy_dict))
#pd.set_option('display.max_columns', None)
#print(pd.DataFrame(accuracy_dict))
pd.DataFrame(accuracy_dict).to_csv("metrics_delta_FGSD.csv")
