import os
import subprocess
from optparse import OptionParser

import numpy as np
from numpy import array as npar
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


def create_training_opt_parser():
    parser = OptionParser()
    parser.add_option("-d", "--dataset_path",type=str, default='/home/anhkhoa/Vu_working/NLP/Human-Short-Seq/data/human_promoters_short.csv',
                      help="Dataset path to load (csv file)")
    parser.add_option("-s", "--save_path",type=str, default='./result',
                      help="Save path")

    parser.add_option("-n", "--model_name",type=str, default='hum_prom_short',
                      help="Available model to train")
    parser.add_option("--version_run",type=str, default='',
                      help="run th")                      

    parser.add_option("--batch_size", type=int, default=64,
                      help="batch size training")
    parser.add_option("-v", "--verbose", default=True,
                      help="don't print status messages to stdout")
    (options, args) = parser.parse_args()
    return options, args


def get_gpu_id_max_memory(acceptable_available_memory):
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_free_values = npar([int(x.split()[0]) for i, x in enumerate(memory_free_info)])
    print(memory_free_values)
    gpu_id = memory_free_values.argmax()
    if memory_free_values[gpu_id] < acceptable_available_memory:
        return -1
    print(gpu_id)
    return gpu_id

def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
def draw_graph_history(history, metrics,path, file_name):
    fig, ax1 = plt.subplots()
    fig.patch.set_facecolor('w')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    # axMETRICS_F.set_major_locator(MultipleLocator(1000))
    # ax1.yaxis.set_major_locator(MultipleLocator(10000))
    ax1.autoscale(enable=True, axis='y', tight=True)
    ax1.plot(history.history['loss'], color='r')
    ax1.plot(history.history['val_loss'], color='y')
    ax1.legend(['train_'+metrics[0], 'val_'+metrics[0]], loc='upper left')
    ax2 = ax1.twinx()
    # print(history.history.keys())
    if 'mean_absolute_error' in history.history.keys():
        ax2.plot(history.history['mean_absolute_error'], color='b')
        ax2.plot(history.history['val_mean_absolute_error'], color='c')
        ax2.legend(['train_MAE', 'val_MAE'], loc='upper right')
        ax2.set_ylabel('MAE')
    else:
        ax2.plot(history.history[metrics[0]], color='b')
        ax2.plot(history.history['val_'+metrics[0]], color='c')
        ax2.legend(['train_'+metrics[0], 'val_'+metrics[0]], loc='upper right')
        ax2.set_ylabel(metrics[0])
    # ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.autoscale(enable=True, axis='y', tight=True)
    plt.title(file_name)
    plt.savefig(os.path.join(path, file_name+'_hist.png'),bbox_inches='tight')

def draw_model(model,path,file_name):
    plot_model(model, to_file=os.path.join(path,file_name+'_model.png'))


def save_history(history, metrics, path, file_name):
    numpy_loss_history = np.array(history.history['loss'])
    numpy_loss_history = np.c_[
        np.arange(1, numpy_loss_history.size+1).astype(np.int64), numpy_loss_history]
    numpy_loss_history = np.c_[numpy_loss_history,
                               np.array(history.history['val_loss'])]
    if 'mean_absolute_error' in history.history.keys():
        numpy_loss_history = np.c_[numpy_loss_history, np.array(
            history.history['mean_absolute_error'])]
        numpy_loss_history = np.c_[numpy_loss_history, np.array(
            history.history['val_mean_absolute_error'])]
    else:
        numpy_loss_history = np.c_[
            numpy_loss_history, np.array(history.history[metrics[0]])]
        numpy_loss_history = np.c_[numpy_loss_history,
                                   np.array(history.history['val_'+metrics[0]])]
    np.savetxt(os.path.join(path, file_name+"loss_history.txt"),
               numpy_loss_history, delimiter=",", header="epoch,loss,val_loss,"+metrics[0]+",val_"+metrics[0])