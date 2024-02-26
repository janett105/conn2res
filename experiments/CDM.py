# ContextDecisionMaking

# input : N_RUNS, TASKS, ALPHAS, ACT_FUNCTION, CLASS_METRICS, 
#         connectivity(+sub_id)/rsn file, input/output node(brain region)
# output : dataset(run==0), 
#          train/test reservoir state(run==0 & alpha==1) 
#          decision function, prediction/ground truth(run==0 & alpha==1) 
#          특정 subj의 connectiviy를 사용한 readout model의 성능 csv file
#               {activation, run, alpha, metric1점수, metric2점수, ...}

import warnings
import os
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
import matplotlib.pyplot as plt
import seaborn as sns
from conn2res.tasks import NeuroGymTask
from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout
from conn2res import readout, plotting

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ~/_conn2res
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# ~/_conn2res/experiments/figs 
OUTPUT_DIR = os.path.join(PROJ_DIR, "experiments", 'figs')
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# 각 task에서 각 activation func마다 실행하는 run 횟수(1 run = 1000 trials)
N_RUNS = 3 
TASKS = [
    'ContextDecisionMaking',
]
# readout model 성능 평가 기준
CLASS_METRICS = [
    'balanced_accuracy_score',
    'f1_score',
]
ALPHAS = np.linspace(0, 2, 41)[1:]
ACT_FCNS = [
    'tanh',
    'sigmoid',
    'relu',
]
# # connectivity matrix heatmap 시각화 여부
# VIZ_CONN = False


# individual일 시, id:0~69
conn = Conn(subj_id=0, filename='consensus_human_250.npy')  
# weight를 [0,1]로 scale후 spectral radius로 normalization
conn.scale_and_normalize()  
if conn.w.ndim == 3:
    print(f'\nConnectivity Matrix : Individual, ROI수 = {conn.n_nodes}')
else:
    print(f'\nConnectivity Matrix : Consensus, ROI수 = {conn.n_nodes}')

print(f'\nTASKS : {TASKS}')

# # conenctivity matrix heatmap
# if VIZ_CONN == True:
#     w_log=np.log(conn.w)
#     w_log=np.nan_to_num(w_log, neginf=-12.5)

#     print(np.min(w_log), np.max(w_log))

#     fig,ax1=plt.subplots()
#     w_show=ax1.imshow(w_log, cmap='turbo', interpolation= 'nearest')
#     cbar=ax1.figure.colorbar(w_show, ax=ax1)
#     #sns.jointplot(data=w_log, kind='scatter')
#     plt.show()

for task_name in TASKS:

    print(f'\n---------------TASK: {task_name}---------------')

    OUTPUT_DIR = os.path.join(PROJ_DIR, 'experiments', 'figs', task_name)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    task = NeuroGymTask(name=task_name)

    print(f'\nActivation functions : {ACT_FCNS}')
   
    df_subj = []
    for activation in ACT_FCNS:

        print(f'\n------ activation function = {activation} ------')

        print(f'run : {N_RUNS}번 \nalpha({len(ALPHAS)}개) : {ALPHAS}')

        esn = EchoStateNetwork(w=conn.w, activation_function=activation)
        df_runs = []
        for run in range(N_RUNS):

            print(f'\n\t\t--- run = {run} ---')

            x, y = task.fetch_data(n_trials=1000)
            x_train, x_test, y_train, y_test = readout.train_test_split(x, y)
            # dataset 시각화
            if run == 0:
                plotting.plot_iodata(
                    x, y, title=task.name, 
                    rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                    show=False,
                    savefig=True, 
                    fname=os.path.join(OUTPUT_DIR, f'io_{task.name}_{activation}')
                )
        
            input_nodes = conn.get_nodes(
                    'random', nodes_from=conn.get_nodes('VIS', filename='rsn_human_250.npy'),
                    n_nodes=task.n_features
                )
            output_nodes = conn.get_nodes('SM', filename='rsn_human_250.npy')

            # input node와 input layer간의 connection matrix 초기화
            w_in = np.zeros((task.n_features, conn.n_nodes))
            w_in[:, input_nodes] = np.eye(task.n_features)

            readout_module = Readout(estimator=readout.select_model(y))
            metrics = CLASS_METRICS

            df_alpha = []
            for alpha in ALPHAS:

                print(f'\n\t\t\t----- alpha = {alpha} -----')

                esn.w = alpha * conn.w

                rs_train = esn.simulate(
                    ext_input=x_train, w_in=w_in, input_gain=1,
                    output_nodes=output_nodes
                )
                rs_test = esn.simulate(
                    ext_input=x_test, w_in=w_in, input_gain=1,
                    output_nodes=output_nodes
                )
                
                # reservoir state 시각화
                if run == 0 and alpha == 1.0:
                    plotting.plot_reservoir_states(
                        x=x_train, reservoir_states=rs_train,
                        title=task.name,
                        savefig=True,
                        fname=os.path.join(OUTPUT_DIR, f'res_states_train_{task.name}_{activation}'),
                        rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                        show=False
                    )
                    plotting.plot_reservoir_states(
                        x=x_test, reservoir_states=rs_test,
                        title=task.name,
                        savefig=True,
                        fname=os.path.join(OUTPUT_DIR, f'res_states_test_{task.name}_{activation}'),
                        rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                        show=False
                    )

                df_res = readout_module.run_task(
                        X=(rs_train, rs_test), y=(y_train, y_test),
                        sample_weight='both', metric=metrics,
                        readout_modules=None, readout_nodes=None,
                    )
            
                # alpha 1개 끝난 후
                # df_res : 특정 alpha에서의 model 성능
                # {alpha(1개), metric1점수, metric2점수, ...} 
                df_res['alpha'] = np.round(alpha, 3)
                df_alpha.append(df_res)

                # visualize diagnostic curves
                if run == 0 and alpha == 1.0 and is_classifier(readout_module.model):
                    plotting.plot_diagnostics(
                        x=x_train, y=y_train, reservoir_states=rs_train,
                        trained_model=readout_module.model, title=task.name,
                        savefig=True,
                        fname=os.path.join(OUTPUT_DIR, f'diag_train_{task.name}_{activation}'),
                        rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                        show=False
                    )
                    plotting.plot_diagnostics(
                        x=x_test, y=y_test, reservoir_states=rs_test,
                        trained_model=readout_module.model, title=task.name,
                        savefig=True,
                        fname=os.path.join(OUTPUT_DIR, f'diag_test_{task.name}_{activation}'),
                        rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                        show=False
                    )


            # alpha가 모두 끝난 후
            # df_alpha : 특정 dataset(run)에서, 전체 alpha에 따른 model 성능
            # {run(1개), alpha, metric1점수, metric2점수, ...}
            df_alpha = pd.concat(df_alpha, ignore_index=True)
            df_alpha['run'] = run
            df_runs.append(df_alpha)

        # run(dataset)이 모두 끝난 후
        # df_runs : 특정 actv func에서, 전체 dataset(runs)에 따른 model 성능
        # {activation(1개), run, alpha, metric1점수, metric2점수, ...}
        df_runs = pd.concat(df_runs, ignore_index=True)
        df_runs['activation'] = activation
        if 'module' in df_runs.columns:
            df_subj.append(
                df_runs[['module', 'n_nodes', 'activation', 'run', 'alpha']
                        + metrics]
            )
        else:
            df_subj.append(df_runs[['activation', 'run', 'alpha'] + metrics])
    
    # activation func이 모두 끝난 후
    # df_subj : 특정 subject에 대한 model 성능
    # {activation, run, alpha, metric1점수, metric2점수, ...}
    df_subj = pd.concat(df_subj, ignore_index=True)
    # csv file로 저장
    df_subj.to_csv(
        os.path.join(OUTPUT_DIR, f'results_{task.name}.csv'),
        index=False
        )
    
    # # 결과 불러오기
    # df_subj = pd.read_csv(
    #     os.path.join(OUTPUT_DIR, f'results_{task.name}.csv'),
    #     index_col=False
    #     )

    # alpha값에 따른 성능 시각화
    for metric in metrics:
        plotting.plot_performance(
            df_subj, x='alpha', y=metric, hue='activation',
            title=task.name, savefig=True,
            fname=os.path.join(OUTPUT_DIR, f'perf_{task.name}_{metric}'),
            rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
            show=False
        )