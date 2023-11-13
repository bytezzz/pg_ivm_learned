import time
from multiprocessing import Process, Barrier, Array
from transactions import *
import itertools
import numpy as np
import pandas as pd
import os

workload = [shipment, good_receive, update_score, adjust_discount]
max_connection = 4
ratio = [0.25, 0.25, 0.25, 0.25]

workloads = list(itertools.chain(*[[workload[i]]*int(max_connection*percentage) for i, percentage in enumerate(ratio)]))

def save_result(result, filename):
    result.loc['mean'] = result.mean()
    result.to_csv(filename)


def worker(id, time_costs):
    conn = get_connection()
    time_before = time.time()
    try:
        workloads[id](conn)
    except psycopg.errors.DeadlockDetected:
        print(f'Process {id} has been detected deadlock')
        conn.rollback()
        time_costs[id] = -1
        conn.close()
        return
    time_after = time.time()
    print("Process {} finished".format(id))
    time_costs[id] = time_after - time_before
    conn.close()

if __name__ == '__main__':

    reptead_times = 2

    filename = time.strftime("%Y_%m_%d_%H_%M.log")
    workload_name = [f"{workload.__name__}#{i}" for i,workload in enumerate(workloads)]
    result = pd.DataFrame(columns = workload_name+['average_time_cost'])

    try:
        for i in range(reptead_times):
            print(f'Testing the {i+1}th time')

            time_costs = Array('d', [0.0]*max_connection)
            processes = [Process(target=worker, args=(i, time_costs)) for i in range(max_connection)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            result.loc[len(result)] = [*time_costs, np.mean(time_costs)]
    except Exception as e:
        print(e)
    finally:
        save_result(result, filename)


