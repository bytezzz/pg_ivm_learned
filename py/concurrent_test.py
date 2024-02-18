import time
from multiprocessing import Process, Barrier, Array, Value, Pipe, Queue
from transactions import *
import itertools
import numpy as np
import pandas as pd
import threading
import os
import debugpy

workload = [shipment, good_receive, update_score, adjust_discount]
max_connection = 8
ratio = [0.25, 0.25, 0.25, 0.25]


workloads = list(
    itertools.chain(
        *[
            [workload[i]] * int(max_connection * percentage)
            for i, percentage in enumerate(ratio)
        ]
    )
)


def save_result(result, filename):
    result.loc["mean"] = result.mean()
    result.to_csv(filename)


def master(conn: psycopg.Connection, queue: Queue):
    while True:
        signal = queue.get()
        if signal == "shutdown":
            #print("Trying to close the connection")
            try:
                conn.cancel()
            except Exception as e:
                print(f"Master thread has been detected exception: {e}")
            finally:
                conn.close()
                break

def slave(conn, id, time_costs):
    time_before = time.time()

    try:
        #print(f"Process {id} is working")
        workloads[id](conn)
    except psycopg.errors.DeadlockDetected:
        print(f"Process {id} has been detected deadlock")
        conn.rollback()
        time_costs[id] = -1
        return
    except Exception as e:
        #print(f"Process {id} has been detected exception: {e}")
        if "connection pointer is NULL" in str(e):
            print(f"Worker {id} finished!")
        time_costs[id] = -1
        return

    time_after = time.time()
    time_costs[id] = time_after - time_before


def worker(id, time_costs, queue):
    conn = get_connection()
    master_thread = threading.Thread(target=master, args=(conn, queue))
    slave_thread = threading.Thread(target=slave, args=(conn, id, time_costs))

    try:
        master_thread.start()
        slave_thread.start()
        slave_thread.join()
        if master_thread.is_alive():
            queue.put("shutdown")
        master_thread.join()
        #print(f"Worker {id} has finished")
    except Exception as e:
        print(f"Worker {id} has been detected exception: {e}")



def broadcast_func(from_queue, to_queue, stop_event:threading.Event):
    while True:
        signal = from_queue.get()
        print(f"Broadcasting signal {signal}")
        stop_event.set()
        for queue in to_queue:
            queue.put(signal)
        break

def run_batch(result, control_queue, stop=True):
    print("Starting the batch")
    time_costs = Array("d", [0.0] * max_connection)

    stop_event = threading.Event()

    queues = [Queue() for _ in range(max_connection)]
    broadcast_threading = threading.Thread(
        target=broadcast_func, args=(control_queue, [queue for queue in queues], stop_event)
    )

    broadcast_threading.start()

    while True:
        #print("Restartiing ")

        processes = [
            Process(target=worker, args=(i, time_costs, queues[i]))
            for i in range(max_connection)
        ]

        try:
            for p in processes:
                p.start()

            for p in processes:
                p.join()

            result.value = np.mean(time_costs)

        except Exception as e:
            for p in processes:
                p.kill()

        if stop or stop_event.is_set():
            control_queue.put("shutdown")
            print("Shutting down the batch")
            break


if __name__ == "__main__":
    control_queue = Queue()
    result = Value("d", 0.0)

    batch_threading = threading.Thread(target=run_batch, args=(result, control_queue))

    batch_threading.start()

    time.sleep(15)

    control_queue.put("shutdown")

    batch_threading.join()

    print(result.value)

if __name__ == "lll__main__":
    reptead_times = 4

    filename = time.strftime("%Y_%m_%d_%H_%M.log")
    workload_name = [f"{workload.__name__}#{i}" for i, workload in enumerate(workloads)]
    result = pd.DataFrame(columns=workload_name + ["average_time_cost"])

    try:
        for i in range(reptead_times):
            print(f"Testing the {i+1}th time")

            time_costs = Array("d", [0.0] * max_connection)
            processes = [
                Process(target=worker, args=(i, time_costs))
                for i in range(max_connection)
            ]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            result.loc[len(result)] = [*time_costs, np.mean(time_costs)]
    except Exception as e:
        print(e)
    finally:
        save_result(result, filename)
