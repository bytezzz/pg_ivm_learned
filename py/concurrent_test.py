import time
import multiprocessing as mp
from transactions import *
from typing import List, Callable
import threading
from alive_progress import alive_bar
import zmq
import os
import numpy as np

workload = [shipment, good_receive, update_score, adjust_discount]


def master(conn: psycopg.Connection, queue: mp.Queue):
    while True:
        signal = queue.get()
        if signal == "shutdown":
            # print("Trying to close the connection")
            try:
                conn.cancel()
            except Exception as e:
                print(f"Master thread has been detected exception: {e}")
            finally:
                conn.close()
                break


def slave(
    conn: psycopg.Connection, workloads: List[Callable], id: int, time_costs: mp.Array
):
    time_before = time.time()

    try:
        for i, workload in enumerate(workloads):
            workload(conn)
    except psycopg.errors.DeadlockDetected:
        print(f"Process {id} has been detected deadlock")
        conn.rollback()
        time_costs[id] = -1
        return
    except Exception as e:
        print(f"Process {id} has been detected exception: {e}")
        if "connection pointer is NULL" in str(e):
            print(f"Worker {id} finished!")
        time_costs[id] = -1
        return

    time_after = time.time()
    time_costs[id] = time_after - time_before


def worker(workloads: List[Callable], time_costs: mp.Array, queue: mp.Queue, id: int):
    conn = get_connection()
    master_thread = threading.Thread(target=master, args=(conn, queue))
    slave_thread = threading.Thread(
        target=slave, args=(conn, workloads, id, time_costs)
    )

    try:
        master_thread.start()
        slave_thread.start()
        slave_thread.join()
        if master_thread.is_alive():
            queue.put("shutdown")
        master_thread.join()
    except Exception as e:
        print(f"Worker {id} has been detected exception: {e}")


def broadcaster(from_queue: mp.Queue, to_queues: List[mp.Queue]):
    while True:
        signal = from_queue.get()
        for queue in to_queues:
            queue.put(signal)
        break


def run_batch(
    workloads_map: List[List[Callable]], time_costs: mp.Array, control_queue: mp.Queue
):
    assert len(time_costs) == len(workloads_map)

    queues = [mp.Queue() for _ in range(len(workloads_map))]
    broadcast_threading = threading.Thread(
        target=broadcaster, args=(control_queue, [queue for queue in queues])
    )
    broadcast_threading.start()

    processes = [
        mp.Process(target=worker, args=(workloads_map[i], time_costs, queues[i], i))
        for i in range(len(workloads_map))
    ]

    try:
        for p in processes:
            p.start()

        for p in processes:
            p.join()

    except Exception as e:
        for p in processes:
            p.kill()

    if broadcast_threading.is_alive():
        control_queue.put("shutdown")

    broadcast_threading.join()


class BatchRunner:
    def __init__(self, workload_map: List[List[Callable]]):
        self.control_queue = mp.Queue()
        self.time_costs = mp.Array("d", len(workload_map))
        self.bootstrap_process = mp.Process(
            target=run_batch, args=(workload_map, self.time_costs, self.control_queue)
        )

    def start(self) -> None:
        self.bootstrap_process.start()

    def get_result(self) -> List[float]:
        self.bootstrap_process.join()
        return list(self.time_costs)

    def kill(self) -> None:
        if not self.bootstrap_process.is_alive():
            return

        self.control_queue.put("shutdown")

        with alive_bar(
            title="Waiting unfinished backend processes to exit",
            spinner="classic",
            bar=None,
            monitor=False,
            elapsed=True,
            stats=False,
        ):
            self.bootstrap_process.join()
            while not os.system('ps -ef | grep "[p]ostgres: vscode" > /dev/null'):
                time.sleep(0.5)


if __name__ == "__main__":
    time_cost = []
    for i in range(20):
        wl = [[shipment]]
        runner = BatchRunner(wl * 8)
        runner.start()
        result = runner.get_result()
        time_cost.extend(result)
        print("Epoch average cost", np.mean(result))
    print("Total Average Cost:", np.mean(time_cost))
