from multiprocessing import Process
from timed import timed

@timed
def run_process():
    p = Process(target=lambda: None)
    p.start()
    p.join()

if __name__ == "__main__":
    run_process()
