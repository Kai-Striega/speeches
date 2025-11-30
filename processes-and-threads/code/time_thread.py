from threading import Thread
from timed import timed

@timed
def run_thread():
    t = Thread(target=lambda: None)
    t.start()
    t.join()

if __name__ == "__main__":
    run_thread()
