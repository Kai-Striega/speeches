from multiprocessing.pool import ThreadPool
from time import sleep
NAMES = ["Kai", "Tessa", "Jess", "Lucy"]

def say_hello(name):
    sleep(10)
    print(f"Hi {name}!")

def main():
    with ThreadPool(4) as p:
        p.map(say_hello, NAMES)

if __name__ == "__main__":
    main()
