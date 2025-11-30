from multiprocessing import Pool
from time import sleep
NAMES = ["Kai", "Tessa", "Jess", "Lucy"]

def say_hello(name):
    sleep(10)
    print(f"Hi {name}!")

def main():
    with Pool() as pool:
        pool.map(say_hello, NAMES)

if __name__ == "__main__":
    main()
