from concurrent.futures import ThreadPoolExecutor
from time import sleep
NAMES = ["Kai", "Tessa", "Jess", "Lucy"]

def say_hello(name):
    sleep(3)
    print(f"Hi {name}!")

def main():
    with ThreadPoolExecutor(3) as p:
        p.map(say_hello, NAMES)

if __name__ == "__main__":
    main()
