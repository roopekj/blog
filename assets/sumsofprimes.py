from math import ceil

import matplotlib.pyplot as plt


def is_prime(n: int) -> bool:
    for c in range(2, ceil(n / 2)):
        if not n % c:
            return False

    return True


maxnum = 10000
primes = [2]
distances = []
curdist = 0
for n in range(3, maxnum, 2):
    if is_prime(n):
        for i, p1 in enumerate(primes):
            for j in range(i):
                p2 = primes[j]
                if p1 + p2 == n:
                    print(f"{p1} + {p2} = {n}")
                    distances.append(curdist)
                    curdist = 1
                curdist += 1

        primes.append(n)

plt.plot(distances)
plt.show()
