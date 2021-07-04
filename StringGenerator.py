import random
import string

def stringGenerator(len: int):
    return ''.join(random.choices(population=string.ascii_letters, k=len))
