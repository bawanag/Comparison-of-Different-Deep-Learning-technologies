import numpy
import sys
a = 58682.7578125
b = 100
print(sys.getsizeof(a))
print(sys.getsizeof(b))
print (type(a))
print (a)
print (type(numpy.float32(a)))
print(sys.getsizeof(type(numpy.float32(a))))
print (numpy.float32(a))