from django.shortcuts import render
from .models import Topic

# Create your views here.
def home(request):
    topics = Topic.objects.all()
    print(topics)
    context = {"topics": topics} 
    return render(request, 'home.html', context)

def about(request):
    pts =[ "Linear algebra library in Python",
            "Used for performing mathematical and logical operations on Array",
            "Provides features for operations on multi-dimensional arrays and matrices in Python",
            "Object defined in NumPy is an N Dimensional array type called ndarray",
            "Describes the collection of items of the same type",
            "Every item in an ndarray takes the same size of block in the memory",
            "Each element in ndarray is an object of data-type object (called dtype)" ]
    
    example = """import numpy as np
    a = np.array([1,2,3])
    b = np.array((1,2,3))
    print (a) # [1 2 3] => numpy.ndarray
    print (b) # [1 2 3] => numpy.ndarray
    b[1] = 4
    print (b) # [1 4 3]    
              """
    context = {"pts":pts, "example":example} 
    return render(request, 'about.html', context)

def types(request):
    topic = "NumPy Types"
    types = [ {"heading" : "1D Array","example":"""import numpy as np
    a = np.array([1,2,3])
    print (a) # [1 2 3] => numpy.ndarray
    """},
    {"heading" : "2D Array","example":"""import numpy as np
    a = np.array([[1,2,3],[4,5,6]])
    print (a) # [1 2 3] => numpy.ndarray
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'types.html', context)

def initialization(request):
    topic = "Initialization"
    types = [ {"heading" : "Initialize an array of 'x' X  'y' dimension with 0","example":"""np.zeros((2,2))
    array([[0., 0.],
        [0., 0.]])
    """},
    {"heading" : "Arranging the numbers between x and y with an interval of z","example":"""np.arange(10,25,5)
    array([10, 15, 20])
    np.arange(10,20,2)
    array([10, 12, 14, 16, 18])
    """},
    {"heading" : "Arranging 'z' numbers between x and y","example":"""import numpy as np
    np.linspace(5,10,6)
    array([ 5.,  6.,  7.,  8.,  9., 10.])
    np.linspace(5,10,5)
    array([ 5.  ,  6.25,  7.5 ,  8.75, 10.  ])
    np.linspace(5,10,4)
    array([ 5.        ,  6.66666667,  8.33333333, 10.        ])
    """},
    {"heading" : "Filling same number in an array of dimensions x X y","example":"""np.full((2,2),5)
    array([[5, 5],
       [5, 5]])
    np.full((2,4),6)
    array([[6, 6, 6, 6],
       [6, 6, 6, 6]])
    """},
    {"heading" : "Filling random numbers in an array of dimensions x X  y","example":"""np.random.random((2,2))
    array([[0.51696879, 0.84046442],
       [0.11127967, 0.05648703]])
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'types.html', context)

def insertion(request):
    topic = "Insertion"
    types = [ {"heading" : "ndarray.shape",
               "description": "It returns a tuple consisting of array dimensions. It can also be used to resize the array.",
               "example":""">>> a.shape
    (2, 3)
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> a.shape = (3,2)
    >>> a
    array([[1, 2],
           [3, 4],
           [5, 6]])
    print(a.shape[0]) #Row size
    3
    print(a.shape[1]) #Column size
    2
    """},
    {"heading" : "ndarray.size",
     "description": "It returns the count of number of elements in an array.",
     "example":"""import numpy as p
    a = np.arange(24)
    print(a)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
    print(a.size)
    24
    """},
    {"heading" : "ndarray.ndim",
     "description": "It returns dimension of the array.",
     "example":"""a = np.arange(24)
    print(a.ndim)
    1
    """},
    {"heading" : "ndarray.dtype",
     "description": "It returns type of each element in the array.",
     "example":"""a.dtype
    dtype('int32')
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'typeswithdesc.html', context)

def maths(request):
    topic = "Numpy Array Mathematics"
    types = [ {"heading" : "Addition","example":"""np.sum(a)
    15
    np.sum([a,b])
    20
    >>> a   #[ 5, 10]
    >>> b  #[2, 3]
    np.sum([a,b], axis = 0)
    array([ 7, 13]) #Column wise addition
    np.sum([a,b], axis = 1)
    array([15,  5])  #Row wise addition
    """},
    {"heading" : "Subtraction","example":""">>> b = [2,3]
    >>> np.subtract(a,b)
    array([3, 7])
    """},
    {"heading" : "Division","example":"""print(np.divide(a,b))
    [2.5        3.33333333]
    """},
    {"heading" : "Multiplication","example":"""print(np.multiply(a,b))
    [10 30]
    """},
    {"heading" : "Exponent","example":"""np.exp(a) #e^a
    """},
    {"heading" : "Square root","example":"""np.sqrt(a)
    """},
    {"heading" : "Sin","example":"""np.sin(a)
    """},
    {"heading" : "Cos","example":"""np.cos(a)
    """},
    {"heading" : "Log","example":"""np.log(a)
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'types.html', context)

def array_comparison(request):
    topic = "Array Comparison"
    types = [ {"heading" : "Element-wise comparison","example":""">>> a = [1,2,3]
    >>> b = [3,4,3]
    >>> np.equal(a,b)
    array([False, False,  True])
    """},
    {"heading" : "Array-wise comparison","example":"""np.array_equal(a,b)
    False
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'types.html', context)

def aggregate(request):
    topic = "Aggregate functions"
    types = [ {"example":""">>> import numpy as np
    >>> a = [1,2,4]
    >>> b = [2,4,4]
    >>> c = [1,2,4]
    >>> print(np.sum(a)) #Array wise sum
    7
    >>> print(np.min(a)) #Min of an array
    1
    >>> print(np.mean(a)) #Mean of the array
    2.3333333333333335
    >>> print(np.median(a)) #Median of the array
    2.0
    >>> print(np.corrcoef(a)) #Correlation coefficient of array
    1.0
    >>> print(np.std(a)) #Standard deviation of array
    1.247219128924647    
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'types.html', context)

def broadcast(request):
    topic = "Numpy Broadcasting"
    types = [ {"description" : "The second matrix is stretched with same elements for performing operation with extra rows in first matrix.","example":"""a = np.array([[1,2,3],[4,5,6]]) #2x3 Matrix
    b = np.array([3,4,5]) #1x3 Matrix
    print(a)
    [[1 2 3]
     [4 5 6]]
    print(b)
    [3 4 5]
    print(np.sum([a,b]))
    [[ 4  6  8]
     [ 7  9 11]]
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'typeswithdesc.html', context)

def indexslice(request):
    topic = "Indexing and slicing in Python"
    types = [ {"heading" : "Indexing",
               "description": "Index refers to a position.",
               "example":"""0    1    2    3    4    5    6    7    8    9    10
                H    E    L    L    O         W    O    R    L    D
               -11    -10    -9    -8    -7    -6    -5    -4    -3    -2    -1
               
               [6:10]= WORLD
               [-11:-7]=HELLO
    """},
    {"heading" : "Slicing",
     "description": "Slicing will slices the array to suitable position.",
     "example":"""a = np.array([[1,2,3],[4,5,6],[7,8,9]]) #3x3 Matrix
        print(a)
        [[1 2 3]
         [4 5 6]
         [7 8 9]]
        print(a[0]) #1st Row
        [1 2 3]
        print(a[:1]) #Extracting till row = 0 (That is 0th row)
        [[1 2 3]]
        print(a[:1,1:]) #Extracting till row = 0 (That is 0th row and the col starting from 1 till last)
        [[2 3]]
        print(a[:,1:])
        [[2 3]
         [5 6]
         [8 9]]
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'typeswithdesc.html', context)

def manipulation(request):
    topic = "Array Manipulation"
    types = [ {"heading" : "Concatenating two arrays together","example":"""a = np.array([[1,2,3],[4,5,6]])
    b = np.array([[7,8,9],[10,11,12]])
    np.concatenate([a,b],axis = 0) #Row wise Concatenation
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])
    """},
    {"heading" : "Stack arrays row-wise (Vertically)","example":""">>> a = np.array([1,2,3])
    >>> b = np.array([4,5,6])
    >>> print(np.stack((a,b), axis=1)) #V Stacking
    [[1 4]
     [2 5]
     [3 6]]
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> b = np.array([[7,8,9],[10,11,12]])
    >>> print(np.vstack((a,b))) #V Stacking
    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]
    """},
    {"heading" : "Stack arrays column-wise (Horizontally)","example":""">>> a = np.array([1,2,3])
    >>> b = np.array([4,5,6])
    >>> print(np.stack((a,b), axis=0)) #H Stacking
    [[1 2 3]
     [4 5 6]]
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> b = np.array([[7,8,9],[10,11,12]])
    >>> print(np.hstack((a,b))) #H Stacking
    [[ 1  2  3  7  8  9]
     [ 4  5  6 10 11 12]]
    """},
    {"heading" : "Combining column wise stacked array","example":"""np.concatenate([a,b],axis = 1) #Column wise Concatenation
    array([[ 1,  2,  3,  7,  8,  9],
           [ 4,  5,  6, 10, 11, 12]])
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'types.html', context)

def split(request):
    topic = "Splitting of arrays"
    types = [ {"description" : """np.split(array, index, axis)
    array => Any NumPy array
    index => int/list
    axis => 0 (Row)/1 (Column)
    ""","example":""">>> a = np.array([1,2,3])
    >>> b = np.array([4,5,6])
    
    >>> print(np.split(a,2,axis=0))
    [array([[1, 2, 3]]), array([[4, 5, 6]])]
    >>> print(np.split(a,3,axis=1))
    [array([[1],
           [4]]), array([[2],
           [5]]), array([[3],
           [6]])]
    >>> print(np.split(a,[1,2],axis=0))
    [array([[1, 2, 3]]), array([[4, 5, 6]]), array([], shape=(0, 3), dtype=int32)]
    >>> a[:1]
    array([[1, 2, 3]])
    >>> a[1:2]
    array([[4, 5, 6]])
    >>> a[2:]
    array([], shape=(0, 3), dtype=int32)
    
    >>> print(np.split(a,[1,2],axis=1))
    [array([[1],
           [4]]), array([[2],
           [5]]), array([[3],
           [6]])]
    >>> a[:,:1]
    array([[1],
           [4]])
    >>> a[:,1:2]
    array([[2],
           [5]])
    >>> a[:,2:]
    array([[3],
           [6]])
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'typeswithdesc.html', context)

def advantage(request):
    topic = "Advantages of NumPy"
    types = [ {"heading" : "Consumes less memory","example":""">>> #Memory Size
    >>> import numpy as np
    >>> import sys
    >>> #define a list
    >>> l = range(1000)
    >>> print("Size of a list: ",sys.getsizeof(l)*len(l))
    Size of a list:  48000
    >>> a = np.arange(1000)
    >>> print("Size of an array: ",a.size*a.itemsize)
    Size of an array:  4000
    """},
    {"heading" : "Faster","example":""">>> #Numpy vs list: Speed
    >>> import time
    >>> def using_List():
        t1 = time.time() #Starting/Initial time
        X = range(1000000)
        Y = range(1000000)
        z = [X[i]+Y[i] for i in range(len(X))]
        return time.time()-t1
    
    >>> def using_Numpy():
        t1 = time.time() #Starting/Initial time
        a = np.arange(1000000)
        b = np.arange(1000000)
        z = a + b #More Convinent than  a list
        return time.time()-t1
    >>> list_time = using_List()
    >>> numpy_time = using_Numpy()
    >>> print(list_time,numpy_time)
    0.38877415657043457 0.00799560546875
    >>> print("In this example Numpy is "+str(list_time/numpy_time)+" times faster than a list")
    In this example Numpy is 48.62347924618321 times faster than a list
    """},
    {"heading" : "More Convenient","example":"""List:        z = [X[i]+Y[i] for i in range(len(X))]
    NumPy:         z = a + b                 #More Convinent than  a list
    """}
        ]
    context = {"topic":topic,"types":types} 
    return render(request, 'types.html', context)