"""
    Some assumptions:
     rectangle is written as (left,top,right,bottom)
     e.g. (10,5,40,15)
"""

import numpy
from util import register

pm = dict()

def find_next_obj(arr, n):
    """
    Finds where the information about next image starts in matrix arr
    
    """
    
    obj = arr[n][0]
    v = arr.shape[0]
    for i in range(n+1,arr.shape[0]):
        if obj != arr[i][0]:
            v = i
            break
    return v

def voc_detection(out, lab):
    """
      Computes Accuracy for VOC detection task.
      input:
        out:
         matrix of vectors consisted of
         image,label,left,top,right,bottom
        lab:
         matrix of vectors consisted of
         image,label,left,top,right,bottom
    """
    
    # initialize counters
    n0,m0,n1,m1 = 0,0,0,0
    preds, correct = 0,0
    
    # sort items by first column
#    print out
    out=out[out[:,0].argsort(),:]
    lab=lab[lab[:,0].argsort(),:]

    # loop through all images
    while 1:
        # get part of matrix related to current image
        n1 = find_next_obj(lab,n0)
        N = lab[range(n0,n1)]

        # get part for the same image in result matrix
        if m0 < out.shape[0] and out[m0][0] == lab[n0][0]:
            m1 = find_next_obj(out,m0)
        else:
            # if there is no such image in predictions just continue
            m1 = m0
        M = out[range(m0,m1)]

        # compare predictions to ground truth
        c,p = compare_objects(N,M)
        
        #update counters
        preds += p
        correct += c
        if n1 >= lab.shape[0]:
            break
        n0=n1
        m0=m1
    if (preds <= 0):
        return 0
    
    # return ratio of true predictions
    res = float(correct)/float(preds)
    return res 

def rectangle_area(r):
    """
        Calculates area of rectangle r
    """
    if r[2] < r[0]:
        return 0
    if r[3] < r[1]:
        return 0
    return (r[2]-r[0])*(r[3]-r[1])
        
def intersection_area(r1,r2):
    """
        Calculates the area of intersection r1*r2
    """
    r = range(0,4)
    r[0] = max(r1[0],r2[0])
    r[1] = max(r1[1],r2[1])
    r[2] = min(r1[2],r2[2])
    r[3] = min(r1[3],r2[3])
    return rectangle_area(r)

def compare_rectangles(r1,r2):
    """
        Calculates the prediction factor according to VOC definition:
        |intersection(r1,r2)| / |sum(r1,r2)|
    """
    s1 = rectangle_area(r1)
    s2 = rectangle_area(r2)
    inr_area = intersection_area(r1,r2)
    sum_area = s1 + s2 - inr_area
    if sum_area <= 0:
        return None
    return float(inr_area)/float(sum_area)

def vec_str2int(mat):
    """
        Converts matrix of ints as strings into vector of ints
    """
    s = mat.shape[0]
    r = range(0,s)
    for i in range(0,s):
        r[i] = int(mat[i])
    return r
        
def compare_objects(N,M):
    """
        Compares predictions on an objects, given by a matrix
    """
    # number of tries should not exceed number of true rectangles
    # on the other hand no predictions are treated as bad predictions
    preds = max(M.shape[0],N.shape[0])
    correct = 0
   
    # if no predictions just stop
    if M.shape[0] == 0:
        return correct, preds 
    
    # for each prediction find corresponding ground truth. If found
    # check the correctness
    for j in range(0,M.shape[0]):
        rmax = 0
        for i in range(0,N.shape[0]):
            # check if we are seeing the same object
            if N[i][1] == M[j][1]:
                # if so, compare the areas
                v1 = vec_str2int(N[i][range(2,6)])
                v2 = vec_str2int(M[j][range(2,6)])
                r = compare_rectangles(v1,v2)
                
                # store the best result
                if r > rmax:
                    rmax = r
        # if there was one correct solution than update counter
        if rmax > 0.5:
            correct += 1
            
    return correct, preds

register(pm, 'VOC detection', voc_detection)