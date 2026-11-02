"""Interpolate estimated feedback onto the checkpoint's own sampling clock."""
import math
from geometry import q_normalize


def interpolate_feedback(a,b,stamp):
    ta,pa,qa,va=a;tb,pb,qb,vb=b
    if not ta<=stamp<=tb or tb<=ta:raise ValueError('feedback interpolation needs a bracket')
    f=(stamp-ta)/(tb-ta)
    qa=q_normalize(qa);qb=q_normalize(qb)
    dot=sum(x*y for x,y in zip(qa,qb))
    if dot<0:qb=tuple(-x for x in qb);dot=-dot
    theta=math.acos(min(1.,dot))
    if dot>.9995:q=q_normalize(tuple((1-f)*x+f*y for x,y in zip(qa,qb)))
    else:q=tuple((math.sin((1-f)*theta)*x+math.sin(f*theta)*y)/math.sin(theta) for x,y in zip(qa,qb))
    return ([x+(y-x)*f for x,y in zip(pa,pb)],q,[x+(y-x)*f for x,y in zip(va,vb)])
