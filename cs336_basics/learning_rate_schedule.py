import math

def learning_rate_schedule(step,max_lr, min_lr, tw,tc):
    if step<tw:
        return  max_lr*step/tw
    if tw <= step <=tc:
        return min_lr+ 0.5*(1+math.cos((step-tw)*math.pi/(tc-tw)))*(max_lr-min_lr)
    if step>tc:
        return min_lr
    return None

