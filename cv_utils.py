import numpy

def applyLowe(matches):
    lowe_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            lowe_matches.append(m)

    return lowe_matches