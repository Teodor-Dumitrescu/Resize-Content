import sys
import numpy as np
import pdb


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    path = [(line, col)]
    for i in range(E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = path[-1][1] + opt
        path.append((line, col))

    return path


def select_greedy_path(E):
    # for the first line, we pick the pixel with the smallest energy
    line = 0
    col = np.argmin(E[line, :])

    path = [(line, col)]

    # we choose each of the other pixels by selecting the pixel with the minimum energy out of at most 3 options
    # (directly under the previous pixel chosen)
    for i in range(1, E.shape[0]):
        line = i
        col_prev = path[-1][1]

        options = []

        # if the last pixel chosen is not the first on it's line we consider the pixel to the bottom-left of it
        if col_prev > 0:
            options.append((E[line, col_prev], -1))

        # we consider the pixel directly under the last pixel chosen anyway
        options.append((E[line, col_prev], 0))

        # if the last pixel chosen is not the last on it's line we consider the pixel to the bottom-right of it
        if col_prev < E.shape[1] - 1:
            options.append((E[line, col_prev + 1], 1))

        # pick the pixel with the smallest energy
        choice = min(options, key=lambda t: t[0])
        col = col_prev + choice[1]

        # append it's coordinates to the path
        path.append((line, col))

    return path


def select_dynamic_programming_path(E):
    # create a shortest paths matrix: each element is the cost of the shortest path in the energy matrix E that ends
    # in the coordinates of that element
    S = np.zeros(E.shape)
    S[0, :] = E[0, :]

    for i in range(1, E.shape[0]):
        for j in range(E.shape[1]):
            options = []
            # if the current element is not the first on it's line we consider the element to the top-left of it
            if j > 0:
                options.append(S[i-1, j-1])

            # we consider the element to the top of the current element anyway
            options.append(S[i-1, j])

            # if the current element is not the last on it's line we consider the element to the top-right of it
            if j < E.shape[1] - 1:
                options.append(S[i-1, j+1])

            # we add the smallest element considered to the shortest path
            S[i, j] = E[i, j] + min(options)

    # get the coordinates of the last element in the shortest path from the first to the last line in the energy matrix
    # and append it to the path
    line = E.shape[0] - 1
    col = np.argmin(S[line, :])
    path = [(line, col)]

    # reconstruct the shortest path starting from the last element
    for i in range(line - 1, -1, -1):
        # line of current element
        line = i
        # column of previous element
        prev_col = path[-1][1]

        # consider the element directly above the last element as the candidate for the shortest path
        candidate = S[line, prev_col]
        poz_candidate = (line, prev_col)

        # if the last element wasn't on the first column and if there is a shortest path through the top-left neighbour
        # of the last element we change the candidate
        if prev_col > 0 and candidate > S[line, prev_col - 1]:
            candidate = S[line, prev_col - 1]
            poz_candidate = (line, prev_col - 1)

        # if the last element wasn't on the last column and if there is a shortest path through the top-right neighbour
        # of the last element we change the candidate
        if prev_col < S.shape[1] - 1 and candidate > S[line, prev_col + 1]:
            candidate = S[line, prev_col + 1]
            poz_candidate = (line, prev_col + 1)

        path.append(poz_candidate)

    # reverse the shortest path so we have it in correct order
    path.reverse()

    return path


def select_path(E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'greedy':
        return select_greedy_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_programming_path(E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)