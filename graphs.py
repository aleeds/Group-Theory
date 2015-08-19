from collections import deque


def GraphEquality(x, y):
    return all([(i, q) in y or (q, i) in y for (i, q) in x])


def PathConnected(i, q, graph):
    vis = [i]
    que = deque([i])
    vis_pre = 0
    vis_size = 1
    while (q not in vis and len(que) > 0):
        a = que.popleft()
        for (j, k) in graph:
            if (k not in vis and j == a and que.count(j) == 0):
                que.append(k)
                vis.append(j)
                if (k == q):
                    return True
            if (j not in vis and k == a and que.count(k) == 0):
                que.append(j)
                vis.append(k)
                if (i == q):
                    return True
    return q in vis
