# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

from collections import deque


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
        "extra_credit": extra_credit,
    }.get(searchMethod)(maze)


# h
def manhattan(end, current):
    dist = abs(end[0] - current[0]) + abs(end[1] - current[1])
    return dist


# g
def actual_cost(current, start, parent_dic):
    dist = 0
    while not (current[0] == start[0] and current[1] == start[1]):
        current = parent_dic[current]
        dist += 1
    return dist


# f
def total_cost(current, start, end, parent_dic):
    path_length = manhattan(end, current) + actual_cost(current, start, parent_dic)
    return path_length


def create_mst(mst_vals, edge_num):
    # accept a dictionary with keys as edges(two nodes tuple) and edge weights as value
    # return a list of edges(two nodes tuple) and total edge weight in MST
    edge_list = mst_vals.keys()
    edge_list = sorted(edge_list, key=lambda x: mst_vals[x], reverse=False)
    # sort the edge according to their weights in increasing order

    edge_count = 0
    disjoint_sets = []
    mst_edges = []

    for edge in edge_list:
        valid_edge = 1
        new_set = 1
        if edge_count >= edge_num:
            break
        for nodes in disjoint_sets:
            if (edge[0] in nodes) and (edge[1] in nodes):
                valid_edge = 0
                new_set = 0
                break
            if (edge[0] in nodes) or (edge[1] in nodes):
                normal_add = 1
                for next_nodes in disjoint_sets:
                    if (nodes != next_nodes) and ((edge[0] in nodes) or (edge[1] in nodes)):
                        new_nodes = nodes.union(next_nodes)
                        disjoint_sets.remove(nodes)
                        disjoint_sets.remove(next_nodes)
                        disjoint_sets.append(new_nodes)
                        normal_add = 0
                        break
                if normal_add:
                    nodes.add(edge[0])
                    nodes.add(edge[1])
                new_set = 0
                break
        if new_set:
            disjoint_sets.append({edge[0], edge[1]})
        if valid_edge:
            mst_edges.append(edge)
            edge_count += 1

    total_edge_weight = 0
    for edge in mst_edges:
        total_edge_weight += mst_vals[edge]

    return mst_edges, total_edge_weight


def bfs(maze):
    # return path, num_states_explored
    start = maze.getStart()  # get the start position
    if maze.isObjective(start[0], start[1]):
        return start, 0
    # end = maze.getObjectives()
    # end = end[0]  # to obtain the one goal in the list
    end = None

    # Initialization (create a queue of paths to store the path for each node)
    queue = deque([start])
    explored = set()
    parents = {}
    found = False

    # when you do a while on a deque and it is not empty it is considered a true
    while queue:
        if found:
            break
        node = queue.popleft()
        # after procuring the last node we do the usual except we store the path for the node instead of just the node
        explored.add(node)
        children = maze.getNeighbors(node[0], node[1])  # get the children of each node
        children = [i for i in children if (i not in explored and i not in queue)]
        # checks if the node has been explored; if not, checks if the goal or else makes a new path to explore
        for i in children:
            parents.update({i: node})
            if maze.isObjective(i[0], i[1]):
                end = (i[0], i[1])
                found = True
            else:
                queue.append(i)

    # if we failed to find a path, return empty path
    try:
        parents[end]
    except KeyError:
        print("Exception")
        return [], 0

    path = deque([])
    while not (end[0] == start[0] and end[1] == start[1]):
        path.appendleft(end)
        end = parents[end]
    path.appendleft(start)
    return path, len(explored)


def dfs(maze):
    # return path, num_states_explored
    start = maze.getStart()  # get the start position
    if maze.isObjective(start[0], start[1]):
        return start, 0
    # end = maze.getObjectives()
    # end = end[0]  # to obtain the one goal in the list
    end = None
    # Initialization (create a stack of paths to store the path for each node)
    stack = deque([start])
    explored = set()
    parents = {}
    found = False

    # when you do a while on a deque and it is not empty it is considered a true
    while stack:
        if found:
            break
        node = stack.pop()
        # after procuring the last node we do the usual except we store the path for the node instead of just the node
        explored.add(node)
        children = maze.getNeighbors(node[0], node[1])  # get the children of each node
        children = [i for i in children if (i not in explored and i not in stack)]
        # checks if the node has been explored; if not, checks if the goal or else makes a new path to explore
        for i in children:
            parents.update({i: node})
            if maze.isObjective(i[0], i[1]):
                end = (i[0], i[1])
                found = True
            else:
                stack.append(i)

    # if we failed to find a path, return empty path
    try:
        parents[end]
    except KeyError:
        return [], 0

    path = deque([])
    while not (end[0] == start[0] and end[1] == start[1]):
        path.appendleft(end)
        end = parents[end]
    path.appendleft(start)
    return path, len(explored)


def greedy_help(end, maze):
    # return path, num_states_explored
    start = maze.getStart()  # get the start position
    if maze.isObjective(start[0], start[1]):
        return start, 0
    # end = maze.getObjectives()
    # end = end[0]

    # Initialization(create a priority_queue of paths to store the path for each node)
    prique = deque([start])
    explored = set([])
    parents = {}

    # when you do a while on a deque and it is not empty it is considered a true
    while prique:
        prique = sorted(prique, key=lambda x: manhattan(end, x), reverse=True)
        # sort in descending order so the smallest one gets popped the fastest
        node = prique.pop()
        # after procuring the last node we do the usual except we store the path for the node instead of just the node
        explored.add(node)
        # if the goal state is at top of the priority queue
        if maze.isObjective(node[0], node[1]):
            break
        children = maze.getNeighbors(node[0], node[1])  # get the children of each node
        children = [i for i in children if (i not in explored and i not in prique)]
        # checks if the node has been explored; if not, checks if the goal or else makes a new path to explore
        for i in children:
            parents.update({i: node})
            prique.append(i)

    # if we failed to find a path, return empty path
    try:
        parents[end]
    except KeyError:
        return [], 0

    path = deque([])
    while not (end[0] == start[0] and end[1] == start[1]):
        path.appendleft(end)
        end = parents[end]
    path.appendleft(start)
    return path, len(explored)


def greedy(maze):
    # return path, num_states_explored
    start = maze.getStart()
    end = maze.getObjectives()

    path, cost = greedy_help(end[0], maze)
    for i in end:
        temp_path, temp_cost = greedy_help(i, maze)
        if len(temp_path) < len(path):
            path = temp_path
            cost = temp_cost
    return path, cost


def astar_two_points(start, end, maze):
    # return path, num_states_explored
    # Initialization(create a priority_queue of paths to store the path for each node)
    prique = deque([start])  # openlist
    explored = set([])  # closelist
    parents = {}

    # when you do a while on a deque and it is not empty it is considered a true
    while prique:
        prique = sorted(prique, key=lambda x: total_cost(x, start, end, parents), reverse=True)
        # sort in descending order so the smallest one gets popped the fastest
        node = prique.pop()
        # after procuring the last node we do the usual except we store the path for the node instead of just the node
        explored.add(node)
        # if the goal state is at top of the priority queue
        if node == end:
            break
        children = maze.getNeighbors(node[0], node[1])  # get the children of each node
        children = [i for i in children if i not in explored]
        for i in children:
            if i in prique:
                pre_distance = actual_cost(i, start, parents)
                new_distance = actual_cost(node, start, parents) + 1
                if pre_distance < new_distance:
                    children.remove(i)
                else:
                    prique.remove(i)
                    del parents[i]
        # checks if the node has been explored; if not, checks if the goal or else makes a new path to explore
        # children = sorted(children, key=lambda x: manhattan(end, x), reverse=True)
        # sort in descending order so the smallest one gets popped the fastest
        for i in children:
            parents.update({i: node})
            prique.append(i)

    # if we failed to find a path, return empty path
    try:
        parents[end]
    except KeyError:
        return [], 0

    path = deque([])
    while not (end[0] == start[0] and end[1] == start[1]):
        path.appendleft(end)
        end = parents[end]
    path.appendleft(start)
    return path, len(explored)


def astar_multiple_points(start, end, maze):
    points = deque(end)
    points.appendleft(start)
    mst_vals = {}

    while points:
        x = points.popleft()
        for i in points:
            path, cost = astar_two_points(x, i, maze)
            mst_vals.update({(x, i): len(path)})
    # print("This is the dictionary of the graph: ", mst_vals)
    mst_graph = create_mst(mst_vals, len(end) - 1)  # sending the dictionary and number of edges

    print("These are the edges in MST: ", mst_graph[0])
    print("This is the total edge weight of MST: ", mst_graph[1])
    print(len(mst_graph[0]))
    print("this is start", maze.getStart())

    return [], 0


def astar(maze):
    # return path, num_states_explored
    start = maze.getStart()
    end = maze.getObjectives()

    path,cost = astar_two_points(start,end[0],maze)
    for i in end:
        temp_path,temp_cost = astar_two_points(start,i,maze)
        if len(temp_path) < len(path):
            path = temp_path
            cost= temp_cost
    return path,cost


####################################### EXTRA CREDIT ############################

def greedy_ec(start, end, maze, objs):
    # Initialization(create a priority_queue of paths to store the path for each node)
    prique = deque([start])
    explored = set()
    parents = {}
    objs = set(objs)
    real_start = maze.getStart()
    # when you do a while on a deque and it is not empty it is considered a true
    while prique:
        prique = sorted(prique, key=lambda x: manhattan(real_start, x), reverse=True)
        # sort in descending order so the smallest one gets popped the fastest
        node = prique.pop()
        # after procuring the last node we do the usual except we store the path for the node instead of just the node
        explored.add(node)
        # if the goal state is at top of the priority queue
        if maze in objs:
            break
        children = maze.getNeighbors(node[0], node[1])  # get the children of each node
        children = [i for i in children if (i not in explored and i not in prique)]
        # checks if the node has been explored; if not, checks if the goal or else makes a new path to explore
        for i in children:
            parents.update({i: node})
            prique.append(i)

    # if we failed to find a path, return empty path
    # try:
    #     parents[end]
    # except KeyError:
    #     return [], 0

    path = deque([])
    while not (end[0] == start[0] and end[1] == start[1]):
        path.appendleft(end)
        end = parents[end]
    path.appendleft(start)
    return path, explored


def extra_credit(maze):
    end = maze.getObjectives()
    points = deque(end)
    start = maze.getStart()
    length = len(end)
    path = []
    sum_path = 0
    explored = set(start)
    while points:
        points = sorted(points, key=lambda x: manhattan(start, x), reverse=True)

        node = points.pop()

        new_path, exp = greedy_ec(start, node, maze, points)

        path.extend(new_path)
        explored = explored | exp
        explored.union(exp)
        start = node

    return path, len(explored)