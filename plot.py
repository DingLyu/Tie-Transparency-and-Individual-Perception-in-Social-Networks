#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ding lyu
"""
# this file includes:
# plot Fig.~1, Fig.~2, Fig.~3, Fig.~4, Fig.~S4, Fig.~S5, Fig.~S6 except the visualization of networks

from utils import *
from datapreprocessing import *

# Fig.~1(a)
# correlation between edge embeddedness and tie transparency
def Correlation_EE_KD():
    G = StandardGlobalNetwork()
    KD = KnowableDegree()
    Embeddedness = {}
    for edge in G.edges():
        a = G.degree(edge[0])
        b = G.degree(edge[1])
        c = len(list(nx.common_neighbors(G, edge[0], edge[1])))
        Embeddedness[edge] = c / (a + b - c)
    E = max(Embeddedness.values())
    X = []
    Y = []
    Err = []
    H = [[]for i in range(27)]
    for edge in G.edges():
        e = Embeddedness[edge]/E
        if edge in KD:
            if KD[edge] >= 0:
                H[KD[edge]].append(e)
        elif (edge[1], edge[0]) in KD:
            if KD[(edge[1], edge[0])] >= 0:
                H[KD[(edge[1], edge[0])]].append(e)
    print(np.std(H[0]))
    print(len(H[0]))
    print(sp.stats.sem(H[0])*sp.stats.t._ppf((1+0.95)/2., 191))
    for i in range(27):
        if H[i] != []:
            X.append(i)
            Y.append(np.mean(H[i]))
            if i < 25:
                Err.append(mean_confidence_interval(H[i], confidence=0.99))
            else:
                Err.append(np.std(H[i]))
    plt.figure(figsize=(8,5))
    plt.errorbar(x=X, y=Y, yerr=Err, fmt="o",color="steelblue",ecolor='grey',elinewidth=.5,capsize=4)
    plt.xlabel('Tie Transparency', fontsize=20)
    plt.ylabel('Edge Embeddedness', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~1(b)
# correlation between edge betweenness and tie transparency
def Correlation_EB_KD():
    G = StandardGlobalNetwork()
    KD = KnowableDegree()
    Data = []
    edge_betweeness = nx.edge_betweenness_centrality(G)
    h = max(edge_betweeness.values())
    X = []
    Y = []
    Err = []
    H = [[]for i in range(27)]
    for e in G.edges():
        a = edge_betweeness[e]/h
        if e in KD:
            if KD[e] >= 0:
                b = KD[e]
        elif (e[1], e[0]) in KD:
            if KD[(e[1], e[0])] >= 0:
                b = KD[(e[1], e[0])]
        H[b].append(a)
    for i in range(27):
        if H[i] != []:
            X.append(i)
            Y.append(np.mean(H[i]))
            if i < 25:
                Err.append(mean_confidence_interval(H[i], confidence=0.99))
            else:
                Err.append(np.std(H[i]))
    plt.figure(figsize=(8,5))
    plt.errorbar(x=X, y=Y, yerr=Err, fmt="o",color="steelblue",ecolor='grey',elinewidth=.5,capsize=4)
    plt.xlabel('Tie Transparency', fontsize=20)
    plt.ylabel('Edge Betweenness', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~1(c)
# correlation between attribute similarity and tie transparency
# to protect the privacy, we hide the attributes of nodes (name, degree, grade, department, class, and gender)
def Correlation_AS_KD():
    from data import attribute as A
    KD = KnowableDegree()
    Err = []
    H = [[]for i in range(27)]
    X = []
    Y = []
    for tie in KD:
        if KD[tie] >= 0:
            vec1 = np.array(A.A[tie[0]])
            vec2 = np.array(A.A[tie[1]])
            distance = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * (np.linalg.norm(vec2)))
            H[KD[tie]].append(distance)
    for i in range(27):
        if H[i] != []:
            X.append(i)
            Y.append(np.mean(H[i]))
            if i < 25:
                Err.append(mean_confidence_interval(H[i], confidence=0.99))
            else:
                Err.append(np.std(H[i]))
    plt.figure(figsize=(8,5))
    plt.errorbar(x=X, y=Y, yerr=Err, fmt="o",color="steelblue",ecolor='grey',elinewidth=.5,capsize=4)
    plt.xlabel('Tie Transparency', fontsize=20)
    plt.ylabel('Attribute Similarity', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~1(e)
# Make communities emerge with the removal of edges in the KD ascending order and the KD descending order
def CommunityEmerge(): #KD从小到大和从大到小去除，看最大连通片规模
    G = StandardGlobalNetwork()
    Up = list()
    up = list()
    Down = list()
    down = list()
    KD = KnowableDegree()
    for i in range(68):
        for edge in G.edges():
            if KD[edge] == i:
                down.append(edge)
            if KD[edge] == 67 - i:
                up.append(edge)
    G = StandardGlobalNetwork()
    for e in up:
        G.remove_edge(e[0], e[1])
        Gc = len(list(max(nx.connected_components(G), key=len)))
        Up.append(Gc)
    G = StandardGlobalNetwork()
    for e in down:
        G.remove_edge(e[0], e[1])
        Gc = len(list(max(nx.connected_components(G), key=len)))
        Down.append(Gc)

    X = [i for i in range(101)]
    x = [i / 100 for i in range(101)]
    UP = []
    DOWN = []
    for i in X:
        UP.append(Up[round(i * 1995 / 100)] / 155)
        DOWN.append(Down[round(i * 1995 / 100)] / 155)

    fig = plt.figure(figsize=(10, 5))
    left, bottom, width, height = 0.085, 0.13, 0.865, 0.82
    ax1 = fig.add_axes([left, bottom, width, height])
    plt.tick_params(labelsize=15)
    ax1.set_xlabel("$f$", fontsize=20)
    ax1.set_ylabel("$S(f)$", fontsize=20)
    ax1.scatter(x, UP, marker='v', color='red', alpha=1, label="KD Descending")
    ax1.scatter(x, DOWN, marker='^', color='steelblue', alpha=1, label="KD Ascending")
    plt.legend(fontsize=15)

    community = mpimg.imread('community.png')
    newcommunity = mpimg.imread('newcommunity.png')

    left, bottom, width, height = 0.1, 0.3, 0.3, 0.3
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.axis('off')
    ax2.imshow(community)
    left, bottom, width, height = 0.4, 0.3, 0.3, 0.3
    ax3 = fig.add_axes([left, bottom, width, height])
    ax3.axis('off')
    ax3.imshow(newcommunity)
    plt.show()

# Fig.~1(f)
# distributions of tie transparency distinguished by same and different separate attributes
# including degree, grade, department, class, and gender.
# to protect the privacy, we hide the attributes of nodes (name, degree, grade, department, class, and gender)
def KD_Attributes():
    from data import attribute as A
    G = StandardGlobalNetwork()
    KD = KnowableDegree()
    Data = []
    for edge in G.edges():
        if A.A[edge[0]][:3] == A.A[edge[1]][:3]: # degree
            Data.append([KD[edge], 'Degree', 'Same'])
        else:
            Data.append([KD[edge], 'Degree', 'Different'])
        if A.A[edge[0]][3:7] == A.A[edge[1]][3:7]: # grade
            Data.append([KD[edge], 'Grade', 'Same'])
        else:
            Data.append([KD[edge], 'Grade', 'Different'])
        if A.A[edge[0]][7:13] == A.A[edge[1]][7:13]: # department
            Data.append([KD[edge], 'Department', 'Same'])
        else:
            Data.append([KD[edge], 'Department', 'Different'])
        if A.A[edge[0]][13:35] == A.A[edge[1]][13:35]: # class
            Data.append([KD[edge], 'Class', 'Same'])
        else:
            Data.append([KD[edge], 'Class', 'Different'])
        if A.A[edge[0]][35:] == A.A[edge[1]][35:]: # sex
            Data.append([KD[edge], 'Gender', 'Same'])
        else:
            Data.append([KD[edge], 'Gender', 'Different'])
    plt.figure(figsize=(8,5))
    df = pd.DataFrame(data=Data, columns=['Tie Transparency', 'Attributes', 'Label'])
    sns.violinplot(data=df, x='Attributes', y='Tie Transparency', hue='Label', palette=['#2980b9','#bdc3c7']
                   ,split=True, inner="quart", linewidth=1)
    plt.xlabel('Attributes', fontsize=20)
    plt.ylabel('Tie Transparency', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    legend = plt.legend(loc='upper right', title='Label', fontsize=15)
    legend.get_title().set_fontsize(fontsize=15)
    plt.show()

# Fig.~2(a)
# Explanation of the phenominon of two degrees of perception with the distribution of perceptual proportion
def TwoDegreesofPerception():
    A = np.load('PerceptualProportion.npy')
    X = np.arange(11)
    B = [[] for x in range(7)]
    for i in range(len(A)):
        count0, count1, count2, count3, count4, count5, count6, count7, count8, count9, count10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for j in range(len(A[i])):
            if A[i][j] == 0:
                count0 += 1
            if A[i][j] <= 0.1 and A[i][j] > 0:
                count1 += 1
            if A[i][j] <= 0.2 and A[i][j] > 0.1:
                count2 += 1
            if A[i][j] <= 0.3 and A[i][j] > 0.2:
                count3 += 1
            if A[i][j] <= 0.4 and A[i][j] > 0.3:
                count4 += 1
            if A[i][j] <= 0.5 and A[i][j] > 0.4:
                count5 += 1
            if A[i][j] <= 0.6 and A[i][j] > 0.5:
                count6 += 1
            if A[i][j] <= 0.7 and A[i][j] > 0.6:
                count7 += 1
            if A[i][j] <= 0.8 and A[i][j] > 0.7:
                count8 += 1
            if A[i][j] <= 0.9 and A[i][j] > 0.8:
                count9 += 1
            if A[i][j] <= 1 and A[i][j] > 0.9:
                count10 += 1
        B[i] = [count0/len(A[i]), count1/len(A[i]), count2/len(A[i]), count3/len(A[i]), count4/len(A[i]), count5/len(A[i]), count6/len(A[i]), count7/len(A[i]), count8/len(A[i]), count9/len(A[i]), count10/len(A[i])]
    fig = plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.set_xticklabels(
        ["0", "(0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]", "(0.5,0.6]", "(0.6,0.7]", "(0.7,0.8]",
         "(0.8,0.9]", "(0.9,1]"])
    plt.xticks(np.linspace(0, 10, 11))
    plt.tick_params(labelsize=12)
    plt.yticks(fontsize=15)
    plt.xlabel("Perceptual Proportion", fontsize=25)
    plt.ylabel("f", fontsize=25)
    plt.plot(X, B[0], linestyle='-', color='#e74c3c', label='$1-1$ edge', alpha=1, marker='o')
    plt.plot(X, B[1], linestyle='-', color='#e67e22', label='$1-2$ edge', alpha=1, marker='8')
    plt.plot(X, B[2], linestyle='-', color='#f1c40f', label='$2-2$ edge', alpha=1, marker='h')
    plt.plot(X, B[3], linestyle='--', color='#1abc9c', label='$2-3$ edge', alpha=1, marker='v')
    plt.plot(X, B[4], linestyle='--', color='#2ecc71', label='$3-3$ edge', alpha=1, marker='^')
    plt.plot(X, B[5], linestyle='--', color='#3498db', label='$3-4$ edge', alpha=1, marker='<')
    plt.plot(X, B[6], linestyle='--', color='#9b59b6', label='$4-4$ edge', alpha=1, marker='>')
    plt.ylim([0, 1])
    plt.legend(fontsize=15)
    plt.show()

# Fig.~3(a)
# Observation of Recovery Coverage when more participants contribute to the reconstruction of the global network
def RecoveryCoverage():
    P, M = IndividualPerception()
    G = StandardGlobalNetwork()
    X = [i/G.number_of_nodes() for i in range(G.number_of_nodes()+1)]
    Cont1 = Contribution(1)
    Cont2 = Contribution(2)
    Cont3 = Contribution(3)

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[1], x[2]), reverse=True)
    Y = [0]
    edgelist = {edge:0 for edge in G.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values())/G.number_of_edges())
    plt.plot(X, Y, 'red', linestyle='--', label='Count only')

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[2], x[1]), reverse=True)
    Y = [0]
    edgelist = {edge: 0 for edge in G.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / G.number_of_edges())
    plt.plot(X, Y, 'black', linestyle='-.', label='Accuracy only')

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[3],x[1]), reverse=True)
    Y = [0]
    edgelist = {edge: 0 for edge in G.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / G.number_of_edges())
    plt.plot(X, Y, 'steelblue', linestyle='-', label='Our model')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('$f$', fontsize=20)
    plt.ylabel('Recovery Coverage', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    legend = plt.legend(loc='lower right', title='Model', fontsize=15)
    legend.get_title().set_fontsize(fontsize=15)
    plt.show()

# Fig.~3(b)
# Observation of Recovery Accuracy when more participants contribute to the reconstruction of the global network
def RecoveryAccuracy():
    P, M = IndividualPerception()
    G = StandardGlobalNetwork()
    g = RefactoredGlobalNetwork()
    X = [i / G.number_of_nodes() for i in range(G.number_of_nodes() + 1)]
    Cont1 = Contribution(1)
    Cont2 = Contribution(2)
    Cont3 = Contribution(3)

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[1], x[2]), reverse=True)

    Y = [1]
    edgelist = {edge: 0 for edge in G.edges()}
    Edgelist = {edge: 0 for edge in g.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        if M[N] != []:
            for l in M[N]:
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / sum(Edgelist.values()))
    plt.plot(X, Y, 'red', linestyle='--', label='Count only')

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[2], x[1]), reverse=True)
    Y = [1]
    edgelist = {edge: 0 for edge in G.edges()}
    Edgelist = {edge: 0 for edge in g.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        if M[N] != []:
            for l in M[N]:
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / sum(Edgelist.values()))
    plt.plot(X, Y, 'black', linestyle='-.', label='Accuracy only')

    nodelist = []
    for node in G.nodes():
        nodelist.append((node, Cont1[node], Cont2[node], Cont3[node]))
    nodelist = sorted(nodelist, key=lambda x: (x[3],x[1]), reverse=True)
    Y = [1]
    edgelist = {edge: 0 for edge in G.edges()}
    Edgelist = {edge: 0 for edge in g.edges()}
    for node in nodelist:
        N = node[0]
        if P[N] != []:
            for l in P[N]:
                if l in edgelist:
                    if edgelist[l] == 0:
                        edgelist[l] = 1
                elif (l[1], l[0]) in edgelist:
                    if edgelist[(l[1], l[0])] == 0:
                        edgelist[(l[1], l[0])] = 1
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        if M[N] != []:
            for l in M[N]:
                if l in Edgelist:
                    if Edgelist[l] == 0:
                        Edgelist[l] = 1
                elif (l[1], l[0]) in Edgelist:
                    if Edgelist[(l[1], l[0])] == 0:
                        Edgelist[(l[1], l[0])] = 1
        Y.append(sum(edgelist.values()) / sum(Edgelist.values()))

    plt.plot(X, Y, 'steelblue', linestyle='-', label='Our model')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('$f$', fontsize=20)
    plt.ylabel('Recovery Accuracy', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    legend = plt.legend(loc='lower right', title='Model', fontsize=15)
    legend.get_title().set_fontsize(fontsize=15)
    plt.show()

# Fig.~4(a)
# correlation between node degree and individual perceived capability (IPC)
# IPC is calculated by utils.contribution(3)
def Correlation_Degree_IPC():
    X = [47, 35, 49, 46, 61, 38, 39, 29, 30, 33, 41, 32, 30, 41, 30, 38, 21, 28, 32, 52, 11, 22, 41, 32, 43, 49, 39, 33, 58, 37, 26, 25, 33, 28, 32, 24, 35, 40, 5, 31, 50, 30, 38, 11, 19, 33, 29, 38, 43, 34, 31, 58, 35, 28, 44, 30, 29, 35, 46, 32, 29, 35, 26, 34, 19, 21, 32, 35, 29, 28, 33, 6, 23, 25, 44, 9, 23, 6, 35, 15, 23, 32, 30, 19, 17, 31, 21, 49, 39, 60, 47, 36, 23, 26, 28, 15, 13, 16, 17, 21, 14, 17, 8, 14, 18, 20, 11, 18, 13, 23, 18, 3, 26, 22, 20, 21, 8, 16, 20, 33, 14, 29, 38, 14, 9, 11, 6, 7, 9, 7, 12, 14, 6, 18, 15, 22, 9, 5, 6, 12, 12, 8, 33, 13, 5, 4, 5, 8, 3, 3, 2, 2, 5, 5]
    Y = [13.660123847936797, 39.70936196717083, 12.747562160062156, 1.514775855952323, 9.853373873512847, 17.49445346320346, 8.422247787901808, 14.30423234316421, 1.8884383590265958, 24.536124633880025, 16.448231910848005, 8.435824957206533, 0, 24.467198555023298, 5.2062229437229455, 3.9784717905073936, 1.9208530633453225, 0, 19.92377695404012, 55.4426659690164, 0.9800366300366301, 2.032701947175631, 0, 0, 14.223677298793385, 12.626136771725005, 9.07571191558773, 96.52762463350301, 120.37278117613226, 4.244173881673881, 8.149106139818212, 3.19229910096783, 5.441047166627658, 4.620096058911849, 13.155419606193597, 8.04414873911778, 6.3995588009864806, 3.259369212658687, 0, 1.6266926771722166, 20.937712407897646, 4.361024514700982, 7.428255954571739, 8.705377384053852, 1.4363095238095236, 10.629817714759318, 11.724463853140312, 39.89780865699564, 8.100283149083454, 1.1732195550071371, 19.341014650554133, 17.557210892616908, 5.596403888509151, 0.252248117379699, 11.515804250788765, 10.24246097040214, 3.9326330532212883, 12.102988397275983, 26.650175396498906, 107.11323345127525, 4.824453221357246, 0, 10.439600395018346, 136.85356360108253, 126.29671683808719, 3.641944575032811, 6.206058196189773, 7.057683003271238, 1.0913842371079214, 3.500609806859807, 0.03918779234181735, 1.9583333333333333, 4.267974070899262, 0.5261655011655011, 2.5613722663645264, 2.5207481845639736, 1.1858884149905822, 102.845647075233, 6.447500416250415, 1.0, -1.8868686868686873, 7.394205981721115, 2.8865402571284924, 1.599344996249021, 14.440936553126955, 37.70421300405818, 4.907254110698383, 69.21488640423975, 0.19038461538461549, 28.8940365039083, 30.624946224655975, 15.45882895269505, 0, 29.46936381107783, 6.728026037430058, 3.656762681762681, 6.35523088023088, 2.780555555555555, 4.850829725829725, 4.661522687838476, 3.361976911976912, 5.04963924963925, 3.9734693003075354, 2.7326118326118327, 6.327083333333333, 30.795340437426315, 0, 0.40681818181818175, 0.9762426900584795, 98.40113721548677, 18.268199850688227, 0.8794871794871795, 0, 11.544222980465229, 14.813269747967894, 9.68123215292333, 1.9166666666666663, 43.10026863014863, 2.1486312298812296, 41.55952477604953, 0.04969146350725295, 13.448069360550004, 0, 13.246968337390168, 0, -1.303296703296703, 94.54171157360011, 1.9, 1.5, 0, 0, 2.8969298245614032, 0.6113636363636363, 4.102247752247754, 15.010984848484844, 0, 3.022435897435897, 1.2575757575757576, 1.8739209320091672, 6.333333333333333, 12.753021284271284, 0, -2.7544511044511033, 0.07142857142857142, 0.41666666666666663, 0, 6.280769230769231, 0, 0, 0.6666666666666666, 0, 4.8611111111111125, 0, 3.3]
    x1 = []
    y1 = []
    for i in range(len(Y)):
        if Y[i]>0 and X[i]>0:
            x1.append(np.log(X[i]))
            y1.append(np.log(Y[i]))
    p = np.polyfit(x1, y1, 1)
    plt.figure(figsize=(8,6))
    x2 = np.arange(2, 70,1)
    y2 = np.exp(np.polyval(p, np.log(x2)))
    plt.loglog(X, Y, '*', x2, y2, '-')
    plt.xlabel('Degree', fontsize=20)
    plt.ylabel('Individual Perceptual Capability', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~4(b)
# correlation between node betweenness and individual perceived capability (IPC)
# IPC is calculated by utils.contribution(3)
def Correlation_Betweenness_IPC():
    X= [0.04201027398118805, 0.02501960513467327, 0.053285764445723645, 0.06128851871299361, 0.1550873158764445, 0.02594920632138305, 0.04816857371959661, 0.012649624176993595, 0.02054621712290947, 0.0216244297298213, 0.03721788161746836, 0.017185374374039997, 0.017285716223763587, 0.050656104685519926, 0.021505717039170157, 0.030069800200402872, 0.006669731058055069, 0.015553797287562506, 0.01970122713410997, 0.07210038545967404, 0.0005189307432749542, 0.006941537699691502, 0.02980032976042326, 0.024452317443532816, 0.04747575058323725, 0.039551108938089406, 0.021829728399087835, 0.007423510141799285, 0.13002407951737388, 0.05119543680474352, 0.07871181848811137, 0.004214466779279239, 0.03052560490813678, 0.019642243969484423, 0.02038216175259185, 0.007094700415257017, 0.024761051047321056, 0.08011374724844571, 0.00021471653599025185, 0.020050581152814554, 0.13994210421092512, 0.020042144808436222, 0.03187633810190637, 0.005279581228522039, 0.0035447914931637154, 0.009630593132848251, 0.009275969861536117, 0.017058184815871142, 0.07206920512971712, 0.015156503639923114, 0.015410859880098401, 0.11228152908444952, 0.0323851893867255, 0.015392858662285415, 0.051022826766478284, 0.027382120329619813, 0.03155015993869307, 0.017785427852219062, 0.04511734150186132, 0.011557593296915709, 0.03664985248623548, 0.011979590839791582, 0.03131829664717289, 0.032926362528669796, 0.012060976716024421, 0.01983827619462697, 0.04284422090030701, 0.019832313906656886, 0.028151731285835414, 0.016110416276069003, 0.023035877103976928, 0.0037688891670188984, 0.007369480166191107, 0.005833875231331379, 0.09593494198105731, 0.008030341773353579, 0.013950516684107774, 0.000316840268127427, 0.034433255947964043, 0.02901658311781338, 0.024005961624209576, 0.01692584706797146, 0.014219811984813174, 0.007092690238736135, 0.0041850322153169, 0.017090142576864576, 0.0037262322446965654, 0.05083563193134105, 0.11148690823386832, 0.16916238645249027, 0.07548115268389693, 0.027315338891846105, 0.016049731126951263, 0.02974343248994833, 0.009417672203670614, 0.019663154594618787, 0.007864831626813635, 0.013908882069503659, 0.019089305039302822, 0.013151524039304457, 0.014769228996484508, 0.012250079126417535, 0.002582597738170783, 0.008725701879119698, 0.007726313328547165, 0.006785315098395734, 0.005289579124843642, 0.03312970536006236, 0.057177201202525216, 0.019992555422733368, 0.004714900804020052, 0.0, 0.01612214845824335, 0.019551060382460853, 0.011917959652704644, 0.00889751770500099, 0.023900700005971408, 0.005700236283555025, 0.012932725046905166, 0.029249713627841123, 0.002937524863410329, 0.010552945025466687, 0.04935002555433023, 0.005917154616637275, 0.001848528788372358, 0.0703728203595768, 0.00044811417479554154, 0.004340867035957601, 0.007264671257088599, 0.005964391333294113, 0.003764060921550895, 0.01736867934668307, 0.0023796264730212295, 0.019639702062508962, 0.021505780941481298, 0.0682127111499049, 0.014032284235107058, 0.001737569441421709, 0.0002548145972141875, 0.005080249251580816, 0.002266177866260671, 0.018123763024364786, 0.04160909290610092, 0.010831014816387074, 0.004934411631399004, 0.01458803795950051, 0.004031718160779199, 0.010608388466439038, 0.0010869301254620935, 0.007998457671850706, 0.0, 0.0, 0.001684972065474983, 0.003608856991178694]
    Y= [13.660123847936797, 39.70936196717083, 12.747562160062156, 1.514775855952323, 9.853373873512847, 17.49445346320346, 8.422247787901808, 14.30423234316421, 1.8884383590265958, 24.536124633880025, 16.448231910848005, 8.435824957206533, 0, 24.467198555023298, 5.2062229437229455, 3.9784717905073936, 1.9208530633453225, 0, 19.92377695404012, 55.4426659690164, 0.9800366300366301, 2.032701947175631, 0, 0, 14.223677298793385, 12.626136771725005, 9.07571191558773, 96.52762463350301, 120.37278117613226, 4.244173881673881, 8.149106139818212, 3.19229910096783, 5.441047166627658, 4.620096058911849, 13.155419606193597, 8.04414873911778, 6.3995588009864806, 3.259369212658687, 0, 1.6266926771722166, 20.937712407897646, 4.361024514700982, 7.428255954571739, 8.705377384053852, 1.4363095238095236, 10.629817714759318, 11.724463853140312, 39.89780865699564, 8.100283149083454, 1.1732195550071371, 19.341014650554133, 17.557210892616908, 5.596403888509151, 0.252248117379699, 11.515804250788765, 10.24246097040214, 3.9326330532212883, 12.102988397275983, 26.650175396498906, 107.11323345127525, 4.824453221357246, 0, 10.439600395018346, 136.85356360108253, 126.29671683808719, 3.641944575032811, 6.206058196189773, 7.057683003271238, 1.0913842371079214, 3.500609806859807, 0.03918779234181735, 1.9583333333333333, 4.267974070899262, 0.5261655011655011, 2.5613722663645264, 2.5207481845639736, 1.1858884149905822, 102.845647075233, 6.447500416250415, 1.0, -1.8868686868686873, 7.394205981721115, 2.8865402571284924, 1.599344996249021, 14.440936553126955, 37.70421300405818, 4.907254110698383, 69.21488640423975, 0.19038461538461549, 28.8940365039083, 30.624946224655975, 15.45882895269505, 0, 29.46936381107783, 6.728026037430058, 3.656762681762681, 6.35523088023088, 2.780555555555555, 4.850829725829725, 4.661522687838476, 3.361976911976912, 5.04963924963925, 3.9734693003075354, 2.7326118326118327, 6.327083333333333, 30.795340437426315, 0, 0.40681818181818175, 0.9762426900584795, 98.40113721548677, 18.268199850688227, 0.8794871794871795, 0, 11.544222980465229, 14.813269747967894, 9.68123215292333, 1.9166666666666663, 43.10026863014863, 2.1486312298812296, 41.55952477604953, 0.04969146350725295, 13.448069360550004, 0, 13.246968337390168, 0, -1.303296703296703, 94.54171157360011, 1.9, 1.5, 0, 0, 2.8969298245614032, 0.6113636363636363, 4.102247752247754, 15.010984848484844, 0, 3.022435897435897, 1.2575757575757576, 1.8739209320091672, 6.333333333333333, 12.753021284271284, 0, -2.7544511044511033, 0.07142857142857142, 0.41666666666666663, 0, 6.280769230769231, 0, 0, 0.6666666666666666, 0, 4.8611111111111125, 0, 3.3]
    x1 = []
    y1 = []
    for i in range(len(Y)):
        if Y[i]>0 and X[i]>0:
            x1.append(np.log(X[i]))
            y1.append(np.log(Y[i]))
    p = np.polyfit(x1, y1, 1)
    x2 = np.arange(0.0001, 0.2, 0.001)
    y2 = np.exp(np.polyval(p, np.log(x2)))
    plt.figure(figsize=(8,6))
    plt.loglog(X, Y, '*', x2, y2, '-')
    plt.xlabel('Betweenness', fontsize=20)
    plt.ylabel('Individual Perceptual Capability', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~S4(a)
# distribution of knowable degree (perceived ties and misperceived ties)
def KDdis():
    KD = KnowableDegree()
    P = []
    M = []
    for tie in KD:
        if KD[tie] < 0:
            M.append(KD[tie])
        else:
            P.append(KD[tie])
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    for i in range(27):
        X1.append(i)
        Y1.append(P.count(i))

    for i in range(11):
        X2.append(-i-1)
        Y2.append(M.count(-i-1))
    plt.xlabel('Tie Transparency', fontsize=20)
    plt.ylabel('Count',fontsize=20)
    plt.bar(X1, Y1, color='#2980b9', label='Perceived Ties')
    plt.bar(X2, Y2, color='#bdc3c7', label='Misperceived Ties')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    legend = plt.legend(loc='upper right', title='Label', fontsize=15)
    legend.get_title().set_fontsize(fontsize=15)
    plt.show()

# Fig.~S4(b)
# distribution of individuals who have several (x-axis) covert ties
def CTdis():
    H = CovertTies()
    G = StandardGlobalNetwork()
    CD = {node:0 for node in G.nodes()}
    for h in H:
        CD[h[0]]+= 1
        CD[h[1]] += 1
    L = list(CD.values())
    X = []
    Y = []
    for i in range(max(L)+1):
        X.append(i)
        Y.append(L.count(i))
    plt.bar(X, Y, color='grey')
    plt.xlabel('Covert Ties', fontsize=20)
    plt.ylabel('$N(p)$', fontsize=20)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Fig.~S5(a-c)
# Distributions of edge embeddedness and edge betweenness of covert ties, transparent ties, and the ratio of covert ties in all ties
def TvsC(flag):

    G = StandardGlobalNetwork()
    KD = KnowableDegree()
    Data = []

    Embeddedness = {}
    for edge in G.edges():
        a = G.degree(edge[0])
        b = G.degree(edge[1])
        c = len(list(nx.common_neighbors(G, edge[0], edge[1])))
        Embeddedness[edge] = c / (a + b - c)
    e = max(Embeddedness.values())
    edge_betweeness = nx.edge_betweenness_centrality(G)
    b = max(edge_betweeness.values())
    H1 = []
    H2 = []
    for edge in G.edges():
        if KD[edge] == 0:
            H1.append([Embeddedness[edge]/e, edge_betweeness[edge]/b]) #covert
        else:
            H2.append([Embeddedness[edge]/e, edge_betweeness[edge]/b]) #transparent
    Hi1 = [[0 for i in range(20)] for j in range(20)]
    Hi2 = [[0 for i in range(20)] for j in range(20)]
    for h in H1:
        if h[0] == 1:
            if h[1] == 1:
                Hi1[19][19] += 1
            else:
                Hi1[19][int(h[1]*20)] += 1
        else:
            if h[1] == 1:
                Hi1[int(h[0]*20)][19] += 1
            else:
                Hi1[int(h[0]*20)][int(h[1]*20)] += 1
    for h in H2:
        if h[0] == 1:
            if h[1] == 1:
                Hi2[19][19] += 1
            else:
                Hi2[19][int(h[1]*20)] += 1
        else:
            if h[1] == 1:
                Hi2[int(h[0]*20)][19] += 1
            else:
                Hi2[int(h[0]*20)][int(h[1]*20)] += 1
    for i in range(20):
        for j in range(20):
            if flag == 1:
                Data.append([i,j,Hi1[i][j]])
            if flag == 2:
                Data.append([i,j,Hi2[i][j]])
            if flag == 3:
                if Hi1[i][j] + Hi2[i][j] == 0:
                    Data.append([i,j,0])
                else:
                    Data.append([i,j, Hi1[i][j]/(Hi1[i][j] + Hi2[i][j])])
    df = pd.DataFrame(Data, columns=['A', 'B', 'num'])
    flights = df.pivot("B", "A", "num")
    plt.figure(figsize=(6,5))
    g = sns.heatmap(flights,linewidths=.1)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
               ['0', ' ', '0.1', ' ', '0.2', ' ', '0.3', ' ', '0.4', ' ', '0.5', ' ', '0.6', ' ', '0.7', ' ', '0.8', ' ', '0.9', ' ', '1'])
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
               ['0', ' ', '0.1', ' ', '0.2', ' ', '0.3', ' ', '0.4', ' ', '0.5', ' ', '0.6', ' ', '0.7', ' ', '0.8', ' ', '0.9', ' ', '1'])
    g.set_xlabel('Edge Embeddedness', fontsize=20)
    g.set_ylabel('Edge Betweenness', fontsize=20)
    plt.show()


# Fig.~S6(a)
# distribution of covert ties with each same/different attribute
def CovertTies_Attributes():
    from data import attribute as A
    KD = KnowableDegree()
    G = StandardGlobalNetwork()
    DG = DirectedGlobalNetwork()
    Data = []
    plt.figure(figsize=(6,5))
    Total = {'Degree':0, 'Grade':0, 'Department':0, 'Class':0, 'Gender':0}
    Covert = {'Degree':0, 'Grade':0, 'Department':0, 'Class':0, 'Gender':0}
    for edge in G.edges():
        if A.A[edge[0]][:3] != A.A[edge[1]][:3]:  # degree
            Total['Degree'] += 1
            if KD[edge] == 0:
                Covert['Degree'] += 1
        if A.A[edge[0]][3:7] != A.A[edge[1]][3:7]:  # grade
            Total['Grade'] += 1
            if KD[edge] == 0:
                Covert['Grade'] += 1
        if A.A[edge[0]][7:13] != A.A[edge[1]][7:13]:  # department
            Total['Department'] += 1
            if KD[edge] == 0:
                Covert['Department'] += 1
        if A.A[edge[0]][13:35] != A.A[edge[1]][13:35]:  # class
            Total['Class'] += 1
            if KD[edge] == 0:
                Covert['Class'] += 1
        if A.A[edge[0]][35:] != A.A[edge[1]][35:]:  # gender
            Total['Gender'] += 1
            if KD[edge] == 0:
                Covert['Gender'] += 1
    for key in Total:
        Data.append([key, Total[key], Covert[key]])
    df = pd.DataFrame(Data, columns=['Attributes', 'Different', 'Covert'])
    df = df.sort_values("Different", ascending=False)
    plt.subplot(211)
    sns.barplot(x="Different", y="Attributes", data=df,
                label="Different", color="#ecf0f1")
    sns.barplot(x="Covert", y="Attributes", data=df,
                label="Covert", color='#bdc3c7')
    plt.legend(fontsize=10)
    plt.xlim([0,2000])
    plt.xlabel(' ')
    plt.tick_params(labelsize=10)
    plt.ylabel(' ')
    Data = []
    Total = {'Degree':0, 'Grade':0, 'Department':0, 'Class':0, 'Gender':0}
    Covert = {'Degree':0, 'Grade':0, 'Department':0, 'Class':0, 'Gender':0}
    for edge in G.edges():
        if A.A[edge[0]][:3] == A.A[edge[1]][:3]:  # degree
            Total['Degree'] += 1
            if KD[edge] == 0:
                Covert['Degree'] += 1
        if A.A[edge[0]][3:7] == A.A[edge[1]][3:7]:  # grade
            Total['Grade'] += 1
            if KD[edge] == 0:
                Covert['Grade'] += 1
        if A.A[edge[0]][7:13] == A.A[edge[1]][7:13]:  # department
            Total['Department'] += 1
            if KD[edge] == 0:
                Covert['Department'] += 1
        if A.A[edge[0]][13:35] == A.A[edge[1]][13:35]:  # class
            Total['Class'] += 1
            if KD[edge] == 0:
                Covert['Class'] += 1
        if A.A[edge[0]][35:] == A.A[edge[1]][35:]:  # gender
            Total['Gender'] += 1
            if KD[edge] == 0:
                Covert['Gender'] += 1
    for key in Total:
        Data.append([key, Total[key], Covert[key]])
    df = pd.DataFrame(Data, columns=['Attributes', 'Same', 'Covert'])
    df = df.sort_values("Same", ascending=False)
    plt.subplot(212)
    sns.barplot(x="Same", y="Attributes", data=df,
                label="Same", color='#3498db')
    sns.barplot(x="Covert", y="Attributes", data=df,
                label="Covert", color='#2980b9')
    plt.legend(fontsize=10)
    plt.xlim([0,2000])
    plt.ylabel(' ')
    plt.xlabel('Count',fontsize=15)
    plt.tick_params(labelsize=10)
    plt.show()

