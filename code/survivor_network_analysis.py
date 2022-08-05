import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from collections import Counter

def returnImmunityTribeResult(res, df):
    '''
    PARAMETERS
    res: str
        res is the result of that tribes immunity challenge. It is either win or lose.
    df: DataFrame
        df is subset of the season statistics filtered by tribe and episode.
        
    RETURNS
        either a undirected fully connected graph if the tribe won immunity challenge 
        or a directed graph of the losing tribe's voting ceremony
    '''
    if res == 'win':
        e = []
        immune = df["Contestants"].unique()
        for i in range(len(immune)):
            for j in range(len(immune)):
                if i != j:
                    e.append((immune[i],immune[j]))
        G = nx.Graph()
        G.add_edges_from(e)
    elif res == 'lose':
        tribal = df[['Contestants','TCV']]
        tr = list(tribal.itertuples(index=False, name=None))
        G = nx.DiGraph()
        G.add_edges_from(tr)
        
    return G

def getNodeColor(G,nodecolor):
    groups = set(nx.get_node_attributes(G,nodecolor).values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = G.nodes()
    colors = [mapping[G.nodes[n][nodecolor]] for n in nodes]
    return colors


def getNodeSize(G,nodesize):
    groups = set(nx.get_node_attributes(G,nodesize).values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = G.nodes()
    size = [mapping[G.nodes[n][nodesize]] for n in nodes]
    return size

def plotPreMerged(seasonstats, nodecolor=None, nodesize=None):
    '''
    PARAMETERS
        seasonstats: DataFrame
            seasonstats is the raw data input read in from the Survivor season statistics
        nodecolor: str
            a column name to color the nodes of the network
        
    RETURNS
        plots each tribe's network for each episode, two networks per figure: 
        a directional graph if a tribe had to go to tribal council
        an undirected fully connected graph if a tribe won immunity challenge
    '''
    season = str(seasonstats['Season'].unique()[0])
    immunityresults = pd.pivot_table(seasonstats[seasonstats['Merged'] == False], values='TICW', index=['Episode'],columns=['Original Tribe'], aggfunc=np.sum)
    tribeList = immunityresults.reset_index(drop=True).columns.to_list()
    
    for i in tribeList:
        immunityresults[i] = immunityresults[i].apply(lambda x: 'lose' if x == 0 else 'win')
        
    tribeCh = list(immunityresults.itertuples(index=True, name=None))
    for i in range(len(tribeCh)):
        # for each episode, get the two tribes subset of statistics
        ep_tribe1 = seasonstats[(seasonstats['Tribe'] == tribeList[0]) & (seasonstats['Episode'] == i+1)]
        ep_tribe2 = seasonstats[(seasonstats['Tribe'] == tribeList[1]) & (seasonstats['Episode'] == i+1)]

        # use returnImmunityTribeResult to return a winning/losing network
        tribe1 = returnImmunityTribeResult(tribeCh[i][1], ep_tribe1)
        tribe2 = returnImmunityTribeResult(tribeCh[i][2], ep_tribe2)

        fig = plt.figure(figsize=(15, 6))
        fig.suptitle('Survivor Season ' + season + ' Episode ' + str(i+1))
        
        # set left side of the two tribes
        
        nx.set_node_attributes(tribe1, getNodeAttributes(seasonstats))
        
        if nodecolor is not None:
            tribe1nodecolor = getNodeColor(tribe1,nodecolor)
            cmap=plt.cm.get_cmap('Blues')
        else:
            tribe1nodecolor = None
            cmap=None
            
        if nodesize is not None:
            tribe1nodesize = getNodeSize(tribe1,nodesize)
        else:
            tribe1nodesize = None
            
        pos=nx.circular_layout(tribe1)
        
        plt.subplot(1, 2, 1)
        nx.draw(tribe1,
                pos=pos,
                with_labels=True,
                # node_size=tribe1nodesize,
                node_color=tribe1nodecolor,
                cmap=cmap)

        plt.title(tribeList[0]+' Tribe')
        
        # set right side of the two tribes
        
        nx.set_node_attributes(tribe2, getNodeAttributes(seasonstats))
        
        if nodecolor is not None:
            tribe2nodecolor = getNodeColor(tribe2,nodecolor)
            cmap=plt.cm.get_cmap('Blues')
        else:
            tribe2nodecolor = None
            cmap=None
        
        if nodesize is not None:
            tribe2nodesize = getNodeSize(tribe2,nodesize)
        else:
            tribe2nodesize = None

        pos=nx.circular_layout(tribe2)

        plt.subplot(1, 2, 2)
        nx.draw(tribe2,
                pos=pos,
                with_labels=True,
                # node_size=tribe2nodesize,
                node_color=tribe2nodecolor,
                cmap=cmap)
        plt.title(tribeList[1] + ' Tribe')

        plt.show()
        picname = "S"+ season + "-Tribe Network-" +"EP" + "{:02d}".format(i+1) + ".png"
        fig.savefig("../images/Season" + season + "/" + picname)
        
        
def plotMerged(seasonstats,nodecolor):
    '''
    PARAMETERS
        seasonstats: DataFrame
            seasonstats is the full imported dataframe of the season statistics
        nodecolor: str
            a column name to color the nodes of the network
        
    RETURNS
        plots a directional graph of each tribal council for each episode during the merged tribal period
    '''
    epList = seasonstats[(seasonstats['Merged'] == True)]['Episode'].unique()
    season = str(seasonstats['Season'].unique()[0])
    tribeName = seasonstats[(seasonstats['Merged'] == True)]['Tribe'].unique()[0]
    lastEp = max(epList)
    
    for i in range(len(epList)):
        if epList[i] != lastEp:
            ep = epList[i]
            tribalVotes = seasonstats[(seasonstats['Merged'] == True) & (seasonstats['Episode'] == ep)][['Contestants','TCV']]
            indvVotes = list(tribalVotes.itertuples(index=False, name=None))

            fig = plt.figure(figsize=(15, 6))

            G = nx.DiGraph()
            G.add_edges_from(indvVotes)
            
            nx.set_node_attributes(G, getNodeAttributes(seasonstats))

            if nodecolor is not None:
                colors = getNodeColor(G,nodecolor)
                cmap=plt.cm.get_cmap('Blues')
            else:
                colors = None
                cmap=None
            
            pos=nx.circular_layout(G)
            
            plt.subplot(1, 1, 1)
            nx.draw(G,
                    pos=pos,
                    with_labels=True,
                    node_color=colors,
                    cmap=cmap)
            
            fig.suptitle(tribeName + " Tribe")
            plt.title('Survivor Season ' + season + ' Episode ' + str(ep))
            picname = "S"+ season + "-Tribe Network-" +"EP" + "{:02d}".format(ep) + ".png"
            fig.savefig("../images/Season" + season + "/" + picname)
            plt.show()
        else:
            ep = epList[i]
            tribalVotes = seasonstats[(seasonstats['Merged'] == True) & (seasonstats['Episode'] == ep)][['Contestants','TCV']]
            indvVotes = list(tribalVotes.itertuples(index=False, name=None))
            e = []
            
            for i in range(len(indvVotes)):
                if str(indvVotes[i][1]) != 'nan':
                    e.append(indvVotes[i])

            n = seasonstats[(seasonstats['Merged'] == True) & (seasonstats['Episode'] == ep)]['Contestants'].unique()
            
            G = nx.DiGraph()
            G.add_nodes_from(n)
            G.add_edges_from(e)
            nx.set_node_attributes(G, getNodeAttributes(seasonstats))
            
            if nodecolor is not None:
                colors = getNodeColor(G,nodecolor)
                cmap=plt.cm.get_cmap('Blues')
            else:
                colors = None
                cmap=None
            
            pos=nx.circular_layout(G)
            
            fig = plt.figure(figsize=(15, 6))
            fig.suptitle(tribeName + " Tribe")
            plt.subplot(1, 1, 1)
            nx.draw(G,
                    pos=pos,
                    with_labels=True,
                    node_color=colors,
                    cmap=cmap)
            
            plt.title('Survivor Season ' + season + ' Episode ' + str(ep))
            plt.show()
            picname = "S"+ season + "-Tribe Network-" +"EP" + "{:02d}".format(ep) + ".png"
            fig.savefig("../images/Season" + season + "/" + picname)

def plotSeasonEpisodes(seasonstats,nodecolor):
    '''
    PARAMETERS
        seasonstats: DataFrame
                seasonstats is the raw data input read in from the Survivor season statistics
        nodecolor: str
            a column name to color the nodes of the network
                
    RETURNS
        A chronological progression of contestants leaving the show
    '''
    plotPreMerged(seasonstats,nodecolor)
    plotMerged(seasonstats,nodecolor)
    
    
def network_summary(G):

    def centrality_stats(x):
        x1=dict(x)
        x2=np.array(list(x1.values())); #print(x2)
        print("	min:" ,min(x2))
        print("	mean:" ,np.mean(x2))
        print("	median:" ,np.median(x2))
        # print("	mode:" ,stats.mode(x2)[0][0])
        print("	max:" ,max(x2))
        x=dict(x)
        sort_dict=dict(sorted(x1.items(), key=lambda item: item[1],reverse=True))
        print("	top nodes:",list(sort_dict)[0:6])
        print("	          ",list(sort_dict.values())[0:6])

    try: 
        print("GENERAL")
        print("	number of nodes:",len(list(G.nodes)))
        print("	number of edges:",len(list(G.edges)))

        print("	is_directed:", nx.is_directed(G))
        print("	is_weighted:" ,nx.is_weighted(G))
        # print("	is_connected:" ,nx.is_connected(G))
        print("	is_tree:" ,nx.is_tree(G))
        # print("	number_connected_components", nx.number_connected_components(G))
        # print("	number of triangle: ",len(nx.triangles(G).keys()))
        print("	density:" ,nx.density(G))
        print("	average_clustering coefficient: ", nx.average_clustering(G))
        print("	degree_assortativity_coefficient: ", nx.degree_assortativity_coefficient(G))

        # if(nx.is_connected(G)):
        #     print("	diameter:" ,nx.diameter(G))
        #     print("	radius:" ,nx.radius(G))
        #     print("	average_shortest_path_length: ", nx.average_shortest_path_length(G))

        #CENTRALITY 
        print("DEGREE")
        centrality_stats(nx.degree(G))

        print("CLOSENESS CENTRALITY")
        centrality_stats(nx.closeness_centrality(G))

        print("BETWEEN CENTRALITY")
        centrality_stats(nx.betweenness_centrality(G))
    except:
        print("unable to run")
        
        
def getNodeAttributes(seasonstats):
    f = seasonstats[seasonstats['Episode'] == 1][["Contestants","Sex","Age","Occupation","Home State", "Original Tribe"]]
    c = list(f.columns)[1:]
    
    attr = {}

    for i in range(len(f)):
        tmp = {}
        attr[f.iloc[i][0]] = tmp

        for j in range(len(c)):
            tmp[c[j]] = f.iloc[i][j+1] 
            
    return attr

def allVotes(seasonstats):
    d = seasonstats[['Contestants','TCV']].dropna()
    y = list(d.itertuples(index=False, name=None))
    p = list(Counter(y).items())

    allVotesAttr = {}

    for i in range(len(p)):
        tmp = {}
        allVotesAttr[p[i][0]] = tmp

        tmp["NumVotes"] = p[i][1]

    G = nx.DiGraph()
    fig = plt.figure(figsize=(15, 10))

    G.add_nodes_from(d["Contestants"].unique())
    G.add_edges_from(y)

    widths = list(nx.get_edge_attributes(G,'NumVotes').values())
    n_color = list(nx.get_node_attributes(G,'Sex').values())

    nx.set_edge_attributes(G, allVotesAttr)
    nx.set_node_attributes(G, getNodeAttributes(seasonstats))


    plt.title('End of Season 1 Aggregated Voting History')
    pos=nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, 
            # width=widths
           )

    network_summary(G)
    

def getMeasures(G):
    epMeasures = []
    epMeasures.append(len(list(G.nodes)))
    epMeasures.append(len(list(G.edges)))

    epMeasures.append(nx.density(G))
    epMeasures.append(nx.average_clustering(G))
    # epMeasures.append(nx.degree_assortativity_coefficient(G))


    nodemeasures=np.array(list(dict(nx.degree(G)).values())); #print(x2)
    epMeasures.append(min(nodemeasures))
    epMeasures.append(np.mean(nodemeasures))
    epMeasures.append(np.median(nodemeasures))
    epMeasures.append(max(nodemeasures))

    closemeasures=np.array(list(dict(nx.closeness_centrality(G)).values())); #print(x2)
    epMeasures.append(min(closemeasures))
    epMeasures.append(np.mean(closemeasures))
    epMeasures.append(np.median(closemeasures))
    epMeasures.append(max(closemeasures))

    betweenmeasures=np.array(list(dict(nx.betweenness_centrality(G)).values())); #print(x2)
    epMeasures.append(min(betweenmeasures))
    epMeasures.append(np.mean(betweenmeasures))
    epMeasures.append(np.median(betweenmeasures))
    epMeasures.append(max(betweenmeasures))
    return epMeasures


def measuresMerged(seasonstats):
    '''
    PARAMETERS
        seasonstats: DataFrame
            seasonstats is the full imported dataframe of the season statistics
        nodecolor: str
            a column name to color the nodes of the network
        
    RETURNS
        plots a directional graph of each tribal council for each episode during the merged tribal period
    '''
    
    measures = [
        "NumberOfNodes",
        "NumberOfEdges",
        "Density",
        "AverageClusteringCoefficient",
        # "DegreeAssortativityCoefficient",
        "DegreeMin",
        "DegreeMean",
        "DegreeMedian",
        "DegreeMax",
        "ClosenessMin",
        "ClosenessMean",
        "ClosenessMedian",
        "ClosenessMax",
        "BetweenMin",
        "BetweenMean",
        "BetweenMedian",
        "BetweenMax"
    ]
    df = pd.DataFrame(index=pd.Index(measures))
    
    epList = seasonstats[(seasonstats['Merged'] == True)]['Episode'].unique()
    season = str(seasonstats['Season'].unique()[0])
    # tribeName = seasonstats[(seasonstats['Merged'] == True)]['Tribe'].unique()[0]
    lastEp = max(epList)
    
    for i in range(len(epList)):
        if epList[i] != lastEp:
            ep = epList[i]
            tribalVotes = seasonstats[(seasonstats['Merged'] == True) & (seasonstats['Episode'] == ep)][['Contestants','TCV']]
            indvVotes = list(tribalVotes.itertuples(index=False, name=None))

            G = nx.DiGraph()
            G.add_edges_from(indvVotes)
            
            df[ep]=getMeasures(G)
        else:
            ep = epList[i]
            tribalVotes = seasonstats[(seasonstats['Merged'] == True) & (seasonstats['Episode'] == ep)][['Contestants','TCV']]
            indvVotes = list(tribalVotes.itertuples(index=False, name=None))
            e = []
            
            for i in range(len(indvVotes)):
                if str(indvVotes[i][1]) != 'nan':
                    e.append(indvVotes[i])

            n = seasonstats[(seasonstats['Merged'] == True) & (seasonstats['Episode'] == ep)]['Contestants'].unique()
            
            G = nx.DiGraph()
            G.add_nodes_from(n)
            G.add_edges_from(e)
            df[ep]=getMeasures(G)
    
    return df.T

def plotCentrality(seasonstats):
    measures = measuresMerged(seasonstats)
    measuresNames = list(measures.columns)
    season = str(seasonstats['Season'].unique()[0])
    for i in range(len(list(measures.columns))):
        plt.figure(figsize=(10,6))
        plt.plot(measures[measuresNames[i]])
        plt.title(measuresNames[i])
        plt.xlabel("Episode")
        picname = "S" + season + "-" +  measuresNames[i] + "-Plot.png"
        plt.savefig("../images/Season" + season + "/" + picname)
        plt.show()
        

def plotAllCentrality(seasonstats):
    measures = measuresMerged(seasonstats)
    measuresNames = list(measures.columns)
    season = str(seasonstats['Season'].unique()[0])

    fig, axs = plt.subplots(4,4, sharex=True)
    fig.set_size_inches(25, 20)
    fig.suptitle('Season 1 Measures of Centrality', fontsize=25)

    t = -1
    for i in range(4):
        for j in range(4):
            t+=1
            axs[i, j].plot(measures[measuresNames[t]])
            axs[i, j].set_ylabel(measuresNames[t])
    
    picname = "S" + season + "-AllCentralities-Plot.png"
    fig.savefig("../images/Season" + season + "/" + picname)
    

def degree_histogram_plot(G,ep,season):
    #COMPUTE DEGREE: --> "LIST" WITH NODE DEGREES
    G_DEGREE=G.degree(); #print(G_DEGREE,G_DEGREE[5],type(G_DEGREE)) 

    #LABELS (DICT)
    labels={}
    for n,d in G_DEGREE: labels[n]=d #str(n)+"-"+str(d) 


    #SORT DEGREE AND STORE IN LIST 
    degree_sequence = sorted((d for n, d in G_DEGREE), reverse=True)
    dmax = max(degree_sequence)   #MAX DEGREE

    #INITIALIZE MPL FIGURE+AX
    fig = plt.figure("Degree of a random graph", figsize=(10, 10))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    #PLOT NETWORK IN UPPER GRID SPACES 
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G)
    nx.draw(G,
        with_labels=True,
        labels=labels,
        node_color='blue',
        node_size=500,
        font_color='white',
        font_size=16,
        pos=pos,ax=ax0
        )
    ax0.set_title("Episode " + str(ep) + " Degree Plots")


    #PLOT RANK (IMPORTANCE BASED ON DEGREE) VS DEGREE)
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    #PLOT RANK HISTOGRAM
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")
    ax0.set_aspect('equal', 'box')
    fig.tight_layout()
    picname = "S"+ season + "-Degree Plots-" + "EP" + "{:02d}".format(ep) + ".png"
    fig.savefig("../images/Season" + season + "/" + picname)
    plt.show()
    
def getMergedHistograms(seasonstats):
    '''
    PARAMETERS
        seasonstats: DataFrame
            seasonstats is the full imported dataframe of the season statistics
        
    RETURNS
        plots histogram statistics of each merged tribal council
    '''
    epList = seasonstats[(seasonstats['Merged'] == True)]['Episode'].unique()
    season = str(seasonstats['Season'].unique()[0])
    tribeName = seasonstats[(seasonstats['Merged'] == True)]['Tribe'].unique()[0]
    lastEp = max(epList)
    
    for i in range(len(epList)):
        if epList[i] != lastEp:
            ep = epList[i]
            tribalVotes = seasonstats[(seasonstats['Merged'] == True) & (seasonstats['Episode'] == ep)][['Contestants','TCV']]
            indvVotes = list(tribalVotes.itertuples(index=False, name=None))

            G = nx.DiGraph()
            G.add_edges_from(indvVotes)

            degree_histogram_plot(G,ep,season)