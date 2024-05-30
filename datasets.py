import sys
from operator import itemgetter
import networkx as nx
import numpy as np
from lxml import etree
import tarfile
import glob
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def load_datasetGEANT():
    current_dir = Path(os.getcwd())
    data_dir = os.path.join(current_dir, 'data', 'GEANT')  # current_dir.parent

    # make data directory
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # download dataset
    print("Downloading dataset...")
    path_to_downloaded_file = tf.keras.utils.get_file(
        os.path.join(data_dir, 'traffic-matrices-anonymized-v2.tar.bz2'),
        'http://totem.run.montefiore.ulg.ac.be/files/data/traffic-matrices-anonymized-v2.tar.bz2')

    # extract tar.bz2 file
    print('Extracting dataset...')
    tar = tarfile.open(path_to_downloaded_file)
    tar.extractall(data_dir)
    tar.close()

    # routing matrix R
    R = np.zeros((74, 23*23))  # 74 links and 529 (=23*23) OD pairs

    # parse xml of topology
    print("Parsing topology XML file...")
    topo = etree.parse(os.path.join(data_dir, 'topology-anonymised.xml'))
    edges = [tuple(map(lambda x: int(x)-1, link.get('id').split('_')))
             for link in topo.findall('.//topology//links//link')]
    # create directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    # get shortest path for all pairs
    path = dict(nx.all_pairs_shortest_path(G))
    # ordered list of edges/links
    sorted_edges = sorted(edges, key=itemgetter(0, 1))
    # mapping IDs to edges
    mapped = dict((e, i) for i, e in enumerate(sorted_edges))
    # Origin Destination pair ID (ordering based on ascending order of nodes)
    od = 0
    for origin in range(23):
        for destination in range(23):
            p = path[origin][destination]
            for e in zip(p, p[1:]):  # for every edge/link in the path
                eid = mapped[e]  # edge/link ID
                R[eid, od] = 1  # mark with 1 the link participating in the OD path
            od += 1

    # save array R for future use
    with open(os.path.join(data_dir, 'R.npy'), 'wb') as f:
        np.save(f, R)

    # there are 10772 files, each file has a TM with 23*23 OD pairs
    X = np.empty((10772, 23, 23))

    print("Parsing TM XML files...")
    index = 0
    # for every TM xml file
    for f in sorted(os.listdir(os.path.join(data_dir, 'traffic-matrices'))):
        # parse TM xml
        xmlTM = etree.parse(os.path.join(data_dir, 'traffic-matrices', f))
        for src in xmlTM.iter("src"):  # for every source
            srcID = int(src.get("id"))
            for dst in src.iter("dst"):  # for every destination
                dstID = int(dst.get("id"))
                val = float(dst.text)  # traffic intensity in kbps
                X[index, srcID-1, dstID-1] = val
        index += 1

    # save array X for future use
    with open(os.path.join(data_dir, 'X.npy'), 'wb') as f:
        np.save(f, X)

    return R, X


def load_datasetABILENE():
    current_dir = Path(os.getcwd())
    data_dir = os.path.join(current_dir, 'data',
                            'ABILENE')  # current_dir.parent

    # make data directory
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # download dataset
    print("Downloading dataset...")
    path_to_downloaded_file = tf.keras.utils.get_file(
        os.path.join(data_dir, 'abilene-TM.tar.gz'),
        'http://totem.run.montefiore.ulg.ac.be/files/data/abilene-TM.tar.gz')

    # extract tar.gz file
    print('Extracting dataset...')
    tar = tarfile.open(path_to_downloaded_file)
    tar.extractall(data_dir)
    tar.close()

    # routing matrix R
    R = np.zeros((30, 12*12))  # 30 links and 144 (=12*12) OD pairs

    # parse xml of topology
    print("Parsing topology XML file...")
    topo = etree.parse(os.path.join(data_dir, 'abilene-TM',
                       'topo', 'Abilene-Topo-10-04-2004.xml'))
    n_map = dict()  # mapping of node names to IDs
    counter = 0
    for node in topo.iter("node"):  # for every source
        n_map[node.get("id")] = counter
        counter += 1
    edges = list()
    for link in topo.findall('.//igp//links/link'):
        # get the two endpoints of the link
        s, d = map(lambda n: n_map[n], link.get('id').split(','))
        for static in link:
            for metric in static:
                # get the weight of the link
                w = int(metric.text)
        edges.append((s, d, {"weight": w}))
    # create directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    # get shortest path for all pairs
    path = dict(nx.all_pairs_dijkstra_path(G))
    # ordered list of edges/links
    sorted_edges = sorted(edges, key=itemgetter(0, 1))
    # mapping IDs to edges
    mapped = dict((e, i) for i, e in enumerate(
        [(s, d) for s, d, _ in sorted_edges]))
    # Origin Destination pair ID (ordering based on ascending order of nodes)
    od = 0
    for origin in range(12):
        for destination in range(12):
            p = path[origin][destination]
            for e in zip(p, p[1:]):  # for every edge/link in the path
                eid = mapped[e]  # edge/link ID
                R[eid, od] = 1  # mark with 1 the link participating in the OD path
            od += 1

    # save array R for future use
    with open(os.path.join(data_dir, 'R.npy'), 'wb') as f:
        np.save(f, R)

    # there are 48096 xml files, each file has a TM with 12*12 OD pairs
    X = np.empty((48096, 12, 12))

    print("Parsing TM XML files...")
    index = 0
    # for every TM xml file
    for f in sorted(glob.glob(os.path.join(data_dir,'abilene-TM','TM','2004','**','*.xml'),
                              recursive = True)):
        # parse TM xml
        xmlTM = etree.parse(f)
        for src in xmlTM.iter("src"):  # for every source
            srcID = n_map[src.get("id")]
            for dst in src.iter("dst"):  # for every destination
                dstID = n_map[dst.get("id")]
                val = float(dst.text)  # traffic intensity in kbps
                X[index, srcID, dstID] = val
        index += 1

    # save array X for future use
    with open(os.path.join(data_dir, 'X.npy'), 'wb') as f:
        np.save(f, X)

    return R, X


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python {} [geant|abilene]'.format(sys.argv[0]))
    elif sys.argv[1] == 'geant':
        load_datasetGEANT()
    elif sys.argv[1] == 'abilene':
        load_datasetABILENE()
    else:
        print("Invalid argument. Must be 'geant' or 'abilene'")
