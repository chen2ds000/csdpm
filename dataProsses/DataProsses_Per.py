import ast
import os
import time
from random import sample

import numpy as np
import csv

import pandas as pd
import torch.nn.functional as F
import torch


def softmax(logits):
    exp_scores = np.exp(logits - np.max(logits))  # 防止溢出
    return exp_scores / np.sum(exp_scores, axis=0)


def DataProsses(uoc,edges):
    x = np.array(list(uoc.values()))
    X=[]
    for i in x:
        if i >=0.5:
            data = [0,1]
        else:
            data = [1,0]
        X.append(data)
    X = np.array(X)


    nodes = sorted(set([node for edge in edges for node in edge]))


    node_indices = {node: idx for idx, node in enumerate(nodes)}


    n = len(nodes)
    adj_matrix = np.zeros((n, n))


    for (node1, node2), weight in edges.items():
        i, j = node_indices[node1], node_indices[node2]
        if weight >=0.5:
            weight=1
        else:
            weight=0
        adj_matrix[i, j] = weight
        adj_matrix[j, i] = weight  # 无向图对称
    return X,adj_matrix

def load_q_matrix(q_matrix_path):

    q_matrix = np.loadtxt(q_matrix_path, dtype=float)
    for i in range(q_matrix.shape[0]):
        temp = q_matrix[i]
        temp = torch.from_numpy(temp)
        temp = F.softmax(temp,dim=0)
        q_matrix[i] = temp.numpy()
    #   q_matrix[i] = softmax(q_matrix[i]) # 将 1 替换为均值小数
    return q_matrix



def load_user_data(csv_file_path):

    user_data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过表头
        for row in csv_reader:
            user_id, exer_id, score = row
            if math1:
                user_id = (int)(user_id) - 1
                exer_id = (int)(exer_id) - 1
            score = float(score)
            if score>=0.5:
                score = 1.0
            else:
                score = 0.0

            user_data.append({"user_id": user_id, "exer_id": exer_id, "score": score})
    return user_data



def calculate_uoc(user_data, q_matrix):

    knowledge_count = q_matrix.shape[1]
    q_count = q_matrix[0]
    knowledge_uoc = np.zeros(knowledge_count)

    for k in range(knowledge_count):
        Y_sum = 0
        X_sum = 0
        for record in user_data:
            exer_id = record["exer_id"]
            score = record["score"]
            weights = q_matrix[int(exer_id)]
            Y_sum += (score / knowledge_count) * weights[k]
            X_sum += (1 / knowledge_count) * weights[k]
        knowledge_uoc[k] = Y_sum / X_sum

    return {f"C{i + 1}": uoc for i, uoc in enumerate(knowledge_uoc)}



def calculate_rd(q_matrix):

    knowledge_count = q_matrix.shape[1]
    q_count = q_matrix.shape[0]
    rd_results = {}

    for i in range(knowledge_count):
        for j in range(i + 1, knowledge_count):
            rd = 0
            for q in range(q_count):
                if q_matrix[q, i] > 0 and q_matrix[q, j] > 0:
                    rd += (1 / q_count) * (q_matrix[q, i] + q_matrix[q, j])
                else:
                    rd += 0
            rd_results[(f"C{i + 1}", f"C{j + 1}")] = rd

    return rd_results



def calculate_uocr(user_data, q_matrix):

    knowledge_count = q_matrix.shape[1]
    q_count = q_matrix.shape[0]
    uocr_results = {}

    for i in range(knowledge_count):
        for j in range(i + 1, knowledge_count):
            X_sum = 0  # 分母
            Y_sum = 0  # 分子

            for record in user_data:
                exer_id = record["exer_id"]
                score = record["score"]

                W_i = q_matrix[int(exer_id), i]
                W_j = q_matrix[int(exer_id), j]

                if W_i > 0 and W_j > 0:
                    P_Qn = 1 / q_count
                    P_QnLs = score / q_count

                    X_Qn = P_Qn * (W_i + W_j)
                    Y_Qn = P_QnLs * (W_i + W_j)

                    X_sum += X_Qn
                    Y_sum += Y_Qn

            uocr_results[(f"C{i + 1}", f"C{j + 1}")] = Y_sum / X_sum if X_sum > 0 else 0

    return uocr_results
def getQm():
    q = pd.read_csv('/root/GDPO-main/data/math1/item.csv')
    q_matrix = []
    for index, row in q.iterrows():
        item_id = row['item_id']
        knowledge_code = row['knowledge_code']
        knowledge_code = ast.literal_eval(knowledge_code)
        q_matrix.append(knowledge_code)
    return q_matrix
def computFQ(knowledge):
    d_edges = []

    with open('/root/GDPO-main/dataProsses/math1undirect_graph.txt', 'r') as file:
        for line in file:
            source, target = map(int, line.split())
            d_edges.append((source, target))

    d_edge_emb = [0.] * len(d_edges)  # d_edge_dim边个数
    for knowledge_code in knowledge:  # log['knowledge_code']Q矩阵
        knowledge_code -= 1
        for index, (source, target) in enumerate(d_edges):
            if knowledge_code == source or knowledge_code == target:
                d_edge_emb[index] = 1
    indices = [index for index, value in enumerate(d_edge_emb) if value == 1]
    extracted_elements = [d_edges[i] for i in indices]
    return extracted_elements
def RkpAndRedge(KN,KE,record,qm):
    knowledge = qm[record['exer_id']]
    true_score = record['score']
    for r in knowledge:
        if true_score == 1:
            KN[r - 1] = [0,1]
        else:
            KN[r - 1] = [1, 0]



    qf = computFQ(knowledge)
    for q in qf:
        KE[q[0]][q[1]] = float(true_score)
    return KN,KE


def process_and_calculate(csv_file_path, q_matrix_path, q_matrix_path_orig,kpn,maxre,output_dir="output"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    user_data = load_user_data(csv_file_path)
    q_matrix = load_q_matrix(q_matrix_path)
    q_matrix_orig = np.loadtxt(q_matrix_path_orig, dtype=float)
    Final_data = []
    sampledata = []
    user_groups = {}
    for record in user_data:
        user_id = record["user_id"]
        if user_id not in user_groups:
            user_groups[user_id] = []
        user_groups[user_id].append(record)
    maxRe = 0
    qm= getQm()
    for user_id, logs in user_groups.items():
        print([user_id,len(logs)])
        starttime = time.time()
        studentsample = []
        timelogs = []
        for num in range(len(logs)):
            if num == len(logs)-1 :
                break
            log = logs[num]

            timelogs.append(log)
            uoc_results = calculate_uoc(timelogs, q_matrix)
        # rd_results = calculate_rd(q_matrix)
            uocr_results = calculate_uocr(timelogs, q_matrix)

            X,Adj = DataProsses(uoc_results,uocr_results)


            re = [[] for _ in range(kpn)]
            for l in timelogs:
                exer_id = l["exer_id"]
                score = l["score"]
                t = q_matrix_orig[int(exer_id)]
                for i, j in enumerate(t):
                    if j > 0 and len(re[i]) < maxre:
                        re[i].append(score)
            for row in re:
                while len(row) < maxre:
                    row.append(-1.0)
            longest_row = max(re, key=len)
            if len(longest_row) > maxRe:
                maxRe = len(longest_row)
            re = np.array(re)



            Final_data.append((Adj,X,re,logs[num+1]))
            studentsample.append((re,logs[num+1]))
        sampledata.append(studentsample)
        endtime = time.time()
        print(endtime-starttime)







    torch.save(sampledata,'math1sample.pt')
    torch.save(Final_data, 'math1Perp.pt')









math1 = True
kpn = 11
maxre = 17
csv_file_path = "/root/GDPO-main/data/math1/math1f.csv"
q_matrix_path = "/root/GDPO-main/data/math1/_D.txt"
q_matrix_path_orig = "/root/GDPO-main/data/math1/q_m.csv"
process_and_calculate(csv_file_path, q_matrix_path,q_matrix_path_orig,kpn,maxre)
