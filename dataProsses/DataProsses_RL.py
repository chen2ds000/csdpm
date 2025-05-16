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
        next(csv_reader)
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
def process_and_calculate(csv_file_path, q_matrix_path, q_matrix_path_orig,kpn,maxre,output_dir="output"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    user_data = load_user_data(csv_file_path)
    q_matrix = load_q_matrix(q_matrix_path)
    q_matrix_orig = np.loadtxt(q_matrix_path_orig, dtype=float)
    Final_data = []

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
        relist = []
        timelogs = []
        relogs = []
        for log  in logs:
            timelogs.append(log)
            relog={'exer_id':log['exer_id']+1, 'score':log['score'],'user_id':log['user_id']+1}

            relogs.append(relog)
            re = [[] for _ in range(kpn)]
            for l in timelogs:
                exer_id = l["exer_id"]
                score = l["score"]
                t = q_matrix_orig[int(exer_id)]
                for i, j in enumerate(t):
                    if j > 0 and len(re[i]) < maxre:

                        re[i].append(score)
            longest_row = max(re, key=len)
            for row in re:
                while len(row) < maxre:
                    row.append(-1.0)

            if len(longest_row) > maxRe:
                maxRe = len(longest_row)
            re = np.array(re)
            relist.append(re)
        uoc_results = calculate_uoc(timelogs, q_matrix)
        uocr_results = calculate_uocr(timelogs, q_matrix)
        X, Adj = DataProsses(uoc_results, uocr_results)
        relogs.append(relist)
        Final_data.append((Adj,X,relist[0],relogs))
        endtime = time.time()
        print(endtime-starttime)
    print(maxRe)
    torch.save(Final_data, 'math1.pt')









math1 = True
kpn = 11
maxre = 20
csv_file_path = "/root/GDPO-main/data/nips/nips50.csv"  #  CSV path
q_matrix_path = "/root/GDPO-main/data/nips/_D.txt"  # Qmatrixfloat path
q_matrix_path_orig = "/root/GDPO-main/data/nips/q_matrix.txt" # Qmatrix path
process_and_calculate(csv_file_path, q_matrix_path,q_matrix_path_orig,kpn,maxre)
