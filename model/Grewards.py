from os import error

import numpy as np
import torch

def rule_reward(graph_list,lable_list,students_num,qm):
    rewards = torch.Tensor([])

    nodeF = [item[0] for item in graph_list]
    nodeF = torch.stack(nodeF)

    nodeF = nodeF.view(students_num, 20, 11)

    raw_true = lable_list[1]


    for student in range(students_num):
        studentRecord = raw_true[student]
        raw_pred = nodeF[student]

        studentRecord = studentRecord[1:]
        raw_pred = raw_pred[: 20]
        student_rewards = torch.tensor([])
        truth = torch.tensor([])

        for i in range(19):
            record = studentRecord[i]
            kt = raw_pred[i]
            reward = 0
            for r in qm[record['item_id'] - 1]:
                truth = torch.cat((truth, torch.tensor([record['score']])))
                true_score = record['score']
                pre_score = int(kt[r - 1])
                if true_score==pre_score:
                    temp = 1
                else:
                    temp = 0

                reward = reward + temp
            if reward != 0 :
                reward = reward / len(qm[record['item_id'] - 1])
            student_rewards = torch.cat((student_rewards, torch.tensor([reward])))

        mean_value = torch.mean(student_rewards)
        student_rewards = torch.cat((student_rewards,torch.tensor([mean_value])))
        rewards = torch.cat([rewards, student_rewards.float()])
    return rewards

def computFQ(knowledge):
    d_edges = []

    with open('/root/GDPO-main/model/math1undirect_graph.txt', 'r') as file:
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

    Rkp = 0.0
    Redge = 0.0
    knowledge = qm[record['exer_id'] - 1]
    true_score = record['score']
    for r in knowledge:
        pre_score = int(KN[r - 1])
        if true_score == pre_score:
            temp = 1
        else:
            temp = 0
        Rkp += temp
    if Rkp != 0:
        Rkp = Rkp / len(knowledge)

    qf = computFQ(knowledge)
    for q in qf:
        pre_score = KE[q[0]][q[1]]
        if true_score == pre_score:
            temp = 1
        else:
            temp = 0
        Redge += temp
    if Redge != 0:
        Redge = Redge / len(qf)
    return Rkp,Redge
def average(lst):
    if not lst:
        return 0
    return sum(lst) / len(lst)
def soloRewardstudent1(graph_list,lable_list,students_num,qm):
    level1 = 0
    level2 = 0
    level3 = 0
    level4 = 0
    level5 = 0

    rewards = torch.Tensor([])

    nodeF = [item[0] for item in graph_list]
    nodeF = torch.stack(nodeF)
    nodeF = nodeF.view(students_num, 20, 11)
    edgeF = [item[1] for item in graph_list]
    edgeF = torch.stack(edgeF)
    edgeF = edgeF.view(students_num, 20, 11, 11)
    raw_true = lable_list[1]

    for student in range(students_num):
        studentRecord = raw_true[student]
        raw_predN = nodeF[student]
        raw_predF = edgeF[student]

        studentRecord = studentRecord[1:]
        raw_predN = raw_predN[: len(raw_predN)]
        raw_predF = raw_predF[: len(raw_predF)]
        student_rewards = torch.tensor([])
        sr_list = []

        for i in range(14):
            reward = 0
            record = studentRecord[i]
            KN = raw_predN[i]  # 当前时刻的点
            kE = raw_predF[i]  # 当前时刻的边
            Rkp, Redge = RkpAndRedge(KN, kE, record, qm)
            if Rkp == 0:
                reward = 0
                level1 += 1
            elif Rkp < 0.5:
                reward = 2.0
                level2 += 1
            elif Rkp >= 0.5 > Redge:
                reward = 16.0
                level3 += 1
            elif 0.5 <= Rkp < 1 and 0.5 <= Redge < 1:
                reward = 32.0
                level4 += 1
            elif (Rkp == 1 and Redge >= 0.5) or (Rkp >= 0.5 and Redge == 1):
                reward = 36.0
                level5 += 1
            else:
                print("erro", (Rkp, Redge))
            sr_list.append(reward)
            rwd = average(sr_list)
            student_rewards = torch.cat((student_rewards, torch.tensor([rwd])))

        mean_value = torch.mean(student_rewards)
        student_rewards = torch.cat((student_rewards, torch.tensor([mean_value])))
        rewards = torch.cat([rewards, student_rewards.float()])




    return rewards,np.array([level1,level2,level3,level4,level5])
def generate_sequence(inputs, window_size, alpha, decay):

    n = len(inputs)
    sequence = []

    for i in range(n):
        if i == 0:

            sequence.append(inputs[i])
        else:

            history_weights = []
            total_weight = 0.0
            for j in range(1, min(window_size, i) + 1):
                weight = (1 - alpha) * (decay ** (j - 1))
                history_weights.append(weight)
                total_weight += weight


            history_weights = [w / total_weight * (1 - alpha) for w in history_weights]


            historical_part = 0.0
            for j, w in enumerate(history_weights):
                historical_part += w * sequence[i - j - 1]


            current = alpha * inputs[i] + historical_part
            sequence.append(current)

    return torch.tensor(sequence)
def soloRewardstudent2(graph_list,lable_list,students_num,qm):
    level1 = 0
    level2 = 0
    level3 = 0
    level4 = 0
    level5 = 0

    rewards = torch.Tensor([])

    nodeF = [item[0] for item in graph_list]
    nodeF = torch.stack(nodeF)
    nodeF = nodeF.view(students_num, 20, 57)
    edgeF = [item[1] for item in graph_list]
    edgeF = torch.stack(edgeF)
    edgeF = edgeF.view(students_num, 20, 57, 57)
    raw_true = lable_list[1]

    for student in range(students_num):
        studentRecord = raw_true[student]
        raw_predN = nodeF[student]
        raw_predF = edgeF[student]

        studentRecord = studentRecord[1:]
        raw_predN = raw_predN[: len(raw_predN)]
        raw_predF = raw_predF[: len(raw_predF)]
        student_rewards = torch.tensor([])
        sr_list = []

        for i in range(19):
            reward = 0
            record = studentRecord[i]
            KN = raw_predN[i]
            kE = raw_predF[i]
            Rkp, Redge = RkpAndRedge(KN, kE, record, qm)
            if Rkp == 0:
                reward = 0
                level1 += 1
            elif Rkp < 0.5:
                reward = 2.0
                level2 += 1
            elif Rkp >= 0.5 > Redge:
                reward = 16.0
                level3 += 1
            elif 0.5 <= Rkp < 1 and 0.5 <= Redge < 1:
                reward = 32.0
                level4 += 1
            elif (Rkp == 1 and Redge >= 0.5) or (Rkp >= 0.5 and Redge == 1):
                reward = 36.0
                level5 += 1
            else:
                print("erro", (Rkp, Redge))

            sr_list.append(reward)

        rwd = generate_sequence(sr_list, window_size=3, alpha=0.7, decay=0.5)


        student_rewards = torch.cat((student_rewards, rwd))


        mean_value = torch.mean(student_rewards)
        student_rewards = torch.cat((student_rewards, torch.tensor([mean_value])))
        rewards = torch.cat([rewards, student_rewards.float()])




    return rewards,np.array([level1,level2,level3,level4,level5])

def soloReward(graph_list,lable_list,timenum,qm):
    level1 = 0
    level2 = 0
    level3 = 0
    level4 = 0
    level5 = 0

    rewards = torch.Tensor([])

    nodeF = [item[0] for item in graph_list]
    nodeF = torch.stack(nodeF)
    edgeF = [item[1] for item in graph_list]
    edgeF = torch.stack(edgeF)

    for t in range(timenum):
        reward = 0
        Record = {'user_id':lable_list['user_id'][t]+1,
                         'item_id':lable_list['exer_id'][t]+1,
                         'score':lable_list['score'][t]}
        raw_predN = nodeF[t]
        raw_predF = edgeF[t]
        Rkp,Redge = RkpAndRedge(raw_predN,raw_predF,Record,qm)
        if Rkp == 0:
            reward = 0
            level1 += 1
        elif Rkp < 0.5:
            reward = 2
            level2 += 1
        elif Rkp >= 0.5 > Redge:
            reward = 16
            level3 += 1
        elif 0.5 <= Rkp < 1 and 0.5 <= Redge < 1:
            reward = 32
            level4 += 1
        elif (Rkp == 1 and Redge >= 0.5) or (Rkp >= 0.5 and Redge == 1):
            reward = 36
            level5 += 1
        else:
            print("erro", (Rkp, Redge))
        rewards = torch.cat((rewards, torch.tensor([reward])))




    return rewards,np.array([level1,level2,level3,level4,level5])
def soloRewardsample(graph_list,lable_list,timenum,qm):
    level1 = 0
    level2 = 0
    level3 = 0
    level4 = 0
    level5 = 0

    rewards = torch.Tensor([])

    nodeF = [item[0] for item in graph_list]
    nodeF = torch.stack(nodeF)
    edgeF = [item[1] for item in graph_list]
    edgeF = torch.stack(edgeF)

    for t in range(timenum):
        reward = 0
        Record = {'user_id':lable_list[t]['user_id']+1,
                         'item_id':lable_list[t]['exer_id']+1,
                         'score':lable_list[t]['score']}
        raw_predN = nodeF[t]
        raw_predF = edgeF[t]
        Rkp,Redge = RkpAndRedge(raw_predN,raw_predF,Record,qm)
        if Rkp == 0:
            reward = 0
            level1 += 1
        elif Rkp < 0.5:
            reward = 2
            level2 += 1
        elif Rkp >= 0.5 > Redge:
            reward = 16
            level3 += 1
        elif 0.5 <= Rkp < 1 and 0.5 <= Redge < 1:
            reward = 32
            level4 += 1
        elif (Rkp == 1 and Redge >= 0.5) or (Rkp >= 0.5 and Redge == 1):
            reward = 36
            level5 += 1
        else:
            print("erro", (Rkp, Redge))
        rewards = torch.cat((rewards, torch.tensor([reward])))




    return rewards,np.array([level1,level2,level3,level4,level5])