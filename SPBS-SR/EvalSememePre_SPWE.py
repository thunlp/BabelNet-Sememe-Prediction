# coding:utf8
'''
利用synset的embedding，基于SPWE进行义原推荐
输入：所有synset(名词)的embedding，训练集synset及其义原，测试集synset
输出：测试集义原，正确率
'''
import sys
import os
import numpy as np
from numpy import linalg
import time
import random

outputMode = eval(sys.argv[1])


def ReadSysnetSememe(fileName):
    '''
    读取已经标注好义原的sysnet
    '''

    start = time.clock()
    synsetSememeDict = {}
    with open(fileName, 'r', encoding = 'utf-8') as file:
        for line in file:
            synset, sememes = line.strip().split('\t')
            synsetSememeDict[synset] = sememes.split()
    print('Have read', len(synsetSememeDict), 'synsets with sememes.')
    return synsetSememeDict

def Read_Test2id(fileName):
    '''
    读取测试集，获取正确答案
    '''
    sememeStd_test = {}
    first_relation_per_head = set()
    with open(fileName, 'r', encoding = 'utf-8') as fin:
        for line in fin:
            synset_id, sememe_id, synset_type = line.strip().split()
            if synset_id in sememeStd_test:
                sememeStd_test[synset_id].append(sememe_id)
            else:
                sememeStd_test[synset_id] = [sememe_id]
                first_relation_per_head.add((synset_id, sememe_id, synset_type))

    return sememeStd_test, first_relation_per_head


def ReadSynsetVec(fileName, synsetList):
    '''
    Read synset vectors from the word embedding file
    '''
    synsetVecDict = dict.fromkeys(synsetList, False)
    readNum = 0
    with open(fileName, 'r') as file:
        synsetNum, dimension = file.readline().strip().split()
        for line in file:
            items = line.strip().split()
            if len(items) == eval(dimension) + 1:
                readNum += 1
                synset = items[0]
                if synset in synsetList:
                    vec = np.array(list(map(eval, items[1:])))
                    if linalg.norm(vec) != 0:
                        synsetVecDict[synset] = vec / \
                            linalg.norm(vec)  # Normalization
    print('Have read', readNum, 'synsets with embeddings')
    return synsetVecDict

def ReadList(fileName):
    se = set()
    with open(fileName, 'r', encoding = 'utf-8') as file:
        for line in file:
            synset = line.strip().split()[0]
            if synset.endswith('n'):
                se.add(synset)
    return list(se)


def Get_AP(sememeStd, sememePre):
    '''
    Calculate the Average Precision of sememe prediction
    '''
    AP = 0
    hit = 0
    for i in range(len(sememePre)):
        if sememePre[i] in sememeStd:
            hit += 1
            AP += float(hit) / (i + 1)
    if AP == 0:
        print('Calculate AP Error')
        print('Sememe Standard:' + ' '.join(sememeStd))
        print('Sememe Predicted:' + ' '.join(sememePre))
        return 0
    else:
        AP /= float(len(sememeStd))
    return AP


def Get_F1(sememeStdList, sememeSelectList):
    '''
    Calculate the F1 score of sememe prediction
    '''
    TP = len(set(sememeStdList) & set(sememeSelectList))
    FP = len(sememeSelectList) - TP
    FN = len(sememeStdList) - TP
    precision = float(TP) / (TP + FN)
    recall = float(TP) / (TP + FP)
    if (precision + recall) == 0:
        return 0
    F1 = 2 * precision * recall / (precision + recall)
    return F1

#测试集内获得的标准答案
test2id_filename = '../data-noun/test.txt'
synset_answer, first_relation_per_head = Read_Test2id(test2id_filename)

print('Start to read sememes of synsts')
synsetSememeFileName = '../BabelSememe/synset_sememes.txt'
synsetSememeDict = ReadSysnetSememe(synsetSememeFileName)


print('Start to read synset vectors')
synsetVecFileName = 'synset_vec.txt'
synsetVecDict = ReadSynsetVec(synsetVecFileName, list(synsetSememeDict.keys()))


# Start Predicting Sememes
# Set hyper-parameters
K = 100  # number of nearest source words for each target word when predicting
c = 0.8  # declining coefficient
simThresh = 0.5  # threshold of chosen sememe score

numThresh = 0

start = time.clock()

synsetListAll = list(synsetSememeDict.keys())
synsetList = []
for synset in synsetListAll:
    if synset.endswith('n'):  # 这里只选名词synset
        synsetList.append(synset)
random.shuffle(synsetList)

testNum = round(len(synsetList) * 0.1)
#testSynsetList = synsetList[:testNum]
#trainSynsetList = synsetList[testNum:]

testSynsetList = ReadList('../data-noun/test.txt')
trainSynsetList = ReadList('../data-noun/train.txt')

print(len(testSynsetList))
print(len(trainSynsetList))

fout = open('sememePre_SPWE.txt','w',encoding='utf-8')

now = 0
allResults = []
for testSynset in testSynsetList:
    if type(synsetVecDict[testSynset]) == type(False):
        continue
    now += 1
    if now % 100 == 0:
        print('Have looked for sememes for %d sysnets' % now)
        print('Time Used: %f' % (time.clock() - start))

    testSynsetVec = synsetVecDict[testSynset]
    # Sort source words according the cosine similarity
    synsetSimList = []
    for trainSynset in trainSynsetList:
        if type(synsetVecDict[trainSynset]) == type(False):
            continue
        if trainSynset == testSynset:
            #print('error A', trainSynset,testSynset)
            continue
        if trainSynset not in synsetVecDict:
            #print('error B')
            continue
        
        trainSynsetVec = synsetVecDict[trainSynset]
        #print(trainSynsetVec.shape,testSynsetVec.shape)
        cosSim = np.dot(trainSynsetVec, testSynsetVec)
        #print(type(cosSim))
        synsetSimList.append((trainSynset, cosSim))
    synsetSimList.sort(key=lambda x: x[1], reverse=True)
    synsetSimList = synsetSimList[:K]

    # Calculate the score of each sememe
    sememeScore = {}
    rank = 1
    for trainSynset, cosSim in synsetSimList:
        sememes = synsetSememeDict[trainSynset]
        for sememe in sememes:
            if sememe in sememeScore:
                sememeScore[sememe] += cosSim * pow(c, rank)
            else:
                sememeScore[sememe] = cosSim * pow(c, rank)
        rank += 1
    # Save the sorted sememes and their scores
    sortedSememe = sorted(sememeScore.items(),
                          key=lambda x: x[1], reverse=True)

    sememePreList = [x[0] for x in sortedSememe]
    #sememeStdList = synsetSememeDict[testSynset]
    sememeStdList = synset_answer[testSynset]
    # Calculate MAP
    AP = Get_AP(sememeStdList, sememePreList)

    sortedSememe = sortedSememe[0:100]
    
    print(testSynset, end='\t', file = fout)
    print(round(AP,2), end='\t', file=fout)
    print(len(sememeStdList),end='\t',file = fout)
    print(end=',',file = fout)
    print(' '.join(sememeStdList),end=',',file=fout)
    for i,item in enumerate(sortedSememe):
        print(item[0],end=' ', file=fout)
    print(end=',',file = fout)
    for i,item in enumerate(sortedSememe):
        print(round(item[1],3),end=' ', file=fout)
    
    print(file=fout)
    
    

    # if AP == 1.0:
    #     print('1.0', sememeStdList, sememePreList)
    # print('AP:', AP)
    # time.sleep(1)
    # Calculate F1 score
    tmp = [x for x in sortedSememe if x[1] > simThresh]
    if tmp == []:
        # Choose the first one sememe if all the semems get scores lower than
        # the threshold
        tmp = sortedSememe[:1]
    sememeSelectList = [x[0] for x in tmp]
    numThresh += len(sememeSelectList)
    F1 = Get_F1(sememeStdList, sememeSelectList)

    allResults.append([testSynset, synsetSimList, sortedSememe, AP, F1])
fout.close()
print('Sememe Prediction Complete')

print('mAP: %f' % np.mean([x[3] for x in allResults]))
print('mean F1: %f' % np.mean([x[4] for x in allResults]))
print('numThresh:', numThresh)

# Save all the results into the file
if outputMode > 0:
    with open('SememePreResults.txt', 'w') as file:
        file.write('Synset\tAP\tF1\n')
        for testSynset, synsetSimList, sortedSememe, AP, F1 in allResults:
            file.write(testSynset + '\t' + str(AP) + '\t' + str(F1) + '\n')
            if outputMode > 1:
                file.write('\tCorrect Sememes: ' +
                           ' '.join(synsetSememeDict[testSynset]) + '\n')
                file.write('\tNeartest Synsets (Cosine Similarity): ' + ' '.join(
                    [synset + '(' + str(cosSim) + ')' for synset, cosSim in synsetSimList]) + '\n')
                file.write('\tSememes (Scores): ' + ' '.join(
                    [sememe + '(' + str(score) + ')' for sememe, score in sortedSememe]) + '\n')
