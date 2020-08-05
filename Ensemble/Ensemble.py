#coding:utf-8
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



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


#为dict插入key和value
def insert(dictObject, key, value):
	if key not in dictObject:
		dictObject[key] = [value]
	else:
		dictObject[key].append(value)

data_path = '../data-noun/'

#获得参数
parser = ArgumentParser("", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument('--alp', type=float, default=0.55)
args = parser.parse_args()


#获得entity2id和id2entity
def Read_Entity2id(fileName):
	'''
	读取文件，获得名词synset的entity id
	'''
	entity2id = {}
	id2entity = {}
	with open(fileName, 'r', encoding = 'utf-8') as fin:
		for line in fin:
			line = line.strip().split()
			synset, sid = line
			entity2id[synset] = eval(sid)
			id2entity[eval(sid)] = synset
	return entity2id, id2entity

#获得测试题答案
def Read_Test2id(fileName):
	'''
	读取测试集，获取正确答案
	'''
	sememeStd_test = {}
	first_relation_per_head = set()
	with open(fileName, 'r', encoding = 'utf-8') as fin:
		for line in fin:
			synset_id, sememe_id, synset_type = line.strip().split()
			synset_id = eval(synset_id)
			sememe_id = eval(sememe_id)
			synset_type = eval(synset_type)
			if synset_id in sememeStd_test:
				sememeStd_test[synset_id].append(sememe_id)
			else:
				sememeStd_test[synset_id] = [sememe_id]
				first_relation_per_head.add((synset_id, sememe_id, synset_type))

	return sememeStd_test, first_relation_per_head

id2entityFileName = data_path + 'entity2id.txt'
entity2id, id2entity = Read_Entity2id(id2entityFileName)

#测试集内获得的标准答案
test2id_filename = data_path + 'test2id.txt'
testSet, first_relation_per_head = Read_Test2id(test2id_filename)


dataDir = data_path

transe = open('sememePre_TransE.txt',encoding ='utf-8')
spwe = open('sememePre_SPWE.txt',encoding ='utf-8')


#读取spwe答案
spwe_answer = {}
for line in spwe:
	line = line.strip().split(',')
	sememeStd = line[1].strip().split()
	sememe_name = line[2].strip().split()
	score = line[3].strip().split()
	line = line[0].strip().split()
	synset_id = line[0]
	AP = line[1]
	sememeStdlen = line[2]
	div_index = 0
	for i, item in enumerate(line[3:]):
		if item.find('.') != -1:
			div_index = i
			break
	
	spwe_answer[synset_id] = {'synset_id': synset_id, 'AP': AP, 'sememeStd':sememeStd,'sememeStdlen': sememeStdlen, 'sememe_name': sememe_name, 'score': score}

#读取transE答案
transe_answer = {}
for line in transe:
	line = line.strip().split()
	synset_id = line[0]
	AP = line[1]
	sememeStdlen = line[2]
	div = 304
	sememe_name = line[3:div]
	sememe_score = line[div:]
	transe_answer[synset_id] = {'AP': AP, 'sememeStdlen': sememeStdlen, 'sememe_name': sememe_name, 'sememe_score': sememe_score}

AP_list = []
F1_list = []
numThresh = 0
		

#仅评测TransE
if args.alp == 1.0:
	for synset_id in testSet.keys():
		synset_name = id2entity[synset_id]
		transe_item = transe_answer[synset_name]
		#print(transe_item, synset_name)
		POS = synset_name[-1]

		sememeStd = testSet[synset_id]
		
		#id转name
		new_sememeStd = []
		for item in sememeStd:
			new_sememeStd.append(id2entity[item])
		sememeStd = new_sememeStd

		sememe_answer = {}
		transe_pair = transe_item['sememe_name']
		for i in range(len(transe_pair)):
			sememe = transe_pair[i]
			if sememe in sememe_answer:
				sememe_answer[sememe] += 1.0*(1/(i+1))
			else:
				sememe_answer[sememe] = 1.0*(1/(i+1))

		sememe_answer = sorted(sememe_answer.items(), key= lambda x : x[1],reverse = True)
		
		final_answer[synset_name] = sememe_answer

		sememePre = [x[0] for x in sememe_answer]
		AP = Get_AP(sememeStd, sememePre)


		AP_list.append(AP)
		mAP_pos[POS].append(AP)

		simThresh = 0.32
		
		tmp = [x[0] for x in sememe_answer if x[1] > simThresh]
		if tmp == []:
			tmp = sememe_answer[:1]
		
		
		
		numThresh += len(tmp)
		
		F1 = Get_F1(sememeStd, tmp)

		F1_list.append(F1)
		

else:
	for synset_id in testSet.keys():


		if synset_id not in id2entity.keys():
			continue
		#print(synset_id)
		synset_name = id2entity[synset_id]
		POS = synset_name[-1]
		

		if synset_name not in spwe_answer.keys():
			continue


		spwe_item = spwe_answer[synset_name]
		transe_item = transe_answer[synset_name]
		sememeStd = spwe_item['sememeStd']
		sememeStd = testSet[synset_id]

		spwe_pair = zip(spwe_item['sememe_name'],spwe_item['score'])
		transe_pair = transe_item['sememe_name']

		spwe_pair = list(spwe_pair)
		transe_pair = list(transe_pair)

		sememe_answer = {}

		alpha = args.alp
		beta = 1.0 - alpha
		for i in range(len(spwe_pair)):
			sememe, score = spwe_pair[i]
			if sememe in sememe_answer:
				sememe_answer[sememe] += alpha*(1/(i+1))
			else:
				sememe_answer[sememe] = alpha*(1/(i+1))

		for i in range(len(transe_pair)):
			sememe = transe_pair[i]
			if sememe in sememe_answer:
				sememe_answer[sememe] += beta*(1/(i+1))
			else:
				sememe_answer[sememe] = beta*(1/(i+1))

		#print(sememe_answer)
		sememe_answer = sorted(sememe_answer.items(), key= lambda x : x[1],reverse = True)
		
		

		sememePre = [x[0] for x in sememe_answer]

		new_Std = []
		for i, item in enumerate(sememeStd):
			new_Std.append(id2entity[item])
		sememeStd = new_Std

		AP = Get_AP(sememeStd, sememePre)

		print(AP)		
		AP_list.append(AP)

		simThresh = 0.32
		tmp = [x[0] for x in sememe_answer if x[1] > simThresh]
		if tmp == []:
			# Choose the first one sememe if all the semems get scores lower than
			# the threshold
			tmp = sememe_answer[:1]
		numThresh += len(tmp)
		
		F1 = Get_F1(sememeStd, tmp)
		F1_list.append(F1)








 
#输出ensemble结果值		
print(np.mean(np.array(AP_list)))
print(np.mean(np.array(F1_list)))
print(numThresh)

