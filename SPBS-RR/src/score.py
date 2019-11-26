#coding:utf-8
import numpy as np
import time
#代码中使用了id=15461作为synset和sememe的区分
SYNSET_SEMEME_DIV = 2189

#constant
TYPE_HAVE_SEMEME = 0

data_path = '../../data-noun/'

def Read_Entity2id(fileName):
	'''
	读取文件，获得名词synset的entity id
	'''
	noun_set = set()
	with open(fileName, 'r', encoding = 'utf-8') as fin:
		for line in fin:
			line = line.strip().split()
			synset = line[0]
			if synset.endswith('n'):
				noun_set.add(eval(line[1]))
	return noun_set


entity2id_filename = data_path + 'entity2id.txt'
noun_set = Read_Entity2id(entity2id_filename)


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

def Read_Entity2id(fileName):
	id2entity = {}
	with open(fileName, 'r', encoding = 'utf-8') as file:
		file.readline()
		for line in file:
			synset_id, entity_id = line.strip().split()
			id2entity[int(entity_id)] = synset_id
	return id2entity

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



# init
synsetSememeFileName = '../../BabelSememe/synset_sememes.txt'
synsetSememeDict = ReadSysnetSememe(synsetSememeFileName)

id2entityFileName = data_path + 'entity2id.txt'
id2entity = Read_Entity2id(id2entityFileName)

#测试集内获得的标准答案
test2id_filename = data_path + 'test2id.txt'
synset_answer, first_relation_per_head = Read_Test2id(test2id_filename)


def Get_AP_by_entity_id(eval_tripe, pre_id_list, score_list):
	head, tail, relation_type = eval_tripe
	if check_triple(eval_tripe) == False:
		return -1,-1,''
	AP_noun = -1
	sememeStd = synset_answer[head]

	
	sememePre = []
	sememePre_score = []
	pre_list = pre_id_list.tolist()
	for i in range(len(pre_list)):
		if pre_list[i] <= SYNSET_SEMEME_DIV:
			sememePre.append(pre_list[i])
			sememePre_score.append(score_list[i])

	#print(len(sememeStd), len(sememePre))
	AP_value = Get_AP(sememeStd, sememePre)
	if head in noun_set:
		AP_noun = AP_value
	
	#输出结果用于观察，开始
	output_str = ''
	output_str += id2entity[head] + '\t' + str(round(AP_noun, 2)) + '\t' + str(len(sememeStd)) + '\t'
	
	hit_list = []
	for i,item in enumerate(sememePre):
		if i > 300:
			break
		output_str += id2entity[item] + ' '

	output_str += '\t'

	for i,item in enumerate(sememePre_score):
		if i > 300:
			break
		output_str += str(item) + ' '

	output_str += '\t'
	# for i,item in enumerate(sememePre):
	# 	output_str += str(round(pre_score_list[i],3)) + ' '
	# output_str += '\t'
	#output_str += str(len(hit_list)) + '\t'

	
		
	output_str += '\n'
	
	#输出结果结束
	#output_str = ''
	return AP_value, AP_noun, output_str

def Get_AP(sememeStd, sememePre):
	'''
	Calculate the Average Precision of sememe prediction
	'''
	#print(sememeStd,sememePre)
	AP = 0
	hit = 0
	for i in range(len(sememePre)):
		if sememePre[i] in sememeStd:
			hit += 1
			AP += float(hit) / (i + 1)
	if AP == 0:
		#print('Calculate AP Error')
		#print('Sememe Standard:' + ' '.join(sememeStd))
		#print('Sememe Predicted:' + ' '.join(sememePre))
		return 0
	else:
		AP /= float(len(sememeStd))
	#print(AP)
	return AP

	# filter triple to cal mAP
def check_triple(triple):
	head = triple[0]
	tail = triple[1]
	relation_type = triple[2]
	if (head, tail, relation_type) not in first_relation_per_head:
		return False
	else:
		return True

def clear_AP_list(AP_list):
	AP_list = []
	return AP_list
	
def get_AP_mean(AP_list):
	#print(len(AP_list))
	arr = np.array(AP_list)
	return np.mean(arr)
