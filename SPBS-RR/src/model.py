import math
import timeit
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from dataset import KnowledgeGraph
import score
import tracemalloc
import gc

import objgraph


class TransE:
	def __init__(self, kg: KnowledgeGraph,
				 embedding_dim, margin_value, score_func,
				 batch_size, learning_rate, n_generator, n_rank_calculator,pre_trained=[]):
		self.kg = kg
		self.embedding_dim = embedding_dim
		self.margin_value = margin_value
		self.score_func = score_func
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.n_generator = n_generator
		self.n_rank_calculator = n_rank_calculator
		'''ops for training'''
		self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
		self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
		self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
		self.train_op = None
		self.loss = None
		self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
		self.merge = None
		'''ops for evaluation'''
		self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
		self.idx_head_prediction = tf.placeholder(dtype=tf.int32, shape=[self.kg.n_entity])
		self.idx_tail_prediction = tf.placeholder(dtype=tf.int32, shape=[self.kg.n_entity])
		self.score_tail_prediction = tf.placeholder(dtype=tf.float32, shape=[self.kg.n_entity])

		self.t_idx_head_prediction = tf.placeholder(dtype=tf.int32, shape=[self.kg.n_entity])
		self.t_idx_tail_prediction = tf.placeholder(dtype=tf.int32, shape=[self.kg.n_entity])
		
		self.triple_ss = tf.placeholder(dtype=tf.int32, shape=[None, kg.max_sememe_num])
		
		self.alpha = 0.93
		self.beta = 1.0 - self.alpha

		tf.set_random_seed(1234)

		print('AP_init')
		self.AP_list = []
		'''embeddings'''
		bound = 6 / math.sqrt(self.embedding_dim)
		with tf.variable_scope('embedding'):
			self.entity_embedding = tf.get_variable(name='entity',
													shape=[kg.n_entity, self.embedding_dim],
													initializer=tf.random_uniform_initializer(minval=-bound,
																							  maxval=bound))
			self.zero_embedding = tf.get_variable(name='zero',
													shape=[kg.n_entity + 1, self.embedding_dim],
													initializer=tf.random_uniform_initializer(minval=-0.0,
																							  maxval=0.0), trainable = False)
			self.M = tf.get_variable(name='M',
									shape=[self.embedding_dim, self.embedding_dim],
									initializer=tf.random_uniform_initializer(minval=-1.0,
																				maxval=1.0))
			#self.entity_embedding = tf.get_variable(name="entity", initializer=pre_trained) 
			print('tfnn:',tf.nn.embedding_lookup(self.entity_embedding, 0))
			tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
			self.relation_embedding = tf.get_variable(name='relation',
													  shape=[kg.n_relation, self.embedding_dim],
													  initializer=tf.random_uniform_initializer(minval=-bound,
																								maxval=bound))
			self.sum_relation = tf.get_variable(name='sum_relation',
												  shape=[self.embedding_dim],
												  initializer=tf.random_uniform_initializer(minval=-bound,
																							maxval=bound))
			tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
			tf.summary.histogram(name=self.M.op.name, values=self.M)
		self.build_graph()
		self.build_eval_graph()

	def build_graph(self):
		with tf.name_scope('normalization'):
			self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
			self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
		with tf.name_scope('training'):
			with tf.name_scope('Mtraining'):
				t_synset = self.triple_ss[:,0]
				t_sememes = self.triple_ss[:,1:]


				# print('t_synset:', t_synset)
				# print('t_sememes:', t_sememes)
				col_sememes = tf.transpose(t_sememes)
				# print('col_sememes:', col_sememes)
				t_bool = (col_sememes == self.kg.n_entity + 1)

				
			with tf.name_scope('Mlookup'):
				t_synset_emb = tf.nn.embedding_lookup(self.entity_embedding, t_synset)
				valid_sememe_emb = tf.nn.embedding_lookup(self.entity_embedding, col_sememes)
				zero_emb = tf.nn.embedding_lookup(self.zero_embedding, col_sememes)

				valid_sememe_emb = tf.nn.l2_normalize(valid_sememe_emb, dim=1)
				
				sememe_emb = tf.where(t_bool, zero_emb, valid_sememe_emb)
				#sememe_emb = tf.where(t_bool, valid_sememe_emb, zero_emb)
				# print('sememe_emb:', sememe_emb)

			with tf.name_scope('loss2'):
				sememe_sum = tf.reduce_sum(sememe_emb, reduction_indices=0)

				rel_emb = self.sum_relation

				#synset_dot =  tf.matmul(t_synset_emb, self.M)
				synset_dot = t_synset_emb
				synset_dot = tf.nn.l2_normalize(synset_dot, dim=1)

				# print('synset_dot:', synset_dot)
				distance = synset_dot + rel_emb - sememe_sum

				self.loss2 = tf.clip_by_value(tf.reduce_sum(tf.nn.relu(tf.abs(distance))), 1e-7, 1e5)
				
				print('loss2:', self.loss2)

			distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
			self.loss1 = self.calculate_loss(distance_pos, distance_neg, self.margin)
			
			self.loss = self.alpha * self.loss1 + self.beta * self.loss2

			tf.summary.scalar(name='loss1', tensor=self.loss1)
			tf.summary.scalar(name='loss2', tensor=self.loss2)
			tf.summary.scalar(name='loss_sum', tensor=self.loss)

			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
			self.merge = tf.summary.merge_all()

	def build_eval_graph(self):
		with tf.name_scope('evaluation'):
			self.idx_head_prediction, self.idx_tail_prediction, self.score_tail_prediction = self.evaluate(self.eval_triple)

	def infer(self, triple_pos, triple_neg):
		with tf.name_scope('lookup'):
			print('triple_pos', triple_pos)
			print('triple_pos[:, 0]', triple_pos[:, 0])
			head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
			tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
			relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
			head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
			tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
			relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
		with tf.name_scope('link'):
			distance_pos = head_pos + relation_pos - tail_pos
			distance_neg = head_neg + relation_neg - tail_neg
		return distance_pos, distance_neg

	def calculate_loss(self, distance_pos, distance_neg, margin):
		with tf.name_scope('loss'):
			if self.score_func == 'L1':  # L1 score
				score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
				score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
			else:  # L2 score
				score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
				score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
			loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
		return loss

	def evaluate(self, eval_triple):
		with tf.name_scope('lookup'):
			head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
			tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
			relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
		with tf.name_scope('link'):
			distance_head_prediction = self.entity_embedding + relation - tail
			distance_tail_prediction = head + relation - self.entity_embedding
		with tf.name_scope('rank'):
			if self.score_func == 'L1':  # L1 score
				score_head_prediction, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
													 k=self.kg.n_entity)
				score_tail_prediction, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
													 k=self.kg.n_entity)
			else:  # L2 score
				_, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
													 k=self.kg.n_entity)
				_, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
													 k=self.kg.n_entity)
		return idx_head_prediction, idx_tail_prediction, score_tail_prediction

	def launch_training(self, session, summary_writer):
		raw_batch_queue = mp.Queue()
		training_batch_queue = mp.Queue()
		for _ in range(self.n_generator):
			mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
																	   'out_queue': training_batch_queue}).start()
		print('-----Start training-----')
		start = timeit.default_timer()
		n_batch = 0
		for raw_batch in self.kg.next_raw_batch(self.batch_size):
			raw_batch_queue.put(raw_batch)
			n_batch += 1
		for _ in range(self.n_generator):
			raw_batch_queue.put(None)
		print('-----Constructing training batches-----')
		epoch_loss = 0
		n_used_triple = 0
		n_used_ss = 0
		for i in range(n_batch):
			batch_pos, batch_neg, ss_batch = training_batch_queue.get()
			batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op, self.merge],
												 feed_dict={self.triple_pos: batch_pos,
															self.triple_neg: batch_neg,
															self.triple_ss: ss_batch,
															self.margin: [self.margin_value] * len(batch_pos)})
			summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
			epoch_loss += batch_loss
			n_used_triple += len(batch_pos)
			n_used_ss += len(ss_batch)
			print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
																			n_used_triple,
																			n_used_ss,
																			self.kg.n_training_triple,
																			batch_loss / len(batch_pos)), end='\r')
		print()
		print('epoch loss: {:.3f}'.format(epoch_loss))
		print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
		print('-----Finish training-----')
		
		self.check_norm(session=session)

	def launch_evaluation(self, session):
		eval_result_queue = mp.JoinableQueue()
		rank_result_queue = mp.Queue()
		print('-----Start evaluation-----')
		start = timeit.default_timer()
		mp_list = []
		for _ in range(self.n_rank_calculator):
			tempp = mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
														   'out_queue': rank_result_queue})
			tempp.start()
			mp_list.append(tempp)
		n_used_eval_triple = 0
		for eval_triple in self.kg.test_triples:
			self.t_idx_head_prediction, self.t_idx_tail_prediction, self.t_score_tail_prediction = session.run(fetches=[self.idx_head_prediction,
																			self.idx_tail_prediction, self.score_tail_prediction],
																   feed_dict={self.eval_triple: eval_triple})
			eval_result_queue.put((eval_triple, self.t_idx_head_prediction, self.t_idx_tail_prediction, self.t_score_tail_prediction))
			n_used_eval_triple += 1
			print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
															   n_used_eval_triple,
															   self.kg.n_test_triple), end='\r')
		print()
		for _ in range(self.n_rank_calculator):
			eval_result_queue.put(None)
		print('-----Joining all rank calculator-----')
		eval_result_queue.join()
		print('-----All rank calculation accomplished-----')
		print('-----Obtaining evaluation results-----')
		'''Raw'''
		head_meanrank_raw = 0
		head_hits10_raw = 0
		tail_meanrank_raw = 0
		tail_hits10_raw = 0
		'''Filter'''
		head_meanrank_filter = 0
		head_hits10_filter = 0
		tail_meanrank_filter = 0
		tail_hits10_filter = 0
		AP_list = []
		AP_n_list = []

		
		fout = open('sememePre_TransE.txt', 'w', encoding = 'utf-8')
		for _ in range(n_used_eval_triple):
			head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter, AP_value, AP_n_value, output_str = rank_result_queue.get()
			head_meanrank_raw += head_rank_raw
			if head_rank_raw < 10:
				head_hits10_raw += 1
			tail_meanrank_raw += tail_rank_raw
			if tail_rank_raw < 10:
				tail_hits10_raw += 1
			head_meanrank_filter += head_rank_filter
			if head_rank_filter < 10:
				head_hits10_filter += 1
			tail_meanrank_filter += tail_rank_filter
			if tail_rank_filter < 10:
				tail_hits10_filter += 1
			if AP_value != -1:
				AP_list.append(AP_value)

			if AP_n_value != -1:
				AP_n_list.append(AP_n_value)
				fout.write(output_str)

	   
		print(rank_result_queue.qsize())
		print('-----Raw-----')
		head_meanrank_raw /= n_used_eval_triple
		head_hits10_raw /= n_used_eval_triple
		tail_meanrank_raw /= n_used_eval_triple
		tail_hits10_raw /= n_used_eval_triple
		print('-----Head prediction-----')
		print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
		print('-----Tail prediction-----')
		print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
		print('------Average------')
		print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
														 (head_hits10_raw + tail_hits10_raw) / 2))
		print('-----Filter-----')
		head_meanrank_filter /= n_used_eval_triple
		head_hits10_filter /= n_used_eval_triple
		tail_meanrank_filter /= n_used_eval_triple
		tail_hits10_filter /= n_used_eval_triple
		#print('-----Head prediction-----')
		#print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
		print('-----Tail prediction-----')
		print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
		#print('-----Average-----')
		#print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
		#                                                 (head_hits10_filter + tail_hits10_filter) / 2))
		print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
		mAP = score.get_AP_mean(AP_list)
		mAP_n = score.get_AP_mean(AP_n_list)
		fout.write(str(mAP_n))
		fout.close()
		gc.collect()
		print('mAP:', mAP)
		print('mAP_n:', mAP_n)
		print('-----Finish evaluation-----')



	def calculate_rank(self, in_queue, out_queue):
		while True:
			#search point
			idx_predictions = in_queue.get()
			if idx_predictions is None:
				in_queue.task_done()
				gc.collect()
				return
			else:
				eval_triple, idx_head_prediction, idx_tail_prediction, score_tail_prediction = idx_predictions
				head, tail, relation = eval_triple
				AP_value, AP_n_value, output_str = score.Get_AP_by_entity_id(eval_triple,idx_tail_prediction[::-1], score_tail_prediction[::-1])
				
				# print('AP_value',AP_value)
				# mAP = score.get_AP_mean(self.AP_list)
				# print('mAP:', mAP)
				head_rank_raw = 0
				tail_rank_raw = 0
				head_rank_filter = 0
				tail_rank_filter = 0
				for candidate in idx_head_prediction[::-1]:
					if candidate == head:
						break
					else:
						head_rank_raw += 1
						if (candidate, tail, relation) in self.kg.golden_triple_pool:
							continue
						else:
							head_rank_filter += 1
				for candidate in idx_tail_prediction[::-1]:
					if candidate == tail:
						break
					else:
						tail_rank_raw += 1
						if (head, candidate, relation) in self.kg.golden_triple_pool:
							continue
						else:
							tail_rank_filter += 1
				out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter, AP_value, AP_n_value, output_str))
				in_queue.task_done()


	def check_norm(self, session):
		print('-----Check norm-----')
		entity_embedding = self.entity_embedding.eval(session=session)
		relation_embedding = self.relation_embedding.eval(session=session)
		entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
		relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
		#print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))
