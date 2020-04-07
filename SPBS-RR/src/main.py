from dataset import KnowledgeGraph
from model import TransE

import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir', type=str, default='../data/FB15k/')
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--margin_value', type=float, default=1.0)
    parser.add_argument('--score_func', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=4800)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_generator', type=int, default=24)
    parser.add_argument('--n_rank_calculator', type=int, default=24)
    parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
    parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=10)
    args = parser.parse_args()
    print(args)
    
    kg = KnowledgeGraph(data_dir=args.data_dir)
    kge_model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                       score_func=args.score_func, batch_size=args.batch_size, learning_rate=args.learning_rate,
                       n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=sess_config) as sess:
        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')
        sess.graph.finalize()
        kge_model.check_norm(session=sess)
        summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=sess.graph)
        for epoch in range(args.max_epoch):
            print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
            kge_model.launch_training(session=sess, summary_writer=summary_writer)
            if (epoch + 1) % args.eval_freq == 0:
                kge_model.launch_evaluation(session=sess)

                # embedding_val = kge_model.entity_embedding
                # vocab_size, dim = embedding_val.shape
                # embedding_val = embedding_val.eval(session = sess)
                # with open('entity_embedding' + str(epoch) + '.txt', 'w') as file_:
                #     for i in range(vocab_size):
                #       embed = embedding_val[i, :]
                #       file_.write('%s\n' % (' '.join(map(str, embed))))

                # relation_val = kge_model.relation_embedding
                # vocab_size, dim = relation_val.shape
                # relation_val = relation_val.eval(session = sess)
                # with open('relation_embedding' + str(epoch) + '.txt', 'w') as file_:
                #     for i in range(vocab_size):
                #       embed = relation_val[i, :]
                #       file_.write('%s\n' % (' '.join(map(str, embed))))


if __name__ == '__main__':
    main()
