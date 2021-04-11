rm -rf logs
mkdir logs
python main.py -m dqn -c minimax_max > logs/minimax_max_logs.txt
python main.py -m dqn -c minimax_rand > logs/minmax_rand_logs.txt
python main.py -m dqn -c minimaxrand_max > logs/minimaxrand_max_logs.txt
python main.py -m dqn -c minimax_minimax > logs/minimax_minimax_logs.txt
python main.py -m dqn -c minimax_rand > logs/minimax_rand_logs.txt
