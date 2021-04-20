rm -rf logs
mkdir logs
python main.py -m dqn -c rand_maxrand > logs/rand_maxrand_logs.txt
python main.py -m dqn -c max_maxrand > logs/max_maxrand_logs.txt
python main.py -m dqn -c maxrand_maxrand > logs/maxrand_maxrand_logs.txt
python main.py -m dqn -c minimax_maxrand > logs/minimax_maxrand_logs.txt
python main.py -m dqn -c minimaxrand_maxrand > logs/minimaxrand_maxrand_logs.txt
