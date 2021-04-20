rm -rf logs
mkdir logs

python main.py -m dqn -c rand_maxrand > logs/rand_maxrand_logs.txt
python main.py -m dqn -c max_maxrand > logs/max_maxrand_logs.txt
python main.py -m dqn -c maxrand_maxrand > logs/maxrand_maxrand_logs.txt
python main.py -m dqn -c minimax_maxrand > logs/minimax_maxrand_logs.txt
python main.py -m dqn -c minimaxrand_maxrand > logs/minimaxrand_maxrand_logs.txt

python main.py -m dqn -c rand_rand > logs/rand_rand_logs.txt
python main.py -m dqn -c max_rand > logs/max_rand_logs.txt
python main.py -m dqn -c maxrand_rand > logs/maxrand_rand_logs.txt
python main.py -m dqn -c minimax_rand > logs/minimax_rand_logs.txt
python main.py -m dqn -c minimaxrand_rand > logs/minimaxrand_rand_logs.txt
