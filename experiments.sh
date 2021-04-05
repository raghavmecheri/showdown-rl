rm -rf logs
mkdir logs
python main.py -m dqn -c rand_max > logs/rand_max_logs.txt
python main.py -m dqn -c max_max > logs/max_max_logs.txt
python main.py -m dqn -c maxrand_max > logs/maxrand_max_logs.txt
python main.py -m dqn -c maxrand_maxrand > logs/maxrand_maxrand_logs.txt
