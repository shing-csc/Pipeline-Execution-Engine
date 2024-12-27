#!/bin/bash

mkdir -p logs

python main.py template_task1 > logs/template_task1.log
python main.py task1 > logs/task1.log

python main.py template_task2 > logs/template_task2.log
python main.py task2 > logs/task2.log

python main.py template_task3 > logs/template_task3.log
python main.py task3 > logs/task3.log
