import os
import openai
import itertools
import sqlite3

def fetch_finetunes(api_key):
    
    openai.api_key = api_key
    
    fine_tunes = openai.FineTune.list()
    fine_tunes_list = []
    for i in range(len(fine_tunes['data'])):
        fine_tunes_list.append(fine_tunes['data'][i]['fine_tuned_model'])
    fine_tunes_list = [ft for ft in fine_tunes_list if ft]
    return fine_tunes_list

def fetch_engines(api_key):
    
    openai.api_key = api_key
    
    engines = openai.Engine.list()
    engine_list = []
    for i in range(len(engines['data'])):
        engine_list.append(engines['data'][i]['id'])
    return engine_list