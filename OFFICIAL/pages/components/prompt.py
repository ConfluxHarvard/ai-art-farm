import openai
import itertools
import pandas as pd
from datetime import datetime

def generate(params, sheet, apikey):
    
    completions_wks = sheet.worksheet('title', 'prompts')
    
    openai.api_key = apikey
    
    project_id = params.project_id
    selected_project = params.selected_project
    # selected_prompt_seed = params.selected_prompt_seed
    
    # prompt_id = params.prompt_id
    # prompt_seed_id = params.prompt_seed_id
    prompt_topic_id = params.prompt_topic_id
    prompt_topic = params.prompt_topic
    prompt_style_id = params.prompt_style_id
    prompt_style = params.prompt_style
    
    model = params.model
    temp = params.temp
    max_tokens = params.max_tokens
    top_p = params.top_p
    frequency_penalty = params.freq_penalty
    presence_penalty = params.pres_penalty
    num_output = params.num_output
    
    model_prompt_dict = params.model_prompt_dict
    model_stop_dict = params.model_stop_dict
    
    full_response = []
    parsed_response = pd.DataFrame(columns=['prompt', 'model', 'model alias', 'finish_reason', 'temperature', 'max_tokens', 'other_parameters', 'prompt_topic', 'prompt_style'])
    
    for model, temp in itertools.product(model, temp):
        stop=model_stop_dict[model] if model_stop_dict[model] and model_stop_dict[model] != "" else None
        
        response = openai.Completion.create(
        model=model,
        prompt=model_prompt_dict[model],
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=num_output,
        stop=stop,
        )
        
        # response['prompt'] = model_prompt_dict[model]
        # response['prompt_seed'] = params.model_prompt_seed_dict[model] if params.model_prompt_seed_dict else None
        response['model_alias'] = [alias for alias, i in params.models_key.items() if i == model][0]
        
        other_parameters = {
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
            'stop': eval( "'" + stop + "'") if stop else None,
        }
    
        for i in range(num_output):
            
            completions_data = completions_wks.get_all_records()
            last_row_completions = len(completions_data)+1
            new_completion_data = [str(last_row_completions), response['choices'][i]['text'], model, response['model_alias'], response['choices'][i]['finish_reason'], temp, max_tokens, str(other_parameters), datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(project_id), selected_project, prompt_topic_id, prompt_topic, prompt_style_id, prompt_style]
            parsed_response.loc[len(parsed_response)] = new_completion_data[1:8] + [new_completion_data[-3]] + [new_completion_data[-1]]
            completions_wks.insert_rows(last_row_completions, number=1, values=new_completion_data)

        full_response.append(response)
        
    return full_response, parsed_response
