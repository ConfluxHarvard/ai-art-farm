import streamlit as st
import numpy as np
import pandas as pd
import pygsheets
from datetime import datetime
import openai
import numpy as np

from .components import prompt
from .components import fetch_data

class Params:
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def app():
    
    # SET APP PROPERTIES

    st.title("PROMPT GENERATION")
    
    # Do we want to cache this?
    # @st.cache
    def get_api_info():
        api_key = st.sidebar.text_input("Enter your OpenAI API key", "", key='api_key')
        return api_key
    openai.api_key = get_api_info()
    
    if openai.api_key != '':
        
        # FETCH MODELS
        
        @st.cache
        def fetch_models(api_key):
            all_models = fetch_data.fetch_finetunes(api_key) + fetch_data.fetch_engines(api_key)
            return all_models
            
        # SELECT PROJECT

        gc = pygsheets.authorize(service_file='pages/ai-art-farm-2c65230a4396.json')
        sheet = gc.open('AIAF_Experimental_Database')
        projects_wks = sheet.worksheet('title', 'projects')
        projects_data = projects_wks.get_all_records()
        projects_df = pd.DataFrame.from_dict(projects_data)
        project_list = { value['project_id']: value['project_alias'] for key, value in projects_df.iterrows() }
        
        selected_project = st.sidebar.selectbox("Select your project", list(project_list.values()))  
        
        with st.sidebar.expander("OR Create new project"):
            new_project_name = st.text_input("Enter project name")
            click_create_new_project = st.button("Save project")
            
        if click_create_new_project:
            last_row_project = len(projects_data)+1
            # DO NOT EDIT UNDERLYING DB, ONLY VIEWS - if projects get deleted, id will sometimes skip a value
            new_project_data = [last_row_project, new_project_name, None, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")]
            projects_wks.insert_rows(last_row_project, number=1, values=new_project_data)

        if selected_project:
            
            # SELECT PROMPT
            
            params = Params()
            project_id = projects_df.at[projects_df.index[projects_df['project_alias'] == selected_project].tolist()[0], 'project_id']        
            params.project_id = project_id
            params.selected_project = selected_project
            
            def get_data_dict(sheet_name, key, value):
                data_wks = sheet.worksheet('title', sheet_name)
                data_all = data_wks.get_all_records()
                data_df = pd.DataFrame.from_dict(data_all)
                try:
                    this_project_data = data_df[data_df['project_id_fk'] == project_id] 
                except:
                    this_project_data = pd.DataFrame()
                data_dict = { v[key] : v[value] for k, v in this_project_data.iterrows() }
                return data_wks, data_all, data_dict, data_df
            
            [prompts_wks, prompts_data, prompts_dict, prompts_df] = get_data_dict('prompts', 'prompt_id', 'prompt')    
            
            with st.expander("Add new free-form prompt"):
                new_prompt = st.text_input("New prompt")
                click_create_new_prompt = st.button("Save prompt")
                
                if click_create_new_prompt:
                    last_row_prompt = len(prompts_data)+1
                    new_prompt_data = [last_row_prompt, new_prompt, None, None, None, None, None, None, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(project_id), selected_project, None, None]
                    prompts_wks.insert_rows(last_row_prompt, number=1, values=new_prompt_data)    
            
            st.markdown("""
                        \n\n
                        ---
                        ##### OR Generate prompt using GPT3:""")

            [prompt_topics_wks, prompt_topics_data, prompt_topics_dict, prompt_topics_df] = get_data_dict('prompt_topics', 'prompt_topic', 'prompt_topic_id')   
            [prompt_styles_wks, prompt_styles_data, prompt_styles_dict, prompt_styles_df] = get_data_dict('prompt_styles', 'prompt_style', 'prompt_style_id')   
            [tags_wks, tags_data, tags_dict, tags_df] = get_data_dict('tags', 'tag_alias', 'tag_id')
            
            tcol, scol = st.columns(2)
            
            
            selected_prompt_topic = tcol.selectbox("Select from prompt topics in this project", list(prompt_topics_dict.keys()), key='prompt_topic')
            params.prompt_topic = selected_prompt_topic
            params.prompt_topic_id = prompt_topics_dict[selected_prompt_topic]
            
            with tcol.expander("OR Create new prompt topic"):
                
                prompt_topic = st.text_input("Topic")
                click_create_new_prompt_topic = st.button("Save topic")
                
                if click_create_new_prompt_topic:
                    last_row_prompt_topic = len(prompt_topics_data)+1
                    new_prompt_topic_data = [last_row_prompt_topic, prompt_topic, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(project_id), selected_project]
                    prompt_topics_wks.insert_rows(last_row_prompt_topic, number=1, values=new_prompt_topic_data)  
                    
                    last_row_tag = len(tags_data)+1
                    new_tag_data = [last_row_tag, prompt_topic, "topic", datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")]# str(project_id), selected_project]
                    tags_wks.insert_rows(last_row_tag, number=1, values=new_tag_data)  
            
            selected_prompt_style = scol.selectbox("Select from prompt styles in this project", list(prompt_styles_dict.keys()), key='prompt_style')
            params.prompt_style = selected_prompt_style
            params.prompt_style_id = prompt_styles_dict[selected_prompt_style]

            with scol.expander("OR Create new prompt style"):
                
                prompt_style = st.text_input("Style")
                click_create_new_prompt_style = st.button("Save style")

                if click_create_new_prompt_style:
                    last_row_prompt = len(prompt_styles_data)+1
                    new_prompt_data = [last_row_prompt, prompt_style, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(project_id), selected_project]
                    prompt_styles_wks.insert_rows(last_row_prompt, number=1, values=new_prompt_data)   
                    
                    last_row_tag = len(tags_data)+1
                    new_tag_data = [last_row_tag, prompt_style, "style", datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")]# str(project_id), selected_project]
                    tags_wks.insert_rows(last_row_tag, number=1, values=new_tag_data)   
            
            selected_prompt_seed = str({ 'topic': selected_prompt_topic, 'style': selected_prompt_style})
            
            # ADJUST MODEL PARAMETERS

            with st.expander("Adjust model parameters"):
                
                # @st.cache
                def get_model_info():
                    main_model_sheet = gc.open('Prompt_Model_Info_AIAF')
                    main_models_wks = main_model_sheet.worksheet('title', 'info')
                    main_models_data = main_models_wks.get_all_records()
                    main_models_df = pd.DataFrame.from_dict(main_models_data)
                    main_models_list = main_models_df['model_alias'].tolist()
                    all_models = fetch_models(openai.api_key)
                    models = main_models_list

                    return main_models_df, models
                
                main_models_df, models = get_model_info()
                    
                params.models_key = { model : main_models_df.at[main_models_df.index[main_models_df['model_alias'] == model].tolist()[0], 'model_engine'] if model in main_models_df['model_alias'].tolist() else model for model in models }
                exploration_type = st.radio("Generation Type", options=["Single Generation", "Parameter Exploration"])
                
                if exploration_type == "Single Generation":
                    params.model = [params.models_key[st.selectbox("Select model", models)]]
                    # params.temp = [st.slider("Temperature", 0.0, 1.0, 0.7)]
                    params.temp = [st.selectbox("Temperature", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]
                else:
                    params.model = [params.models_key[model] for model in st.multiselect("Select model", models)]
                    params.temp = st.multiselect("Temperature", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    st.write(params.temp)

                params.max_tokens = st.slider("Response Length", 0, 500, 250) # change standard to our length?
                params.num_output = st.slider("Number of Outputs", 1, 10, 1)
                show_more_params = st.checkbox("Show more parameters")
                
                params.stop_sequences = st.text_input("Stop sequences") if show_more_params else None
                params.top_p = st.slider("Top P", 0.0, 1.0, 1.0) if show_more_params else 1.0
                params.freq_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0) if show_more_params else 0.0
                params.pres_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0) if show_more_params else 0.0
                params.best_of = st.slider("Best Of", 0, 10, 1) if show_more_params else 1
                        
            
            # RUN EXPLORATION
            
            if st.button("Generate Prompts"):
                    
                if set(params.model).intersection(main_models_df['model_engine'].tolist()) == set(params.model):
                    
                    def format_prompts(model):
                        fields = main_models_df.at[main_models_df.index[main_models_df['model_engine'] == model ].tolist()[0], 'fields']
                        fields_formatted = ",".join([field + "=\"" + eval(selected_prompt_seed)[field] + "\"" for field in fields.split(", ")])
                        syntax = "\"" + main_models_df.at[main_models_df.index[main_models_df['model_engine'] == model ].tolist()[0], 'prepend']+"\".format(" + fields_formatted + ")"
                        # st.write(syntax)
                        formatted_prompt = eval(syntax)
                        return formatted_prompt
                    
                    params.model_prompt_topic_dict = { model : selected_prompt_topic for model in params.model }
                    params.model_prompt_style_dict = { model : selected_prompt_style for model in params.model }
                    
                    params.model_prompt_dict = { model : format_prompts(model) for model in params.model }
                    params.model_stop_dict = { model : main_models_df.at[main_models_df.index[main_models_df['model_engine'] == model ].tolist()[0], 'stop_sequences'] if model in main_models_df['model_engine'].tolist() else None for model in params.model }
                
                else:
                    
                    st.code("ERROR: No format available for one or more of the selected models. Please select models that are in the 'Model Info' sheet if you want to auto-format.")

                # OPTION TO SEE EXPLORATION PARAMETERS
                with st.expander("See Parameters"):
                    st.write(params.__dict__)

                with st.spinner('Generating..'):

                    full_response, parsed_response = prompt.generate(params, sheet, openai.api_key)
                    
                    # OPTIONS TO DISPLAY DETAILED RESPONSES
                    # with st.expander("See Full API Response"):
                    #     st.code(full_response)
                    # with st.expander("See Data Table Response"):
                    #     st.write(parsed_response)
                        
                    def display_completion(full_completion, display_key):
                        with st.expander(full_completion['prompt']):
                            completion_parameters = full_completion.drop('prompt')
                            st.write(completion_parameters.to_dict(), key=display_key)
                    
                    for i in range(len(parsed_response)):
                        display_completion(parsed_response.loc[i], i)

                    st.success('Done!')