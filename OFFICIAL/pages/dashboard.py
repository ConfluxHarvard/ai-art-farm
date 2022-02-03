# collapse Deploy into Design? 

import streamlit as st
import numpy as np
import pandas as pd
import pygsheets
from datetime import datetime
import openai
import numpy as np
import itertools
import time
from re import sub

import s3fs
import os
import boto3

# from .components import prompt
# from .components import fetch_data
# from ...archive import art

class Params:
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def app():
    
    # SET APP PROPERTIES
    
    # apptitle = 'AI ART FARM'
    # st.set_page_config(page_title=apptitle, page_icon=":brain:")
    st.title("DASHBOARD")
    
        
    # SELECT PROJECT

    gc = pygsheets.authorize(service_file='pages/ai-art-farm-2c65230a4396.json')
    sheet = gc.open('AIAF_Experimental_Database')
    projects_wks = sheet.worksheet('title', 'projects')
    projects_data = projects_wks.get_all_records()
    projects_df = pd.DataFrame.from_dict(projects_data)
    project_list = { value['project_id']: value['project_alias'] for key, value in projects_df.iterrows() }
    
    selected_project = st.sidebar.selectbox("Select your project", list(project_list.values()))  
    
    with st.sidebar.expander("Create new project"):
        new_project_name = st.text_input("Enter new project name")
        click_create_new_project = st.button("Create new_project")
        
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

        # change semantics - this is generally used for any data grab from sheet, prompts, images, etc. included
        # def get_prompt_list(sheet_name, value_key1, value_key2):
        #     prompts_wks = sheet.worksheet('title', sheet_name)
        #     prompts_data = prompts_wks.get_all_records()
        #     prompts_df = pd.DataFrame.from_dict(prompts_data)
        #     try:
        #         this_project_prompts = prompts_df[prompts_df['project_id'] == project_id] 
        #     except:
        #         this_project_prompts = pd.DataFrame()
        #     # st.write(this_project_prompts)
        #     prompt_list = { value[value_key1] : value[value_key2] for key, value in this_project_prompts.iterrows() }
        #     return prompts_wks, prompts_data, prompt_list, prompts_df
        
        st.markdown("""
                ### Deploy Experiment""")
        
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
        
        
        # [prompts_wks, prompts_data, prompts_dict, prompts_df] = get_data_dict('prompts', 'prompt', 'prompt_seed_fd')  
        # [images_wks, images_data, images_dict, images_df] = get_data_dict('input_imgs', 'input_img_alias', 'input_img_url')

        # with st.expander("Add new prompt"): 
                
        #     new_prompt = st.text_input("New prompt")
        #     click_create_new_prompt = st.button("Save prompt")
                
        #     if click_create_new_prompt:
        #         last_row_prompt = len(prompts_data)+1
        #         new_prompt_data = [last_row_prompt, new_prompt, None, None, None, None, None, None, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(project_id), selected_project, None, None]
        #         prompts_wks.insert_rows(last_row_prompt, number=1, values=new_prompt_data)    
                
        # with st.expander("Add new reference image"):
        #     new_image_alias = st.text_input("New image alias")
        #     new_image_url = st.text_input("New image url")
        #     st.image(new_image_url, caption=new_image_alias if new_image_alias else "Untitled") if new_image_url else None
        #     click_create_new_image = st.button("Save image")
            
        #     if click_create_new_image:
        #         last_row_image = len(images_data)+1
        #         new_image_data = [last_row_image, new_image_alias, new_image_url, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(project_id), selected_project]
        #         images_wks.insert_rows(last_row_image, number=1, values=new_image_data)  
        
        
        # st.markdown("""
        #     \n\n
        #     ---
        #     ##### Design your Experiment:""")
        
        def get_experiment_dict(sheet_name, key, value):
            data_wks = sheet.worksheet('title', sheet_name)
            data_all = data_wks.get_all_records()
            data_df = pd.DataFrame.from_dict(data_all)
            try:
                this_project_data = data_df[data_df['project_id_fk'] == project_id] 
                this_project_data_unstarted = this_project_data[this_project_data['experiment_status'] == 'Not Started']
            except:
                this_project_data_unstarted = pd.DataFrame()
            data_dict = { v[key] : v[value] for k, v in this_project_data_unstarted.iterrows() }
            return data_wks, data_all, data_dict, data_df
        
        [experiments_wks, experiments_data, experiments_dict, experiments_df] = get_experiment_dict('experiments', 'experiment_id', 'experiment_alias')  
        selected_experiment = st.selectbox("Select experiment to deploy", list(experiments_dict.values()), help="Only \"Unstarted\" experiments are shown")  
    
        # with st.expander("Create new experiment"):
        #     new_experiment_name = st.text_input("Enter experiment name")
        #     click_create_new_experiment = st.button("Save experiment")
            
        # if click_create_new_experiment:
        #     last_row_experiment = len(experiments_data)+1
        #     # DO NOT EDIT UNDERLYING DB, ONLY VIEWS - if projects get deleted, id will sometimes skip a value
        #     new_experiment_data = [str(last_row_experiment), new_experiment_name, None, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), None, str(project_id), selected_project]
        #     experiments_wks.insert_rows(last_row_experiment, number=1, values=new_experiment_data)

        def display_run(full_completion, display_key):
            with st.expander(full_completion['run_prompt']):
                completion_parameters = full_completion.drop(['run_prompt', 'experiment_id_fk', 'experiment_alias_fd', 'project_id_fk', 'project_alias_fd'])
                i1, t2 = st.columns(2)
                i1.image(full_completion['run_initial_img'], caption='Initial Image') if full_completion['run_initial_img'] else None
                t2.image(full_completion['run_target_imgs'], caption='Target Image') if full_completion['run_target_imgs'] else None
                st.write(completion_parameters.to_dict(), key=display_key)
                if st.button("Delete run", key=display_key):
                    with st.spinner("Deleting run..."):
                        to_delete = runs_df.index[runs_df['run_id'] == full_completion['run_id']].tolist()[0]+2
                        runs_wks.delete_rows(to_delete)
                        st.success("Run deleted")

        [runs_wks, runs_data, runs_dict, runs_df] = get_data_dict('runs', 'run_prompt', 'run_target_imgs')
        
        if selected_experiment:
            
            experiment_id = experiments_df.at[experiments_df.index[experiments_df['experiment_alias'] == selected_experiment].tolist()[0], 'experiment_id']        
        
            # [prompt_seed_wks, prompt_seed_data, prompt_seed_dict, prompt_seed_df] = get_data_dict('prompt_seeds', 'prompt_seed', 'prompt_seed_id')   
            # [prompts_wks, prompts_data, prompts_dict, prompts_df] = get_data_dict('prompts', 'prompt', 'prompt_seed_fd')  
            # [images_wks, images_data, images_dict, images_df] = get_data_dict('input_imgs', 'input_img_alias', 'input_img_url')
            # [runs_wks, runs_data, runs_dict, runs_df] = get_data_dict('runs', 'run_prompt', 'run_target_imgs')
            this_experiment_runs_df = runs_df[runs_df['experiment_id_fk'] == experiment_id]
            this_experiment_runs_df.reset_index(drop=True, inplace=True)    
            
            if st.checkbox("Preview experiment"):
                
                total_runs = len(this_experiment_runs_df)
                total_iterations = this_experiment_runs_df['run_max_iters'].sum()
                estimated_time = 0.5 * total_iterations * total_runs + total_runs * 10
                
                with st.expander("See statistics:"):
                    st.code(f"Total Runs: {total_runs}\nTotal Iterations: {total_iterations}\nEstimated Time: {estimated_time}")
                
                # if st.checkbox("See current experiment runs"):
                st.markdown("""
                \n\n
                ---
                ##### Experiment Runs""")

                # st.write(runs_data)
                # this_experiment_runs_df = runs_df[runs_df['experiment_id_fk'] == experiment_id]
                # this_experiment_runs_df.reset_index(drop=True, inplace=True)
                for i in range(len(this_experiment_runs_df)):
                    # st.write(runs_data)
                    # st.write(i)
                    # st.write(this_experiment_runs_df)
                    # st.write(this_experiment_runs_df.iloc[i])
                    display_run(this_experiment_runs_df.loc[i], i)
            
            # [prompts_wks, prompts_data, prompt_list, prompts_df] = get_prompt_list('prompts', 'prompt_seed', 'prompt')
            # [images_wks, images_data, image_list, images_df] = get_prompt_list('ref_imgs', 'img_alias', 'img_url')
            
            # selected_prompt = st.multiselect("Select from prompts in this project", list(prompts_dict.keys()))
                    
            # params.images_key = { image : images_df.at[images_df.index[images_df['img_url'] == image].tolist()[0], 'img_alias'] for image in image_list }
            
            # [prompt_seed_wks, prompt_seed_data, prompt_seed_list] = get_prompt_list('prompt_seeds')
            
            # selected_prompt_seed = st.selectbox("Select from prompts in this project", list(prompt_seed_list.keys()), key='prompt_seed')
            # [params.prompt_seed_id, params.prompt_id] = [prompt_seed_list[selected_prompt_seed] if selected_prompt_seed else None, None]
            # params.selected_prompt_seed = selected_prompt_seed
            
            # with st.expander("Create new prompt_seed"):
            #     topic = st.text_input("Topic")
            #     style = st.text_input("Style")
            #     prompt_seed = str({ 'topic': topic, 'style': style})
            #     click_create_new_prompt_seed = st.button("Save prompt_seed")
                
            #     if click_create_new_prompt_seed:
            #         last_row_prompt = len(prompt_seed_data)+1
            #         new_prompt_data = [last_row_prompt, str(project_id), selected_project, prompt_seed, topic, style, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")]
            #         prompt_seed_wks.insert_rows(last_row_prompt, number=1, values=new_prompt_data)    
            
            
            # ADJUST MODEL PARAMETERS

            def display_images(image_list):
                for image, caption in image_list.items():
                    st.image(image, caption=caption)
            
            # with st.expander("Create run(s)"):
                    
            #     exploration_type = st.radio("Generation Type", options=["Single Run", "Parameter Exploration"])
                
            #     if exploration_type == "Single Run":
            #         selected_prompt = [st.selectbox("Select from prompts in this project", list(prompts_dict.keys()))]
            #         # selected_prompt_seed = [prompts_dict[selected_prompt[0]] if selected_prompt else None]
            #         # params.temp = [st.slider("Temperature", 0.0, 1.0, 0.7)]
            #         # params.width = [col1.slider("Width", 50, 800, 400)]
            #         # params.height = [col2.slider("Height", 50, 800, 400)]
            #         col1, col2 = st.columns(2)
            #         params.width = [col1.selectbox("Width", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], index=8)]
            #         params.height = [col2.selectbox("Height", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], index=8)]
            #         params.learning_rate = [st.selectbox("Learning Rate", [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=4)]
            #         params.max_iters = [st.selectbox("Max Iterations", [0, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], index=4)]
            #         # params.learning_rate = [st.slider("Learning Rate", 0.0, 1.0, 0.2)]
            #         # params.max_iters = [st.slider("Max Iterations", 0, 1000, 200)]
            #         params.initial_img = [images_dict[st.selectbox("Select initial image", images_dict.keys())]]
            #         params.target_imgs = [images_dict[st.selectbox("Select target image", images_dict.keys())]]
            #     else:
            #         selected_prompt = st.multiselect("Select from prompts in this project", list(prompts_dict.keys()))
            #         # selected_prompt_seed = [prompts_dict[sprompt] if selected_prompt else None for sprompt in selected_prompt]
            #         # params.width = col1.slider("Width", 50, 800, 400)
            #         col1, col2 = st.columns(2)
            #         params.width = col1.multiselect("Width", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], default=[400])
            #         params.height = col2.multiselect("Height", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], default=[400])
            #         # params.height = col2.slider("Height", 50, 800, 400)
            #         # params.learning_rate = st.slider("Learning Rate", 0.0, 1.0, 0.2)
            #         params.learning_rate = st.multiselect("Learning Rate", [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], default=0.2)
            #         # params.max_iters = st.slider("Max Iterations", 0, 1000, 200)
            #         params.max_iters = st.multiselect("Max Iterations", [0, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], default=200)
            #         # st.write(temp_list)
            #         # params.temp = list(map(int, temp_list))
            #         # temperature_range = st.slider("Temperature", 0.0, 1.0, (0.5, 0.7))
            #         # params.temp = list(np.arange(temperature_range[0], temperature_range[1], 0.1))

            #     # params.initial_img = st.text_input("Initial Image url")
            #     # params.target_imgs = st.text_input("Target Image url") # single url for now
                
            #         params.initial_img = [images_dict[image] for image in st.multiselect("Select initial image", images_dict.keys())]  
            #         params.target_imgs = [images_dict[image] for image in st.multiselect("Select target image", images_dict.keys())]
                
                if st.checkbox("See reference images"):
                    display_initial_image_list = { image : 'Intial Image' for image in params.initial_img }
                    display_target_image_list = { image : 'Target Image' for image in params.target_imgs }
                    col3, col4 = st.columns(2)
                    with col3:
                        display_images(display_initial_image_list)
                    with col4:
                        display_images(display_target_image_list)
                    
                    # col3.image(params.initial_img, caption="Initial Image") if params.initial_img != '' else col3.write("No initial image selected.")
                    # col4.image(params.target_imgs, caption="Target Image") if params.target_imgs != '' else col4.write("No target image selected.")
                
                # click_create_run = st.button("Create Run") if exploration_type == "Single Run" else st.button("Create runs")
                
                # params.initial_img = [""] if params.initial_img == [] else params.initial_img
                # params.target_imgs = [""] if params.target_imgs == [] else params.target_imgs
                
                # click_deploy_experiment = st.button("Deploy experiment")
                
                # if click_deploy_experiment:
                #     # st.write(selected_prompt, params.width, params.height, params.learning_rate, params.max_iters, params.initial_img, params.target_imgs)
                #     for prompt, width, height, learning_rate, max_iters, initial_img, target_imgs in itertools.product(selected_prompt, params.width, params.height, params.learning_rate, params.max_iters, params.initial_img, params.target_imgs):
                        
                #         [runs_wks, runs_data, runs_dict, runs_df] = get_data_dict('runs', 'run_prompt', 'run_target_imgs')
                #         last_row_run = len(runs_data)+1
                #         # st.write(runs_data)
                #         # st.write(type(prompt), type(width), type(height), type(learning_rate), type(max_iters), type(initial_img), type(target_imgs))
                #         # st.write(params.initial_img, params.target_imgs)
                #         # DO NOT EDIT UNDERLYING DB, ONLY VIEWS - if projects get deleted, id will sometimes skip a value
                #         new_run_data = [str(last_row_run), prompt, str(width), str(height), initial_img, target_imgs, str(learning_rate), str(max_iters), datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), None, str(experiment_id), selected_experiment, str(project_id), selected_project, str(prompts_dict[prompt])]
                #         runs_wks.insert_rows(last_row_run, number=1, values=new_run_data)
                        
            
            # RUN EXPLORATION
            
            if st.button("Deploy experiment"):
                
                # with st.expander("See Parameters"):
                #     st.write(params.__dict__)
                
                with st.spinner('Deploying experiment...'):
                    
                    with st.spinner('Spinning up Sagemaker notebook...'):
        
                        
                        session = boto3.Session(
                            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                        )
                        # sagemaker = session.resource('sagemaker')
                        
                        lambda_client = session.client('lambda', region_name='us-east-2')
                        lambda_payload = '{"ckpt1": "checkpoint validated"}'
                        lambda_client.invoke(FunctionName='arn:aws:lambda:us-east-2:841009426293:function:start_sagemaker', 
                                            InvocationType='RequestResponse',
                                            Payload=lambda_payload)
                        
                        st.success('Sagemaker notebook alive.')
                    
                    # st.code("Spinning up Sagemaker notebook...")
                    
                    with st.spinner("Sagemaker code running..."):
                    
                        st.success('Sagemaker code finished.')
                    # st.code("Sagemaker notebook running...")
                    
                    # num_runs = len(runs_data)
                    # experiments_wks.update_value('C%s'%str(experiment_id+1), num_runs)
                    
                    # art.commit(params, sheet) 
                    st.success('Experiment complete!')
                    # st.balloons()
                    
                # with st.spinner("Running experiment..."):
                #     # import time

                #     my_bar = st.progress(0)

                #     for percent_complete in range(100):
                #         time.sleep(0.1)
                #         my_bar.progress(percent_complete + 1)
                
                
        st.markdown("""
        \n\n
        ---
        #### Check Experiment Status""")
        
        
        [experiments_s_wks, experiments_s_data, experiments_s_dict, experiments_s_df] = get_data_dict('experiments', 'experiment_id', 'experiment_alias')  
        selected_experiment_s = st.selectbox("Select experiment to check status", list(experiments_s_dict.values()))  
            
        if st.button("Check status"):
            
            experiment_id_s = experiments_s_df.at[experiments_s_df.index[experiments_s_df['experiment_alias'] == selected_experiment_s].tolist()[0], 'experiment_id']        
        
            this_experiment_s_runs_df = runs_df[runs_df['experiment_id_fk'] == experiment_id_s]
            this_experiment_s_runs_df.reset_index(drop=True, inplace=True)  
            
            # change this
            notebook_status = "Running"
            total_runs = len(this_experiment_s_runs_df)
            total_iterations = this_experiment_s_runs_df['run_max_iters'].sum()
            estimated_time = 0.5 * total_iterations * total_runs + total_runs * 10
            
            st.code(f"""Notebook instance status {notebook_status}\nRuns completed: X/{total_runs} total runs\nTime elapsed: X/{estimated_time} estimated total time\n---\nTotal Runs: {total_runs}\nTotal Iterations: {total_iterations}\nEstimated Time: {estimated_time}\n""")
            
            def display_generation(full_completion, display_key):
                with st.expander(full_completion['prompt']):
                    completion_parameters = full_completion.drop('prompt')
                    st.write(completion_parameters.to_dict(), key=display_key)

            
            fs = s3fs.S3FileSystem(anon=False)

            # Retrieve file contents.
            # Uses st.cache to only rerun when the query changes or after 10 min.
            # @st.cache(ttl=600)
            # def read_file(filename):
            #     with fs.open(filename) as f:
            #         return f.read()
            
            
            # AWS_ACCESS_KEY_ID = "AKIA4HUARN526PCVJ44X"
            # AWS_SECRET_ACCESS_KEY = "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"

            session = boto3.Session(
                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            )
            s3 = session.resource('s3')
            bucket = s3.Bucket('aiaf-output')
            # object = bucket.Object('tiles/10/S/DG/2015/12/7/0/B01.jp2')

            # for my_bucket_object in my_bucket.objects.all():
            #     print(my_bucket_object.key)
            # @st.cache
            def read_image(image):
                object = bucket.Object(image.key)
                img_data = object.get().get('Body').read()
                return img_data
            
            def display_run_results(run_id, interval):
                
                # folder_prefix = "E" + str(experiment_id) + "R" + str(run_id)
                count = 0
                for image in bucket.objects.all():
                    if image.key.startswith(folder_name):
                        if count % interval == 0:
                            # st.write(count, interval)
                        # object = bucket.Object(image.key)
                        # img_data = object.get().get('Body').read()
                        
                            img_data = read_image(image)
                            # print(img_data)
                            st.image(img_data, caption=image.key)
                        count += 1
                
            
            for i in range(len(this_experiment_s_runs_df)):
                # st.write(runs_data)
                # st.write(i)
                # st.write(this_experiment_runs_df)
                # st.write(this_experiment_runs_df.iloc[i])
                run_alias = sub("\W", "", this_experiment_s_runs_df.iloc[i]['run_alias'].lower().strip().replace(" ","_"))
                folder_name = "E" + str(experiment_id_s) + "R" + str(this_experiment_s_runs_df.iloc[i]['run_id']) + "_" + run_alias
                with st.expander(folder_name):
                    img_side, param_side = st.columns(2)
                    with img_side:
                        display_run_results(this_experiment_s_runs_df.iloc[i]['run_id'], 25)
                    # if st.checkbox("Show parameters", key=i):
                    with param_side:
                        st.markdown("##### Parameters")
                        # if st.checkbox("Show parameters", key=i):
                            # display_run(this_experiment_runs_df.loc[i], i)
                        completion_parameters = this_experiment_s_runs_df.loc[i].drop(['run_prompt', 'experiment_id_fk', 'experiment_alias_fd', 'project_id_fk', 'project_alias_fd'])
                        # i1, t2 = st.columns(2)
                        st.image(this_experiment_s_runs_df.loc[i]['run_initial_img'], caption='Initial Image') if this_experiment_s_runs_df.loc[i]['run_initial_img'] else None
                        st.image(this_experiment_s_runs_df.loc[i]['run_target_imgs'], caption='Target Image') if this_experiment_s_runs_df.loc[i]['run_target_imgs'] else None
                        st.write(completion_parameters.to_dict(), key=i)
            
            # for i in range(len(parsed_response)):
            #     display_completion(parsed_response.loc[i], i)
        
        # if st.button("Deploy experiment"):
            
        #     st.code("Spinning up Sagemaker instance, auto-running notebook on lifecycle config, outputting to S3 bucket")
                
        #     # OPTION TO SEE EXPLORATION PARAMETERS
        #     with st.expander("See Parameters"):
        #         st.write(params.__dict__)

        #     with st.spinner('Generating..'):

        #         full_response, parsed_response = art.generate(params, sheet)
                
        #         # OPTIONS TO DISPLAY DETAILED RESPONSES
        #         # with st.expander("See Full API Response"):
        #         #     st.code(full_response)
        #         # with st.expander("See Data Table Response"):
        #         #     st.write(parsed_response)
                    
        #         def display_completion(full_completion, display_key):
        #             with st.expander(full_completion['prompt']):
        #                 completion_parameters = full_completion.drop('prompt')
        #                 st.write(completion_parameters.to_dict(), key=display_key)
                
        #         for i in range(len(parsed_response)):
        #             display_completion(parsed_response.loc[i], i)

        #         st.success('Done!')

# app()    
            
            
            
            