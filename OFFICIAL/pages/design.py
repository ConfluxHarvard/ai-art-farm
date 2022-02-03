import streamlit as st
import numpy as np
import pandas as pd
import pygsheets
from datetime import datetime
import openai
import numpy as np
import itertools

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
    st.title("SETUP EXPERIMENT")
    
        
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
        
        [prompts_wks, prompts_data, prompts_dict, prompts_df] = get_data_dict('prompts', 'prompt', 'prompt_id') 
        [prompt_topics_wks, prompt_topics_data, prompt_topics_dict, prompt_topics_df] = get_data_dict('prompts', 'prompt', 'prompt_topic_fd') 
        [prompt_styles_wks, prompt_styles_data, prompt_styles_dict, prompt_styles_df] = get_data_dict('prompts', 'prompt', 'prompt_style_fd') 
        [images_wks, images_data, images_dict, images_df] = get_data_dict('input_imgs', 'input_img_alias', 'input_img_url')

        with st.expander("Add new prompt"): 
                
            new_prompt = st.text_input("New prompt")
            click_create_new_prompt = st.button("Save prompt")
                
            if click_create_new_prompt:
                last_row_prompt = len(prompts_data)+1
                new_prompt_data = [last_row_prompt, new_prompt, None, None, None, None, None, None, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(project_id), selected_project, None, None]
                prompts_wks.insert_rows(last_row_prompt, number=1, values=new_prompt_data)    
                
        with st.expander("Add new reference image"):
            
            # Add upload to s3 bucket?
            
            new_image_alias = st.text_input("New image alias")
            new_image_url = st.text_input("New image url")
            st.image(new_image_url, caption=new_image_alias if new_image_alias else "Untitled") if new_image_url else None
            click_create_new_image = st.button("Save image")
            
            if click_create_new_image:
                last_row_image = len(images_data)+1
                new_image_data = [last_row_image, new_image_alias, new_image_url, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(project_id), selected_project]
                images_wks.insert_rows(last_row_image, number=1, values=new_image_data)  
        
        
        st.markdown("""
            \n\n
            ---
            ##### Design your Experiment""")
        
        [experiments_wks, experiments_data, experiments_dict, experiments_df] = get_data_dict('experiments', 'experiment_id', 'experiment_alias')  
        selected_experiment = st.sidebar.selectbox("Select your experiment", list(experiments_dict.values()))  
        
        with st.sidebar.expander("Create new experiment"):
            new_experiment_name = st.text_input("Enter experiment name")
            click_create_new_experiment = st.button("Save experiment")
            
        if click_create_new_experiment:
            last_row_experiment = len(experiments_data)+1
            # DO NOT EDIT UNDERLYING DB, ONLY VIEWS - if projects get deleted, id will sometimes skip a value
            new_experiment_data = [str(last_row_experiment), new_experiment_name, None, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), None, None, "Incomplete", str(project_id), selected_project]
            experiments_wks.insert_rows(last_row_experiment, number=1, values=new_experiment_data)

        if selected_experiment:
            
            experiment_id = experiments_df.at[experiments_df.index[experiments_df['experiment_alias'] == selected_experiment].tolist()[0], 'experiment_id']        
        
            # [prompt_seed_wks, prompt_seed_data, prompt_seed_dict, prompt_seed_df] = get_data_dict('prompt_seeds', 'prompt_seed', 'prompt_seed_id')   
            # [prompts_wks, prompts_data, prompts_dict, prompts_df] = get_data_dict('prompts', 'prompt', 'prompt_seed_fd')  
            # [images_wks, images_data, images_dict, images_df] = get_data_dict('input_imgs', 'input_img_alias', 'input_img_url')
            [runs_wks, runs_data, runs_dict, runs_df] = get_data_dict('runs', 'run_prompt', 'run_target_imgs')
                
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
            
            with st.expander("Add run(s)"):
                
                run_alias = st.text_input("Run alias")
                    
                exploration_type = st.radio("Generation Type", options=["Single Run", "Parameter Exploration"])
                
                if exploration_type == "Single Run":
                    selected_prompt = [st.selectbox("Select from prompts in this project", list(prompts_dict.keys()))]
                    selected_prompt_topic = [prompt_topics_dict[selected_prompt[0]] if selected_prompt else None]
                    selected_prompt_style = [prompt_styles_dict[selected_prompt[0]] if selected_prompt else None]
                    # selected_prompt_seed = [prompts_dict[selected_prompt[0]] if selected_prompt else None]
                    # params.temp = [st.slider("Temperature", 0.0, 1.0, 0.7)]
                    # params.width = [col1.slider("Width", 50, 800, 400)]
                    # params.height = [col2.slider("Height", 50, 800, 400)]
                    col1, col2 = st.columns(2)
                    params.width = [col1.selectbox("Width", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], index=8)]
                    params.height = [col2.selectbox("Height", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], index=8)]
                    params.learning_rate = [st.selectbox("Learning Rate", [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=4)]
                    params.max_iters = [st.selectbox("Max Iterations", [0, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], index=4)]
                    # params.learning_rate = [st.slider("Learning Rate", 0.0, 1.0, 0.2)]
                    # params.max_iters = [st.slider("Max Iterations", 0, 1000, 200)]
                    params.initial_img = [st.selectbox("Select initial image", [''] + list(images_dict.keys()))]
                    params.target_imgs = [st.selectbox("Select target image", [''] + list(images_dict.keys()))]
                    
                    # params.initial_img = [images_dict[params.initial_img]] if params.initial_img != '' else ''
                    # params.target_imgs = [images_dict[params.initial_img]] if params.initial_img != '' else ''
                
                else:
                    selected_prompt = st.multiselect("Select from prompts in this project", list(prompts_dict.keys()))
                    selected_prompt_topic = [prompt_topics_dict[sprompt] if selected_prompt else None for sprompt in selected_prompt]
                    selected_prompt_style = [prompt_styles_dict[sprompt] if selected_prompt else None for sprompt in selected_prompt]
                    # selected_prompt_seed = [prompts_dict[sprompt] if selected_prompt else None for sprompt in selected_prompt]
                    # params.width = col1.slider("Width", 50, 800, 400)
                    col1, col2 = st.columns(2)
                    params.width = col1.multiselect("Width", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], default=[400])
                    params.height = col2.multiselect("Height", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], default=[400])
                    # params.height = col2.slider("Height", 50, 800, 400)
                    # params.learning_rate = st.slider("Learning Rate", 0.0, 1.0, 0.2)
                    params.learning_rate = st.multiselect("Learning Rate", [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], default=0.2)
                    # params.max_iters = st.slider("Max Iterations", 0, 1000, 200)
                    params.max_iters = st.multiselect("Max Iterations", [0, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], default=200)
                    # st.write(temp_list)
                    # params.temp = list(map(int, temp_list))
                    # temperature_range = st.slider("Temperature", 0.0, 1.0, (0.5, 0.7))
                    # params.temp = list(np.arange(temperature_range[0], temperature_range[1], 0.1))

                # params.initial_img = st.text_input("Initial Image url")
                # params.target_imgs = st.text_input("Target Image url") # single url for now
                    params.initial_img = st.multiselect("Select initial image", [''] + list(images_dict.keys())) 
                    params.target_imgs = st.multiselect("Select target image", [''] + list(images_dict.keys()))

                    # params.initial_img = [images_dict[image] for image in st.multiselect("Select initial image", [''] + list(images_dict.keys()))]  
                    # params.target_imgs = [images_dict[image] for image in st.multiselect("Select target image", [''] + list(images_dict.keys()))]

                params.initial_img = [images_dict[image] if image != '' else '' for image in params.initial_img]
                params.target_imgs = [images_dict[image] if image != '' else '' for image in params.target_imgs]
                    # [images_dict[image] for image in st.multiselect("Select initial image", [''] + [images_dict.keys()])]  
                    # params.target_imgs = [images_dict[image] for image in st.multiselect("Select target image", [''] + [images_dict.keys()])]
                
                
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
                
                click_create_run = st.button("Commit run to experiment") if exploration_type == "Single Run" else st.button("Commit runs to experiment")
                
                params.initial_img = [""] if params.initial_img == [] else params.initial_img
                params.target_imgs = [""] if params.target_imgs == [] else params.target_imgs
                
                if click_create_run:
                    
                    with st.spinner("Commiting run to database...") if exploration_type == "Single Run" else st.spinner("Commiting runs to database..."):
                    # st.write(selected_prompt, params.width, params.height, params.learning_rate, params.max_iters, params.initial_img, params.target_imgs)
                        # runs_df_filtered = runs_df[runs_df.run_id != ""]
                        # st.write(runs_df_filtered)
                        
                        i = 1
                        for prompt, width, height, learning_rate, max_iters, initial_img, target_imgs, topic, style in itertools.product(selected_prompt, params.width, params.height, params.learning_rate, params.max_iters, params.initial_img, params.target_imgs, selected_prompt_topic, selected_prompt_style):
                            # run_alias = run_alias + "*"
                            [runs_wks, runs_data, runs_dict, runs_df] = get_data_dict('runs', 'run_prompt', 'run_target_imgs')
                            # runs_df_filtered = runs_df[runs_df.output_bucket_alias != "None"]
                            # st.write(i, runs_df_filtered)
                            # st.write("DF", runs_df)
                            try:
                                new_index = runs_df['run_id'].max() + 1
                                # st.write("NEW Index", new_index)
                                # st.write("RUN ID COL", runs_df['run_id'])
                                # st.write("MAX", runs_df['run_id'].max())
                            except:
                                new_index = 1
                                
                            # st.write(runs_df)
                            # runs_df_filtered = runs_df[runs_df.run_id != ""]
                            # st.write(runs_df_filtered)
                            last_row_run = len(runs_df) + 1
                            
                            run_alias_i = run_alias + str(i)
                            i += 1
                            # st.write(runs_data)
                            # st.write(type(prompt), type(width), type(height), type(learning_rate), type(max_iters), type(initial_img), type(target_imgs))
                            # st.write(params.initial_img, params.target_imgs)
                            # DO NOT EDIT UNDERLYING DB, ONLY VIEWS - if projects get deleted, id will sometimes skip a value
                            output_bucket_simp = "E" + str(runs_df.iloc[i]['experiment_id_fk']) + "R" + str(int(runs_df.iloc[i]['run_id'])) + "_" + run_alias_i
                            output_bucket = output_bucket_simp.replace(" (", "_").replace(")", "")
                            new_run_data = [str(new_index), run_alias_i, prompt, str(width), str(height), initial_img, target_imgs, str(learning_rate), str(max_iters), datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), output_bucket, str(experiment_id), selected_experiment, str(project_id), selected_project, topic, style]
                            # st.write(new_run_data)
                            runs_wks.insert_rows(last_row_run, number=1, values=new_run_data)
                            # run_alias = run_alias + "*"
                            # run_alias = run_alias + str(i)
                            # i += 1
                        st.success("Run committed!") if exploration_type == "Single Run" else st.success("Runs committed!")
                        
            
            
            try:
                this_experiment_runs_df = runs_df[runs_df['experiment_id_fk'] == experiment_id]
                this_experiment_runs_df.reset_index(drop=True, inplace=True)
                total_runs = len(this_experiment_runs_df)
                total_iterations = this_experiment_runs_df['run_max_iters'].sum()
                estimated_time = 0.5 * total_iterations * total_runs + total_runs * 10
            except:
                this_experiment_runs_df = pd.DataFrame()
                total_runs = 0
                total_iterations = 0
                estimated_time = 0
            
            
            # if st.checkbox("Preview experiment"):
            
            st.markdown("""
                \n\n
                ---
                ##### Experiment Preview""")
            
            if st.button("Commit experiment"):
            
                # with st.expander("See Parameters"):
                #     st.write(params.__dict__)
                
                with st.spinner('Committing experiment..'):
                    
                    # num_runs = len(runs_data)
                    num_runs = len(runs_df.loc[runs_df['experiment_id_fk'] == experiment_id])
                    experiments_wks.update_value('C%s'%str(experiment_id+1), num_runs)
                    experiments_wks.update_value('G%s'%str(experiment_id+1), "Not Started")
                    
                    # art.commit(params, sheet) 
                    st.success('Experiment committed!')
            
            st.code(f"Total Runs: {total_runs}\nTotal Iterations: {total_iterations}\nEstimated Time: {estimated_time}")
            
            # if st.checkbox("See current experiment runs"):
                
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
                
            for i in range(len(this_experiment_runs_df)):
                display_run(this_experiment_runs_df.loc[i], i)
        
        
        # RUN EXPLORATION
        
        # if st.button("Commit experiment"):
            
        #     # with st.expander("See Parameters"):
        #     #     st.write(params.__dict__)
            
        #     with st.spinner('Committing experiment..'):
                
        #         # num_runs = len(runs_data)
        #         num_runs = len(runs_df.loc[runs_df['experiment_id_fk'] == experiment_id])
        #         experiments_wks.update_value('C%s'%str(experiment_id+1), num_runs)
        #         experiments_wks.update_value('G%s'%str(experiment_id+1), "Not Started")
                
        #         # art.commit(params, sheet) 
        #         st.success('Experiment committed!')
                
            
            # def display_completion(full_completion, display_key):
            #     with st.expander(full_completion['prompt']):
            #         completion_parameters = full_completion.drop('prompt')
            #         st.write(completion_parameters.to_dict(), key=display_key)
                
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