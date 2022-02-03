import streamlit as st
import numpy as np
import pandas as pd
import pygsheets
from datetime import datetime
import openai
import numpy as np
from re import sub

import s3fs
import boto3

class Params:
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def app():
    
    # SET APP PROPERTIES
    
    st.title("DATABASE")
        
    # SELECT PROJECT

    gc = pygsheets.authorize(service_file='pages/ai-art-farm-2c65230a4396.json')
    sheet = gc.open('AIAF_Experimental_Database')
    projects_wks = sheet.worksheet('title', 'projects')
    projects_data = projects_wks.get_all_records()
    projects_df = pd.DataFrame.from_dict(projects_data)
    project_list = { value['project_id']: value['project_alias'] for key, value in projects_df.iterrows() }
    selected_project = st.sidebar.selectbox("Select your project", ["All"] + list(project_list.values()))  
    selected_data_source = st.sidebar.radio("Data source", ["Inputs", "Outputs"])

    params = Params()
    # params_foreign = Params()
    
    if 'search_state' not in st.session_state:
        st.session_state.search_state = 0
    if 'current_tags' not in st.session_state:
        st.session_state.current_tags = []

    if selected_project:
        
        project_id = projects_df.at[projects_df.index[projects_df['project_alias'] == selected_project].tolist()[0], 'project_id'] if selected_project != "All" else None      
        
        def get_data_dict(sheet_name, key, value):
            data_wks = sheet.worksheet('title', sheet_name)
            data_all = data_wks.get_all_records()
            data_df = pd.DataFrame.from_dict(data_all)
            try:
                this_project_data = data_df[data_df['project_id_fk'] == project_id] if selected_project != "All" else data_df
            except:
                this_project_data = pd.DataFrame()
            data_dict = { v[key] : v[value] for k, v in this_project_data.iterrows() }
            return data_wks, data_all, data_dict, data_df
        
        
        if selected_data_source == "Inputs":
            
            search, view = st.columns((2, 1))
        
            search_query = search.text_input("Keyword search")

            search_refimg = view.checkbox("Reference Image", value=True)
            
            search_prompt = view.checkbox("Prompt", value=True)
            
            with st.expander("Filter Search"):
                
                [prompts_wks, prompts_data, prompts_dict, prompts_df] = get_data_dict('prompts_full', 'prompt', 'prompt_id')  
                [images_wks, images_data, images_dict, images_df] = get_data_dict('input_imgs_full', 'input_img_alias', 'input_img_url')
                
                scoreg, scoren, scorea, scorem = st.columns(4)
                params.score_good_fd = [float(i) for i in scoreg.multiselect("Good", [-1, 0, 1])]
                params.score_novel_fd = [float(i) for i in scoren.multiselect("Novel", [-1, 0, 1])]
                params.score_aesthetic_fd = [float(i) for i in scorea.multiselect("Aesthetic", [-1, 0, 1])]
                params.score_meaning_fd = [float(i) for i in scorem.multiselect("Meaning", [-1, 0, 1])]

                [tags_wks, tags_data, tags_dict, tags_df] = get_data_dict('tags', 'tag_alias', 'tag_category')
                tags_list = list(tags_dict.keys())
                
                selected_tags = st.multiselect("General Tags", tags_list)
                
                if search_prompt:
                    
                    st.markdown("##### Prompt Parameters")
                    
                    def get_model_info():
                        main_model_sheet = gc.open('Prompt_Model_Info_AIAF')
                        main_models_wks = main_model_sheet.worksheet('title', 'info')
                        main_models_data = main_models_wks.get_all_records()
                        main_models_df = pd.DataFrame.from_dict(main_models_data)
                        main_models_list = main_models_df['model_alias'].tolist()
                        models = main_models_list

                        return main_models_df, models
                    
                    main_models_df, models = get_model_info()
                    
                    [seed_properties_wks, seed_properties_data, seed_properties_dict, seed_properties_df] = get_data_dict('runs_full', 'prompt_topic_fd', 'prompt_style_fd')
                    
                    topics_list = list(seed_properties_dict.keys())
                    newt = topics_list.remove("None")
                    styles_list = list(seed_properties_dict.values())
                    news = styles_list.remove("None")


                    tcol, scol = st.columns(2)
                    params.prompt_topic_fd = tcol.multiselect("Topic", topics_list)
                    params.prompt_style_fd = scol.multiselect("Style", styles_list)
                    
                    params.prompt_temperature = [float(i) for i in st.multiselect("Temperature", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]
                    params.prompt_model_alias = st.multiselect("Model", models)
                
                # RUN SEARCH
                
            search = st.button("Search")
                
            if search:
                st.session_state.search_state = 1
            if st.session_state.search_state == 1:
                
                with st.spinner('Searching..'):
                    
                    def filter_df(df, filter_by_main):

                        df.iloc[:, 0].replace('', np.nan, inplace=True)
                        df = df[df.iloc[:, 0].notna()]
                        for key, value in filter_by_main.items():
                            if value != []:
                                df = df[df[key].isin(value)]
                            else:
                                df = df
                                
                        return df
                    
                    def filter_by_tag(df, selected_tags):
                        
                        if selected_tags != []:
                            [tagpairs_wks, tagpairs_data, tagpairs_dict, tagpairs_df] = get_data_dict('tagpairs', 'tag_alias_fd', 'tagpair_item_id_fk')
                            tagpairs_df = tagpairs_df[tagpairs_df['tagpair_item_category'] == 'prompts']
                            tagpairs_df = tagpairs_df[tagpairs_df['tag_alias_fd'].isin(selected_tags)]
                            df = df[df['prompt_id'].isin(tagpairs_df['tagpair_item_id_fk'].tolist())]
                            
                        else:
                            df = df
                            tagpairs_df = pd.DataFrame()
                            
                        return df, tagpairs_df     
                        
                    def filter_by_query(df, query):
                        if query != '':
                            filtered_df = df[df.apply(lambda r: r.str.contains(search_query, case=False).any(), axis=1)] 
                        else: filtered_df = df
                            
                        return filtered_df
                
                
                    if search_prompt:
                    
                        filtered_prompts_df = filter_df(prompts_df, params.__dict__)
                        filtered_prompts_df.reset_index(drop=True, inplace=True) 
                        
                        filtered_prompts_by_tag_df, tagpairs_df_filtered = filter_by_tag(filtered_prompts_df, selected_tags)
                        filtered_prompts_by_tag_df.reset_index(drop=True, inplace=True)  
                        
                        filtered_prompts_by_query_and_tag = filter_by_query(filtered_prompts_by_tag_df, search_query)
                        
                    if search_refimg:
                        
                        filtered_refimgs_df = filter_df(images_df, params.__dict__)
                        filtered_refimgs_df.reset_index(drop=True, inplace=True) 
                        
                        filtered_refimgs_by_tag_df, tagpairs_df_filtered = filter_by_tag(filtered_refimgs_df, selected_tags)
                        filtered_refimgs_by_tag_df.reset_index(drop=True, inplace=True)  
                        
                        filtered_refimgs_by_query_and_tag = filter_by_query(filtered_refimgs_by_tag_df, search_query)
                    
                    
                    def display_search_result(full_completion, datatype, display_key):
                        with st.expander(full_completion[datatype]):
                            
                            in_params = Params()
                            scoreg, scoren, scorea, scorem = st.columns(4)
                            score_dict = {-1: 0, 0: 1, 1: 2, 'None': 1}
                            
                            current_score_good = full_completion['score_good_fd']
                            current_score_novel = full_completion['score_novel_fd']
                            current_score_aesthetic = full_completion['score_aesthetic_fd']
                            current_score_meaning = full_completion['score_meaning_fd']
                            
                            in_params.score_good = scoreg.selectbox("Good", [-1, 0, 1], index=score_dict[current_score_good], key=str(i) + "in_params" + datatype)
                            in_params.score_novel = scoren.selectbox("Novel", [-1, 0, 1], index=score_dict[current_score_novel], key=str(i) + "in_params" + datatype)
                            in_params.score_aesthetic = scorea.selectbox("Aesthetic", [-1, 0, 1], index=score_dict[current_score_aesthetic], key=str(i) + "in_params" + datatype)
                            in_params.score_meaning = scorem.selectbox("Meaning", [-1, 0, 1], index=score_dict[current_score_meaning], key=str(i) + "in_params" + datatype)

                            [tagpairs_wks, tagpairs_data, tagpairs_dict, tagpairs_df] = get_data_dict('tagpairs', 'tag_alias_fd', 'tagpair_item_id_fk')
                            
                            try:
                                filtered_tagpairs_df = tagpairs_df[tagpairs_df['tagpair_item_category'] == 'output'] 
                                filteredd_tagpairs_df = filtered_tagpairs_df.loc[filtered_tagpairs_df['tagpair_item_id_fk'] == run_id]
                                current_tags = filteredd_tagpairs_df['tag_alias_fd'].tolist()
                            except:
                                current_tags = []
                            tags, save = st.columns(2)
                            st.session_state.current_tags = tags.multiselect("Tags", options=st.session_state.current_tags if st.session_state.current_tags != [] else tags_list, default=current_tags, key=str(i) + "in_params" + datatype)
                            in_params.tags = st.session_state.current_tags
                            
                            click_save_in_params = save.button("Update", key=str(i) + datatype)
                            
                            if click_save_in_params:
                                
                                with st.spinner("Updating prompt info..."):
                                
                                    [inscores_wks, inscores_data, inscores_dict, inscores_df] = get_data_dict('input_scores', 'input_score_id', 'input_item_category')
                                    if run_id in outscores_df['output_score_run_id_fk'].astype(int).tolist():
                                        row = outscores_df.loc[outscores_df['output_score_run_id_fk'] == run_id].index[0]
                                        inscores_wks.update_values(crange=('B{0}:E{0}'.format(row + 2)), values=[[in_params.score_good, in_params.score_novel, in_params.score_aesthetic, in_params.score_meaning]])
                                    else:
                                        last_row_score = len(outscores_data)+1
                                        new_score_data = [last_row_score, in_params.score_good, in_params.score_novel, in_params.score_aesthetic, in_params.score_meaning, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), int(full_completion['run_id'])]
                                        outscores_wks.insert_rows(last_row_score, number=1, values=new_score_data)
                                    
                                    tags_list_old = list(tagpairs_dict.keys())
                                    unchanged_tags = set(current_tags).intersection(set(in_params.tags))
                                    tags_list_edit = list(set(in_params.tags).union(set(tags_list_old)).difference(unchanged_tags))
                                    st.write("TAGS LIST EDIT", tags_list_edit)
                                    for tag in tags_list_edit:
                                        if tag in tags_list_old:
                                            [tagpairs_wks, tagpairs_data, tagpairs_dict, tagpairs_df] = get_data_dict('tagpairs', 'tag_alias_fd', 'tagpair_item_id_fk')
                                            delete_row = tagpairs_df.index[tagpairs_df['tag_alias_fd'] == tag].tolist()[0]
                                            st.write("delete row", delete_row)
                                            tagpairs_wks.delete_rows(delete_row + 2)
                                        else:
                                            last_row_tag = len(tagpairs_data)+1
                                            tag_id = tags_df.loc[tags_df['tag_alias'] == tag]['tag_id'].values[0]
                                            tag_category = tags_df.loc[tags_df['tag_alias'] == tag]['tag_category'].values[0]
                                            new_tag_data = [str(last_row_tag), 'output', datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(run_id), str(tag_id), tag, tag_category]
                                            tagpairs_wks.insert_rows(last_row_tag, number=1, values=new_tag_data)
                                    
                                    st.success("Updated run info")
                            
                            st.image(full_completion['input_img_url'], width=600) if datatype == 'input_img_alias' else None
                            completion_parameters = full_completion.drop(datatype)
                            st.write(completion_parameters.to_dict(), key=display_key)
                            
                    if search_prompt:
                        for i in range(len(filtered_prompts_by_query_and_tag)):
                            display_search_result(filtered_prompts_by_query_and_tag.iloc[i], 'prompt', i)
                    if search_refimg:
                        for i in range(len(filtered_refimgs_by_query_and_tag)):
                            display_search_result(filtered_refimgs_by_query_and_tag.iloc[i], 'input_img_alias', i)
                    
            
        else:
        
            search, view = st.columns(2)
            
            search_query = search.text_input("Keyword search")
            view_mode = view.selectbox("View mode", ['Simple', 'Representative', 'All'])
            
            with st.expander("Filter Search"):
                
                [experiments_wks, experiments_data, experiments_dict, experiments_df] = get_data_dict('experiments', 'experiment_id', 'experiment_alias')  
                params.experiment_alias_fd = st.multiselect("Select your experiment", list(experiments_dict.values()))  
                
                [prompts_wks, prompts_data, prompts_dict, prompts_df] = get_data_dict('prompts', 'prompt', 'prompt_id')  
                [images_wks, images_data, images_dict, images_df] = get_data_dict('input_imgs', 'input_img_alias', 'input_img_url')

                [runs_wks, runs_data, runs_dict, runs_df] = get_data_dict('runs_full', 'run_prompt', 'run_target_imgs')
                [seed_properties_wks, seed_properties_data, seed_properties_dict, seed_properties_df] = get_data_dict('runs_full', 'prompt_topic_fd', 'prompt_style_fd')
                    
                
                scoreg, scoren, scorea, scorem = st.columns(4)
                params.score_good_fd = scoreg.multiselect("Good", [-1, 0, 1])
                params.score_novel_fd = scoren.multiselect("Novel", [-1, 0, 1])
                params.score_aesthetic_fd = scorea.multiselect("Aesthetic", [-1, 0, 1])
                params.score_meaning_fd = scorem.multiselect("Meaning", [-1, 0, 1])
                
                params.run_prompt = st.multiselect("Prompt", list(prompts_dict.keys()), key='prompts')

                topics_list = list(seed_properties_dict.keys())
                newt = topics_list.remove("None")
                styles_list = list(seed_properties_dict.values())
                news = styles_list.remove("None")

                params.prompt_topic_fd = st.multiselect("Topic", topics_list)
                params.prompt_style_fd = st.multiselect("Style", styles_list)

                [tags_wks, tags_data, tags_dict, tags_df] = get_data_dict('tags', 'tag_alias', 'tag_category')
                tags_list = list(tags_dict.keys())

                if st.checkbox("More parameters"):
                    
                    st.markdown("### GAN Parameters")
                    
                    w, h = st.columns(2)
                    params.run_width = w.multiselect("Width", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
                    params.run_height = h.multiselect("Height", [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
                    params.run_learning_rate = st.multiselect("Learning Rate", [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    params.run_max_iters = st.multiselect("Max Iterations", [0, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
                    
                    initial_img_alias = st.multiselect("Initial image", list(images_dict.keys())) 
                    target_imgs_alias = st.multiselect("Target image", list(images_dict.keys()))
                    
                    params.run_initial_img = [images_dict[image] if image != '' else '' for image in initial_img_alias]
                    params.run_target_imgs = [images_dict[image] if image != '' else '' for image in target_imgs_alias]
                    
                    st.markdown("### Prompt Parameters")
                    
                    def get_model_info():
                        main_model_sheet = gc.open('Prompt_Model_Info_AIAF')
                        main_models_wks = main_model_sheet.worksheet('title', 'info')
                        main_models_data = main_models_wks.get_all_records()
                        main_models_df = pd.DataFrame.from_dict(main_models_data)
                        main_models_list = main_models_df['model_alias'].tolist()
                        models = main_models_list

                        return main_models_df, models
                    
                    main_models_df, models = get_model_info()
                    
                    params.prompt_temperature_fd = st.multiselect("Temperature", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    params.prompt_model_alias_fd = st.multiselect("Model", models)
                
                # RUN SEARCH
                
                search = st.button("Search")
                
            if search:
                st.session_state.search_state = 1
            if st.session_state.search_state == 1:
                
                with st.spinner('Searching..'):
                    
                    def filter_df(df, filter_by_main):

                        df['run_id'].replace('', np.nan, inplace=True)
                        df.dropna(subset = ["run_id"], inplace=True)
                        for key, value in filter_by_main.items():
                            if value != []:
                                df = df[df[key].isin(value)]
                            else:
                                df = df
                        return df
                
                    filtered_runs_df = filter_df(runs_df, params.__dict__)
                    filtered_runs_df.reset_index(drop=True, inplace=True)  
                    
                    def display_generation(full_completion, display_key):
                        with st.expander(full_completion['prompt']):
                            completion_parameters = full_completion.drop('prompt')
                            st.write(completion_parameters.to_dict(), key=display_key)

                    fs = s3fs.S3FileSystem(anon=False)

                    session = boto3.Session(
                        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                    )
                    s3 = session.resource('s3')
                    bucket = s3.Bucket('aiaf-output')

                    def read_image(image):
                        object = bucket.Object(image.key)
                        img_data = object.get().get('Body').read()
                        return img_data
                    
                    def read_image_by_key(key):
                        object = bucket.Object(key)
                        img_data = object.get().get('Body').read()
                        return img_data
                    
                    def display_run_results(run_id, interval, display_type):
                        
                        if display_type == "Representative":
                            interval = interval
                        elif display_type == "All":
                            interval = 1
                        else:
                            return
                        
                        count = 0
                        for image in bucket.objects.all():
                            if image.key.startswith(folder_name):
                                if count % interval == 0:
                                    img_data = read_image(image)
                                    st.image(img_data, caption=image.key)
                                count += 1
                        
                    for i in range(len(filtered_runs_df)):
                        run_id = int(filtered_runs_df.iloc[i]['run_id'])
                        run_alias = sub("\W", "", filtered_runs_df.iloc[i]['run_alias'].lower().strip().replace(" ","_"))
                        folder_name = "E" + str(filtered_runs_df.iloc[i]['experiment_id_fk']) + "R" + str(int(filtered_runs_df.iloc[i]['run_id'])) + "_" + run_alias

                        def display_simple_results(run_id):
                            
                            imgicol, imgfcol = st.columns(2)
                            image_key_list = []
                            for image in bucket.objects.all():
                                if image.key.startswith(folder_name):
                                    image_key_list.append(image.key)
                            image_key_list_sorted = sorted(image_key_list)
                            
                            imgi_data = read_image_by_key(image_key_list_sorted[0])
                            imgf_data = read_image_by_key(image_key_list_sorted[-1])  
                            
                            imgicol.image(imgi_data, caption='Initial')
                            imgfcol.image(imgf_data, caption='Final')
                        
                        display_simple_results(filtered_runs_df.iloc[i]['run_id']) if view_mode == "Simple" else None
                                
                        with st.expander(folder_name + " Details") if view_mode == "Simple" else st.expander(folder_name):

                            in_params = Params()
                            scoreg, scoren, scorea, scorem = st.columns(4)
                            score_dict = {-1: 0, 0: 1, 1: 2, 'None': 1}
                            
                            current_score_good = filtered_runs_df.iloc[i]['score_good_fd']
                            current_score_novel = filtered_runs_df.iloc[i]['score_novel_fd']
                            current_score_aesthetic = filtered_runs_df.iloc[i]['score_aesthetic_fd']
                            current_score_meaning = filtered_runs_df.iloc[i]['score_meaning_fd']
                            
                            in_params.score_good = scoreg.selectbox("Good", [-1, 0, 1], index=score_dict[current_score_good], key=str(i) + "in_params")
                            in_params.score_novel = scoren.selectbox("Novel", [-1, 0, 1], index=score_dict[current_score_novel], key=str(i) + "in_params")
                            in_params.score_aesthetic = scorea.selectbox("Aesthetic", [-1, 0, 1], index=score_dict[current_score_aesthetic], key=str(i) + "in_params")
                            in_params.score_meaning = scorem.selectbox("Meaning", [-1, 0, 1], index=score_dict[current_score_meaning], key=str(i) + "in_params")

                            [tagpairs_wks, tagpairs_data, tagpairs_dict, tagpairs_df] = get_data_dict('tagpairs', 'tag_alias_fd', 'tagpair_item_id_fk')
                            
                            try:
                                filtered_tagpairs_df = tagpairs_df[tagpairs_df['tagpair_item_category'] == 'output'] 
                                filteredd_tagpairs_df = filtered_tagpairs_df.loc[filtered_tagpairs_df['tagpair_item_id_fk'] == run_id]
                                current_tags = filteredd_tagpairs_df['tag_alias_fd'].tolist()
                            except:
                                current_tags = []
                            tags, save = st.columns(2)
                            st.session_state.current_tags = tags.multiselect("Tags", options=st.session_state.current_tags if st.session_state.current_tags != [] else tags_list, default=current_tags, key=str(i) + "in_params")
                            in_params.tags = st.session_state.current_tags
                            
                            click_save_in_params = save.button("Update", key=i)
                            
                            if click_save_in_params:
                                
                                with st.spinner("Updating run info..."):
                                
                                    [outscores_wks, outscores_data, outscores_dict, outscores_df] = get_data_dict('output_scores', 'output_score_id', 'output_score_run_id_fk')
                                    if run_id in outscores_df['output_score_run_id_fk'].astype(int).tolist():
                                        row = outscores_df.loc[outscores_df['output_score_run_id_fk'] == run_id].index[0]
                                        outscores_wks.update_values(crange=('B{0}:E{0}'.format(row + 2)), values=[[in_params.score_good, in_params.score_novel, in_params.score_aesthetic, in_params.score_meaning]])
                                    else:
                                        last_row_score = len(outscores_data)+1
                                        new_score_data = [last_row_score, in_params.score_good, in_params.score_novel, in_params.score_aesthetic, in_params.score_meaning, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), int(filtered_runs_df.iloc[i]['run_id'])]
                                        outscores_wks.insert_rows(last_row_score, number=1, values=new_score_data)
                                    
                                    tags_list_old = list(tagpairs_dict.keys())
                                    unchanged_tags = set(current_tags).intersection(set(in_params.tags))
                                    tags_list_edit = list(set(in_params.tags).union(set(tags_list_old)).difference(unchanged_tags))
                                    st.write("TAGS LIST EDIT", tags_list_edit)
                                    for tag in tags_list_edit:
                                        if tag in tags_list_old:
                                            [tagpairs_wks, tagpairs_data, tagpairs_dict, tagpairs_df] = get_data_dict('tagpairs', 'tag_alias_fd', 'tagpair_item_id_fk')
                                            delete_row = tagpairs_df.index[tagpairs_df['tag_alias_fd'] == tag].tolist()[0]
                                            st.write("delete row", delete_row)
                                            tagpairs_wks.delete_rows(delete_row + 2)
                                        else:
                                            last_row_tag = len(tagpairs_data)+1
                                            tag_id = tags_df.loc[tags_df['tag_alias'] == tag]['tag_id'].values[0]
                                            tag_category = tags_df.loc[tags_df['tag_alias'] == tag]['tag_category'].values[0]
                                            new_tag_data = [str(last_row_tag), 'output', datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), str(run_id), str(tag_id), tag, tag_category]
                                            tagpairs_wks.insert_rows(last_row_tag, number=1, values=new_tag_data)
                                    
                                    st.success("Updated run info")
                            
                            img_side, param_side = st.columns(2)
                            with img_side:
                                display_run_results(filtered_runs_df.iloc[i]['run_id'], 25, view_mode)
                            with param_side:
                                st.markdown("##### Parameters")
                                completion_parameters = filtered_runs_df.loc[i].drop(['run_prompt', 'experiment_id_fk', 'experiment_alias_fd', 'project_id_fk', 'project_alias_fd'])
                                st.image(filtered_runs_df.loc[i]['run_initial_img'], caption='Initial Image') if filtered_runs_df.loc[i]['run_initial_img'] else None
                                st.image(filtered_runs_df.loc[i]['run_target_imgs'], caption='Target Image') if filtered_runs_df.loc[i]['run_target_imgs'] else None
                                st.write(completion_parameters.to_dict(), key=i)
                                
                    if len(filtered_runs_df) == 0:
                        st.success("No runs found.")
                            