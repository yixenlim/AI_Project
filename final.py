import streamlit as st
import altair as alt
import pandas as pd, seaborn as sns,numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# q1
import time,random

# q2
import constraint,math

# q3
# from sklearn.cluster import KMeans
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import silhouette_score, roc_auc_score, roc_curve,confusion_matrix,precision_score, recall_score, f1_score

st.sidebar.header('TIC3151 - ARTIFICIAL INTELLIGENCE PROJECT')

question = st.sidebar.radio('Please choose a question',['Q1','Q2','Q3'])

if question == 'Q1':
    st.title('Q1 - Vacation Planner (Genetic Algorithm)')
    st.write('---')

    st.header("Overview")
    st.write("The Genetic Algorithm (GA) is a search-based optimization technique based on genetics and natural selection principles. It's routinely used to find optimal or near-optimal solutions to tough problems that would take an eternity to solve otherwise. It's commonly utilised to tackle optimization problems, as well as in research and machine learning.")
    
    st.header("Challenges")
    st.write("As in regards to the challenges faced by us in this question, one of the hardest problem for us is determining the random ranges of each of the items. Next, we have to write our own fitness function to determine the fitness score of the individual combinations that required us to use logical thinking skills and common sense to complete.")
    st.write('---')

    cols = st.columns(2)
    with cols[0]:
        MONEY = int(st.number_input('Insert vacation budget', min_value=100, step=1, value=5000))
    with cols[1]:
        VACATION_DURATION = int(st.number_input('Insert vacation duration', min_value=1, step=1, value=5))

    cols = st.columns(2)
    with cols[0]:
        POPULATION_SIZE = int(st.number_input('Insert population size', min_value=1, step=1, value=10))
    with cols[1]:
        GENERATION_NUM = int(st.number_input('Insert number of generations', min_value=1, step=1, value=100))

    cols = st.columns(3)
    with cols[0]:
        RETAIN_PERC = st.number_input('Insert retain percentage', min_value=0.0001, max_value=1.0, value=0.2)
    with cols[1]:
        RAND_SELECT_PERC = st.number_input('Insert random select percentage', min_value=0.0001, max_value=1.0, value=0.05)
    with cols[2]:
        MUTATE_PERC = st.number_input('Insert mutation percentage', min_value=0.0001, max_value=1.0, value=0.01)
    
    HOTEL_PER_NIGHT_RM = int(MONEY * 0.05)
    TOURIST_SPOT_NUM = int(VACATION_DURATION * 3)
    TOURIST_SPOT_RM = int(MONEY * 0.06)
    FOOD_MEAL_RM = int((MONEY * 0.2) / (VACATION_DURATION * 3))
    TRANSPORT_FREQ_NUM = 15
    TRANSPORT_TRIP_RM = int(MONEY * 0.01)

    def generate_combination():
        combination = []
        combination.append(random.randint(int(HOTEL_PER_NIGHT_RM*0.1), HOTEL_PER_NIGHT_RM))
        combination.append(random.randint(2, TOURIST_SPOT_NUM))
        combination.append(random.randint(0, TOURIST_SPOT_RM))
        combination.append(random.randint(1, FOOD_MEAL_RM))
        combination.append(random.randint(1, TRANSPORT_FREQ_NUM))
        combination.append(random.randint(1, TRANSPORT_TRIP_RM))
        
        return combination

    def generate_population(size):
        population = []
        for i in range(size):
            population.append(generate_combination())
        
        return population

    #Fitness score LOWER IS BETTER
    def fitness(individual):
        cost = 0
        
        cost += individual[0] * VACATION_DURATION
        cost += individual[1] * individual[2]
        cost += individual[3] * VACATION_DURATION * 3
        cost += individual[4] * individual[5] * VACATION_DURATION
        
        if cost > MONEY:
            return MONEY

        return abs(MONEY-cost)

    def grade(pop):
        total = 0
        for p in pop:
            f = fitness(p)
            total += f
            
        return total / len(pop)
                
    def evolve(pop, retain=0.2, random_select=0.05, mutate=0.01): #rank selection, random mutation, single point crossover
        graded = [(fitness(x),x) for x in pop]

        graded = [x[1] for x in sorted(graded)]
        
        retain_length = int(len(graded)*retain)
        parents = graded[0:retain_length]

        for individual in graded[retain_length:]:
            if random_select > random.random():
                parents.append(individual)

        for individual in parents:
            if mutate > random.random():
                pos_to_mutate = random.randint(0,len(individual)-1)
                individual[pos_to_mutate] = random.randint(min(individual),max(individual))
        
        #crossover parents to create children
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = random.randint(0,parents_length) - 1
            female = random.randint(0,parents_length) - 1
            if male != female:
                male = parents[male]
                female = parents[female]
                half = int(len(male)/2)
                child = male[:half] + female[half:]
                children.append(child)
                
        parents.extend(children)
        return parents

    start_time = time.time()
    value_lst =[]
    fitness_history = []

    p = generate_population(POPULATION_SIZE)

    for i in range(GENERATION_NUM):
        p = evolve(p, retain=RETAIN_PERC, random_select=RAND_SELECT_PERC, mutate=MUTATE_PERC)
        value = grade(p)
        value_lst.append((p[0], value))
        fitness_history.append(value)

    end_time = time.time()
    run_time = end_time-start_time
    st.write('')

    fitness_data = pd.DataFrame({'Generation':range(1,GENERATION_NUM+1),'Fitness_score':fitness_history})
    fitness_line = alt.Chart(fitness_data,title='Fitness graph over generations based on user input').mark_line().encode(
                                                                                                                            x='Generation',
                                                                                                                            y='Fitness_score'
                                                                                                                        )
    fitness_marks = alt.Chart(fitness_data).mark_circle(size=50).encode(
                                                                        x='Generation',
                                                                        y='Fitness_score',
                                                                        tooltip=['Generation','Fitness_score']
                                                                    ).interactive()
    st.altair_chart(fitness_line+fitness_marks, use_container_width=True)

    st.write("Fitness score of last generation: " + str(value_lst[-1][1]))
    st.write("Hotel star rating: <RM" + str(value_lst[-1][0][0]))
    st.write("Tourist spots: " + str(value_lst[-1][0][1]) + " spots")
    st.write("One tourist spot: <RM" + str(value_lst[-1][0][2]))
    st.write("Food price: <RM" + str(value_lst[-1][0][3]) + " per meal")
    st.write("Transportation fees: <RM" + str(value_lst[-1][0][4]) + " per trip")
    st.write("Transport frequency: " + str(value_lst[-1][0][5]) + " trip per day")
    st.write("**Run time: " + str(run_time) + "s**")
    st.write('---')

    st.header('Detailed steps')
    st.write("1. Generate a population of individuals which consists of the criterias of the vacation in a list. (eg. hotel star rating)")
    st.write("2. Evolve the population using the parameters set by user, which are parent retention, random selection and mutation percentages.")
    st.write("3. Evaluate the fitness score of every individual in the population and sort the individuals by their fitness score and add the top individuals with highest fitness score to a list (as set by user).")
    st.write("4. Randomly select individuals from the population list to append to the top fitness score individuals list.")
    st.write("5. Randomly mutate items in the individuals.")
    st.write("6. Use single point crossover to generate new individual by merging 2 halves of 2 individuals.")
    st.write("7. Repeat from step 2 until generation limit is reached.")
    st.write('---')

    st.header('Comparison')

    cols = st.columns(3)
    with cols[0]:
        image1 = Image.open('Q1/r02_rs005_m0.01.png')
        st.image(image1)#,width=800
    with cols[1]:
        image2 = Image.open('Q1/r05_rs01_m005.png')
        st.image(image2)
    with cols[2]:
        image3 = Image.open('Q1/r07_rs02_m01.png')
        st.image(image3)
        
    st.write("The three figures plots the fitness score to the number of generations the population has evolved using the same budget and vacation days. In the first figure, we evolved the population using a parent retention percentage of 0.2, random selection of 0.05 and mutation of 0.01. A fitness score of around 200 was obtained before the 10th generation. Next using parent retention percentage of 0.5, random selection of 0.1 and mutation of 0.05, we obtained a fitness score of around 200 as well but only at the 19th generation. In the last plot, we used parent retention percentage of 0.7, random selection of 0.2 and mutation of 0.1 to evolve the population and found that it took around 30 generations to obtain a the fitness score of 1000. We can say that the higher the retention, random selection and mutation percentage, the slower it takes to reach a better fitness score.")
             
elif question == 'Q2':
    st.title('Q2 - Vaccine Distribution Modeling (Constraint-Satisfaction Problems)')
    st.write('---')

    st.header('Description and formulation of the problem')
    st.subheader('Objectives')
    st.write('- Assign the right vaccine types and amount to the vaccination centres.')
    st.write('- Keep the rental as low as possible.')
    st.write('- Complete the vaccination as soon as possible.')
    st.subheader('Limitations')
    st.write('- Each state has different numbers and types of vaccination centres.')
    st.write('- Each state has maximum capacity of vaccination per day.')
    st.write('---')

    st.header('Detailed steps in CSP Problem Formulation')
    st.write('1. Calculate the number of vaccination center needed for ONE DAY in each state under the condition achieving the maximum vaccination capacity of the state but with the lowest rental.')
    st.write('2. By using that number of vaccination center, calculate how many days needed. (Normal day)')
    st.write('3. After that, there must be some remaining population which the number is less than the maximum vaccination capacity of the state after the vaccination has been given for N days calculated above.')
    st.write('4. Recalculate the number of vaccination center needed for the remaining population with the lowest rental. (Last day)')
    st.write('5. Lastly, calculate the total rental needed.')
    st.info('Note: All three types of vaccine will be given each day by the percentage of that age group over the total population in the state.')
    st.write('---')

    st.header('Important parameters and functions in CSP Problem Formulation')
    st.write('- One problem was formulated for normal day and last day.')
    st.write('- The problem has one constraint: Do not exceed the maximum vaccination capacity of the state.')
    st.write('- For normal day, get the solution under two conditions: Achieve as many vaccination as possible at the lowest rental.')
    st.write('- For last day, get the solution which costs the lowest rental.')
    st.write('---')

    st.header('Assessing and Ranking the Solutions')
    st.write('- For normal day, achieve the maximum possible vaccination and the lowest rental.')
    st.write('- For last day, achieve the lowest rental.')
    st.info('Note: There might be several solutions that have the same maximum possible vaccination, thus finding the lowest rental is the grading.')
    st.write('---')

    st.header('Result')
    state = st.selectbox('Please select a state',['ST-1','ST-2','ST-3','ST-4','ST-5'],index=0)
    state_dict = {'ST-1':0,'ST-2':1,'ST-3':2,'ST-4':3,'ST-5':4}
    i = state_dict[state]

    number_of_center = [[21,16,11,22,6],
                    [31,17,16,11,3],
                    [23,16,12,13,4],
                    [17,11,21,16,2],
                    [20,11,21,16,2]]
    state_max_cap = [5000,10000,7500,8500,9500]
    state_population_c_b_a = [[115900,434890,15000],
                            [100450,378860,35234],
                            [223400,643320,22318],
                            [269300,859900,23893],
                            [221100,450500,19284]]

    # grading fucntion
    def grade(vaccine_per_day,max_cap):
        return abs(vaccine_per_day-max_cap)

    # sort the solution according to state
    def sort_solution(s):
        sorted = {}
        sorted['CR-1'] = s['CR-1']
        sorted['CR-2'] = s['CR-2']
        sorted['CR-3'] = s['CR-3']
        sorted['CR-4'] = s['CR-4']
        sorted['CR-5'] = s['CR-5']

        return sorted

    # initialize the problem
    problem = constraint.Problem()

    # set varaibles: number of center
    problem.addVariable('CR-1', range(number_of_center[i][0]))  
    problem.addVariable('CR-2', range(number_of_center[i][1]))
    problem.addVariable('CR-3', range(number_of_center[i][2]))
    problem.addVariable('CR-4', range(number_of_center[i][3]))
    problem.addVariable('CR-5', range(number_of_center[i][4]))

    # check whether the solutions exceed the maximum vaccination capacity of the state
    def check_state_max_cap(a, b, c, d, e):  
        if (a*200 + b*500 + c*1000 + d*2500 + e*4000) <= state_max_cap[i]:
            return True

    # add the constraint
    problem.addConstraint(check_state_max_cap,['CR-1','CR-2','CR-3','CR-4','CR-5'])

    #####--------------------------------- Normal Day ---------------------------------------#####
    minimum_rental_normal = float('inf')
    dif = float('inf')
    solution_found_normal = {}
    total_day = 0
    solutions = problem.getSolutions()

    for s in solutions:
        vaccine_per_day = s['CR-1']*200 + s['CR-2']*500 + s['CR-3']*1000 + s['CR-4']*2500 + s['CR-5']*4000

        if (vaccine_per_day != 0 and dif >= grade(vaccine_per_day,state_max_cap[i])): #reach max capacity from all the solutions
            current_rental_per_day = s['CR-1']*100+s['CR-2']*250+s['CR-3']*500+s['CR-4']*800+s['CR-5']*1200

            # find the minimum rental
            if current_rental_per_day < minimum_rental_normal:
                dif = grade(vaccine_per_day,state_max_cap[i])
                minimum_rental_normal = current_rental_per_day
                solution_found_normal = s

                total_day = math.floor(sum(state_population_c_b_a[i]) / vaccine_per_day)

    # vaccine given per day
    current_vaccine_given = total_day * (solution_found_normal['CR-1']*200 + solution_found_normal['CR-2']*500 + solution_found_normal['CR-3']*1000 + solution_found_normal['CR-4']*2500 + solution_found_normal['CR-5']*4000)
    
    # remaining population for the last day
    remaining = sum(state_population_c_b_a[i])-current_vaccine_given

    #####--------------------------------- Remaining ---------------------------------------#####
    minimum_rental_last = float('inf') 
    solution_found_last = {}

    for s in solutions:  
        vaccine_given = s['CR-1']*200 + s['CR-2']*500 + s['CR-3']*1000 + s['CR-4']*2500 + s['CR-5']*4000 #capacity

        if vaccine_given >= remaining:
            remaining_rental = s['CR-1']*100+s['CR-2']*250+s['CR-3']*500+s['CR-4']*800+s['CR-5']*1200

            if remaining_rental < minimum_rental_last:
                minimum_rental_last = remaining_rental
                solution_found_last = s

    st.write('Total population:',sum(state_population_c_b_a[i]))
    st.write('Vac_A needed:',state_population_c_b_a[i][2])
    st.write('Vac_B needed:',state_population_c_b_a[i][1])
    st.write('Vac_C needed:',state_population_c_b_a[i][0])
    st.write('Total day needed:',total_day+1,'days')
    st.write('Total rental: RM{}\*{} days + RM{}\*1 day = RM{}'.format(minimum_rental_normal,total_day,minimum_rental_last,minimum_rental_normal*total_day+minimum_rental_last))    
    st.write('---')
    st.write('Normal day solution:',sort_solution(solution_found_normal))
    vac_per_normal_day = solution_found_normal['CR-1']*200 + solution_found_normal['CR-2']*500 + solution_found_normal['CR-3']*1000 + solution_found_normal['CR-4']*2500 + solution_found_normal['CR-5']*4000
    vac_A = math.floor(vac_per_normal_day*state_population_c_b_a[i][2]/sum(state_population_c_b_a[i]))
    vac_B = math.floor(vac_per_normal_day*state_population_c_b_a[i][1]/sum(state_population_c_b_a[i]))
    vac_C = vac_per_normal_day - vac_A - vac_B
    st.write('Normal day vaccination per day:',vac_per_normal_day)
    st.write('Vac_A per day:',vac_A)
    st.write('Vac_B per day:',vac_B)
    st.write('Vac_C per day:',vac_C)
    st.write('Normal day total vaccination coverage:',(solution_found_normal['CR-1']*200 + solution_found_normal['CR-2']*500 + solution_found_normal['CR-3']*1000 + solution_found_normal['CR-4']*2500 + solution_found_normal['CR-5']*4000)*total_day)
    st.write('---')
    st.write('Last day solution:',sort_solution(solution_found_last))
    vac_A = state_population_c_b_a[i][2] - (vac_A*total_day)
    vac_B = state_population_c_b_a[i][1] - (vac_B*total_day)
    vac_C = state_population_c_b_a[i][0] - (vac_C*total_day)
    st.write('Last day vaccination:',remaining)
    st.write('Vac_A last day:',vac_A)
    st.write('Vac_B last day:',vac_B)
    st.write('Vac_C last day:',vac_C)
    st.write('Last day vaccination coverage:',solution_found_last['CR-1']*200 + solution_found_last['CR-2']*500 + solution_found_last['CR-3']*1000 + solution_found_last['CR-4']*2500 + solution_found_last['CR-5']*4000)

elif question == 'Q3':
    st.title('Q3 - Loan Application Modeling')
    st.write('---')
    
    st.header('Description of the dataset')
    dataset = pd.read_csv('Bank_CreditScoring.csv')
    st.dataframe(dataset)

    df_types = pd.DataFrame(dataset.dtypes, columns=['Data_type'])
    numerical_cols = df_types[~df_types['Data_type'].isin(['object','bool'])].index.values
    df_types['Count'] = dataset.count()
    df_types['Unique values'] = dataset.nunique()
    df_types['Null values'] = dataset.isnull().sum()
    df_types['Min'] = dataset[numerical_cols].min()
    df_types['Max'] = dataset[numerical_cols].max()
    df_types['Mean'] = dataset[numerical_cols].mean()
    df_types['Median'] = dataset[numerical_cols].median()
    df_types['St. Dev.'] = dataset[numerical_cols].std()

    df_types.loc[df_types.Data_type == 'object', ['Min','Max','Mean','Median','St. Dev.']] = '-'
    df_types.loc[df_types.Data_type == 'int64', 'Unique values'] = '-'

    st.dataframe(df_types.astype(str))
    
    # Correction of the dataset
    st.subheader('Correction of the dataset')
    col1, col2 = st.columns(2)

    with col1:
        st.write('Before correction:',pd.DataFrame(np.unique(dataset['State']),columns=['State']))

    dataset.loc[dataset['State'] == 'Johor B', 'State'] = 'Johor'
    dataset.loc[dataset['State'] == 'K.L', 'State'] = 'Kuala Lumpur'
    dataset.loc[dataset['State'] == 'N.S', 'State'] = 'N.Sembilan'
    dataset.loc[dataset['State'] == 'P.Pinang', 'State'] = 'Pulau Penang'
    dataset.loc[dataset['State'] == 'Penang', 'State'] = 'Pulau Penang'
    dataset.loc[dataset['State'] == 'SWK', 'State'] = 'Sarawak'
    dataset.loc[dataset['State'] == 'Trengganu', 'State'] = 'Terengganu'

    with col2:
        st.write('After correction:',pd.DataFrame(np.unique(dataset['State']),columns=['State']))
    st.write('---')
    
    # Count plot for decision of applications
    count_plot_data = dataset['Decision'].value_counts().reset_index()
    count_plot_data.columns = ['Decision','Count']
    count_plot = alt.Chart(
                            count_plot_data,
                            title='Count plot for decision of applications'
                        ).mark_bar().encode(
                            x="Decision",
                            y='Count',
                            color=alt.Color('Decision', scale=alt.Scale(scheme='paired'))
                        ).configure_axisX(
                            labelAngle=0
                        ).properties(
                            height=400
                        )
    st.altair_chart(count_plot, use_container_width=True)
    st.write('---')

    # Distribution by decision & month salary group
    fig = plt.figure(figsize=(6,3))
    analysis = dataset.copy()
    analysis['Salary_Group'] = pd.cut(analysis['Monthly_Salary'],bins = np.arange(0,16000,2000))

    dist = analysis.groupby(['Salary_Group', 'Decision'])['Decision'].count()
    dist = dist.unstack('Decision').reset_index()
    dist['Salary_Group'] = dist['Salary_Group'].astype(str)
    dist2 = pd.melt(dist, id_vars=['Salary_Group'], value_vars=['Accept','Reject'])
    d = alt.Chart(
                    dist2,
                    title='Distribution by decision & month salary group'
                ).mark_bar().encode(
                    x=alt.X("value", type="quantitative",title='Count'),
                    y=alt.Y("Salary_Group", type="nominal",sort=(['(0, 2000]','(2000, 4000]','(4000, 6000]','(6000, 8000]','(8000, 10000]','(10000, 12000]','(12000, 14000]'])),
                    color=alt.Color("Decision", type="nominal")
                ).properties(
                    height=400
                )
    st.altair_chart(d, use_container_width=True)
    st.write('---')

    # Percentage of each employment type get accepted
    employment_list = []
    success_rate_list = []
    employment_Type = pd.unique(dataset['Employment_Type'].values)

    for employment in employment_Type:
        current = dataset[dataset['Employment_Type'] == employment]
        total_application = sum(current['Decision'].value_counts())
        accept = current['Decision'].value_counts()[0]/total_application
        
        employment_list.append(employment)
        success_rate_list.append(round(accept*100,2))
    
    success_rate_data = pd.DataFrame({'Employment_type':employment_list,'Success_rate(%)':success_rate_list})

    success_rate_plot = alt.Chart(
                                    success_rate_data,
                                    title='Success rate of loan application by employment type'
                                ).mark_bar().encode(
                                    x=alt.X("Employment_type", type="nominal",sort='-y'),
                                    y=alt.Y("Success_rate(%)", type="quantitative"),
                                    color=alt.Color('Employment_type', scale=alt.Scale(scheme='paired'))
                                ).configure_axisX(
                                    labelAngle=0
                                ).properties(
                                    height=400
                                )
    st.altair_chart(success_rate_plot, use_container_width=True)
    st.write('---')
    
    # Correlation between Loan_Amount and some other variables
    cor = dataset.corr().reset_index().melt('index')
    cor.columns = ['var1', 'var2', 'correlation']
    col_names = np.unique(cor['var1'])

    base = alt.Chart(cor,title="Correlation between Loan_Amount and some other variables").transform_filter(
                                            alt.datum.var1 < alt.datum.var2
                                        ).encode(
                                            x=alt.X("var1", type="nominal", title="",sort=col_names),
                                            y=alt.X("var2", type="nominal", title="",sort=col_names)
                                        ).properties(
                                            height=800
                                        )

    rects = base.mark_rect().encode(color='correlation')

    text = base.mark_text(
                            size=10
                        ).encode(
                            text=alt.Text('correlation', format=".2f"),
                            color=alt.condition(
                                "datum.correlation > 0.5",
                                alt.value('white'),
                                alt.value('black')
                            )
                        )

    st.altair_chart(rects + text, use_container_width=True)
    st.write('---')

    st.header('Model Construction and Assessment')

    st.header('Cluster analysis: K-means')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Advantages:')
        st.write('- Relatively simple to implement.')
        st.write('- Guarantees convergence.')
        st.write('- Scales to large data sets.')
        st.write('- Easily adapts to new examples.')
    with col2:
        st.subheader('Disadvantages:')
        st.write('- Clustering outliers.')
        st.write('- Choosing k value manually.')
    st.write('---')

    st.header('Find the optimal k value')

    col1, col2, col3 = st.columns(3)
    with col1: # elbow: line very smooth, unclear
        image = Image.open('Q3/Clustering/km_distortion.png')
        st.image(image)
    with col2: # silhouette: choose highest
        image = Image.open('Q3/Clustering/km_silhouette.png')
        st.image(image)#km_silhouette_visualizer
    with col3: # silhouette: choose highest
        image = Image.open('Q3/Clustering/km_silhouette_visualizer.png')
        st.image(image)
    st.write('---')

    image = Image.open('Q3/Clustering/km_result.png')
    st.image(image)
    st.write('---')

    st.header('Classification: Random forest classifier and Support vector classifier')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Random forest classifier')
        st.write('- Bagged decision tree models.')
        st.write('- Split on a random subset of features on each split.')
        st.write('- Able to handle binary features, categorical features, and numerical features.')
        st.write('- Work great with high dimensional data.')
    with col2:
        st.subheader('Support vector classifier')
        st.write('- Memory efficient.')
        st.write('- More parameters can be tuned in order to get the best performance.')
        st.write('- Effective in high dimensional spaces.')
    st.write('---')

    # ---------------------------------------------------------------------------------------------------------------------
    st.subheader('Random forest classifier')

    col1, col2, col3 = st.columns(3)
    with col1:
        image = Image.open('Q3/Classification/rf_depth.png')
        st.image(image)
    with col2:
        image = Image.open('Q3/Classification/rf_criterion.png')
        st.image(image)
    with col3:
        image = Image.open('Q3/Classification/rf_n.png')
        st.image(image)

    st.write('---')

    # ---------------------------------------------------------------------------------------------------------------------
    st.subheader('Support vector classifier')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        image = Image.open('Q3/Classification/svm_kernel.png')
        st.image(image)
    with col2:
        image = Image.open('Q3/Classification/svm_gamma.png')
        st.image(image)
    with col3:
        image = Image.open('Q3/Classification/svm_degree.png')
        st.image(image)

    st.write('---')
    # ---------------------------------------------------------------------------------------------------------------------
    st.subheader('Comparison')

    image = Image.open('Q3/Classification/classification_evaluation.png')
    st.image(image)
    st.write('---')

    image = Image.open('Q3/Classification/classification_cm.png')
    st.image(image)
    st.write('---')

    # ---------------------------------------------------------------------------------------------------------------------
    st.header('Recommendation')
    st.write('- Explore more types of clustering technique and classifiers.')
    st.write('- Explore more complex parameters that can be tuned.')
    st.write('---')

    st.header('Conclusion')
    st.write('- Machine learning can be unsupervised or supervised.')
    st.write('- Exploratory Data Analysis (EDA) is a significant step before training a model.')
    st.write('- Unbalanced dataset can reduce the model performance.')
    st.write('- Model fine-tuning is needed to achieve higher accuracy.')