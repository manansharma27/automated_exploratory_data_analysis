
#Importing important libraries
import streamlit as st 
import pandas as pd  
import numpy as np
import base64
import io
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('deprecation.showPyplotGlobalUse', False)



# Data Viz Pkg
import matplotlib.image as img 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sb 
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score


# Making a function to download csv file.
def csv_download(data):
    csvfile = data.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    st.markdown("Download CSV File from below.")
    href = f'<a href="data:file/csv;base64,{b64}" download="updated_dataset.csv"> Download Here!!</a>'
    st.markdown(href,unsafe_allow_html=True)

#Making the main function
def main():
    """ Automated Exploratory Data Analysis App """
    st.header("Automated Data Analysis App")
    task = ['Data Exploration','EDA','Fill Null Values','Machine Learning']
    choice = st.sidebar.selectbox("Select Options",task)   #Making a sidebar for selection of activity
    data = st.file_uploader("Upload a dataset to analysis",type=["csv"])
    if(choice == 'Data Exploration'):
	#Reading image and storing it
	image = img.imread("eda.png")
	#Displaying Image
	st.image(image)    
        st.subheader("Basic Data Exploration!")
        
        
        if data is not None:
            df = pd.read_csv(data,encoding='utf-8',engine='python')  #Reading the csv file
            st.dataframe(df.head())
            
            st.write("Stastistics of dataset")
            st.write(df.describe())
            st.write("Shape")
            st.write(df.shape)
            st.write("Info of columns")
            buffer = io.StringIO()   
            df.info(buf=buffer)
            s = buffer.getvalue()   #Assigning the string data of info to s
            st.text(s)
                
            if st.checkbox("Show Columns"):   
                st.write(df.columns)    #To show columns of the dataset
                
                
            if st.checkbox("Null Values"):
                st.write(df.isnull().sum())     #Checking Null values
                
            if st.checkbox("Visualize Null Values"):
                st.write(sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis'))      #Visulaizing null values
                st.pyplot()
                
            if st.checkbox("Show Correlation Map of Columns"):
                st.write(sb.heatmap(df.corr(numeric_only=True),annot=True))      #Checking correlation using heatmap
                st.pyplot()
                
            if st.checkbox("Check Outliers of Columns using IQR"):
                all_columns_names = df.columns.tolist()
                selected_columns_name = st.selectbox("Select Columns To detect outliers",all_columns_names)
                # Outlier Observation Analysis
                for feature in df[[selected_columns_name]]:
    
                    Q1 = df[feature].quantile(0.25)
                    Q3 = df[feature].quantile(0.75)
                    IQR = Q3-Q1
                    lower = Q1- 1.5*IQR
                    upper = Q3 + 1.5*IQR
                    
                    if df[(df[feature] > upper)].any(axis=None) or df[(df[feature] < lower)].any(axis=None):
                        st.write(feature," - yes")
                        st.write(sb.boxplot(df[feature]))
                        st.pyplot()
                    else:
                        st.write(feature, " - no")
                
    elif choice == 'EDA':
	#Reading image and storing it
	image = img.imread("eda.png")
	#Displaying Image
	st.image(image)
        st.subheader("Data Visualization")
        #data= st.file_uploader("Upload a dataset",type=["csv"])
        if data is not None:
            df = pd.read_csv(data,encoding = 'utf-8',engine='python')
            st.dataframe(df.head())
            
            if st.checkbox("Info of columns"):
                st.write("Info of columns")
                buffer = io.StringIO() 
                df.info(buf=buffer)
                s = buffer.getvalue()     #Assigning the string data of info to s
                st.text(s)
                
                all_columns_names = df.columns.tolist()
                #Making a dropdown option to choose the type of plot to generate
                type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","histplot","box","kde","pie","pairplot"])
                selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
                
                if st.button("Generate Plot"):
                    st.success("Generating Plot of {} for {}".format(type_of_plot,selected_columns_names))
                    
                    # Plots provided by Streamlit
                    if type_of_plot == "area":
                        plot_data = df[selected_columns_names]
                        st.area_chart(plot_data)        #Making area chart
                        
                    elif type_of_plot == "bar":
                        plot_data = df[selected_columns_names]
                        st.bar_chart(plot_data,use_container_width=True)  #Making bar chart
                        
                    elif type_of_plot == "line":
                        plot_data = df[selected_columns_names]
                        st.line_chart(plot_data)        #Making line chart
                      
                        #Plotting pie plot using matplotlib
                    elif type_of_plot == "pie":
                        plot_data = df[selected_columns_names].sum()   #Taking sum of each column and plotting it
                        newdf = df[selected_columns_names]
                        labels = newdf.columns
                        st.write(plt.pie(plot_data,labels=labels,autopct="%1.1f%%",startangle=90)) 
                        st.pyplot()
                        
                        # Creating Pairplot of the dataset
                    elif type_of_plot == "pairplot":
                        st.write(sb.pairplot(df))
                        st.pyplot()
                        
                    elif type_of_plot == "histplot":
                        plt.figure(figsize=(8,5))
                        plot_data = df[selected_columns_names]
                        st.write(sb.histplot(plot_data,kde=True))
                        st.pyplot()
                        
                        
                     #Custom Plot   
                    elif type_of_plot:
                        plot = df[selected_columns_names].plot(kind=type_of_plot)
                        st.write(plot)
                        st.pyplot()
    
    elif choice == 'Fill Null Values':
	#Reading image and storing it
	image = img.imread("eda.png")
	#Displaying Image
	st.image(image)
        st.subheader("Fill Null Values")
        #data= st.file_uploader("Upload a dataset",type=["csv"])
        if data is not None:
            df = pd.read_csv(data,encoding = 'utf-8',engine='python')
            st.dataframe(df.head())
            
            st.write("Info of columns")
            buffer = io.StringIO()   
            df.info(buf=buffer)
            s = buffer.getvalue()   #Assigning the string data of info to s
            st.text(s)
            
            if st.checkbox("Visualize Null Values"):
                st.write(sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis'))
                st.pyplot()
            
            all_columns_names = df.columns.tolist()
            type_fill_null = st.selectbox("Select Type to fill null values",["mean","median","mode"])
            selected_columns_name = st.selectbox("Select Column To fill null values",all_columns_names)
            
            if type_fill_null == "mean":
                if st.button('Fill Null Value'):
                    #Filling numeric null values with mean of that column
                    df[selected_columns_name]=df[selected_columns_name].fillna(df[selected_columns_name].mean())
                    st.write("Null values has been filled by mean of that column.")
                    st.write('Number of null values:',df[selected_columns_name].isnull().sum())
                    csv_download(df)     #Calling download function to download updated csv file
                    
            if type_fill_null == "median":
                if st.button('Fill Null Value'):
                    #Filling numeric null values with median of that column
                    df[selected_columns_name]=df[selected_columns_name].fillna(df[selected_columns_name].median())
                    st.write("Null values has been filled by median of that column.")
                    st.write('Number of null values:',df[selected_columns_name].isnull().sum())
                    csv_download(df)    ##Calling download function to download updated csv file
                    
            if type_fill_null == "mode":
                if st.button('Fill Null Categorical Values'):
                    #Filling categorical null values with mode of that column
                    df[selected_columns_name]=df[selected_columns_name].fillna(df[selected_columns_name].mode()[0])
                    st.write("Null values has been filled by mode of that column.")
                    st.write('Number of null values:',df[selected_columns_name].isnull().sum())
                    csv_download(df)     ##Calling download function to download updated csv file
                    
    elif choice == 'Machine Learning':
        #Reading image and storing it
        image = img.imread("ML.png")
        #Displaying Image
        st.image(image)
        st.subheader("Training Machine Learning model")
        #data= st.file_uploader("Upload a dataset",type=["csv"])
        if data is not None:
            df = pd.read_csv(data,encoding = 'utf-8',engine='python')
            st.dataframe(df.head())
            
            if st.checkbox("Info of columns"):
                st.write("Info of columns")
                buffer = io.StringIO() 
                df.info(buf=buffer)
                s = buffer.getvalue()     #Assigning the string data of info to s
                st.text(s)
            if st.checkbox("Show Columns"):   
                st.write(df.columns)    #To show columns of the dataset
                
                
            if st.checkbox("Null Values"):
                st.write(df.isnull().sum())     #Checking Null values
                
            if st.checkbox("Visualize Null Values"):
                st.write(sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis'))      #Visulaizing null values
                st.pyplot()
                
            if st.checkbox("Show Correlation Map of Columns"):
                
                st.write(sb.heatmap(df.corr(numeric_only=True),annot=True))      #Checking correlation using heatmap
                st.pyplot()
                    
                
            
            if st.checkbox("What type of problem do you want to solve?"):
                problem_type = st.selectbox("Select Type",["Regression","Classification"])
                
                if problem_type == "Regression":
                    st.checkbox("Please select the type of machine learning algorithm")
                    ml_type = st.selectbox("Select Type",["Linear Regression","Decision Tree Regression","Random Forest Regression","KNN Regression","SVM Regression"])
                    
                    if ml_type == "Linear Regression":
                        #X = df.iloc[:,:-1] # Using all column except for the last column as X
                        #Y = df.iloc[:,-1] # Selecting the last column as Y
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        from sklearn.linear_model import LinearRegression   #import the func "linear regression" from sllearn
                        lm = LinearRegression()     #making the object of linear regression model
                        model = lm.fit(xtrain,ytrain)
                        
                        
                        pred= model.predict(xtest)
                        if len(selected_columns_name_x)==1:
                            b0 = model.intercept_
                            b1 = model.coef_
                        
                            st.write("Intercept of model is %.2f"%b0)
                            st.write("Slope of model is %.2f"%b1)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["MAE","MSE","Accuracy Score"])
                        if metric_type == "MAE":
                            from sklearn.metrics import mean_absolute_error
                            MAE = mean_absolute_error(ytest,pred)
                            st.write("Overall error is %.2f"%MAE)
                            
                        elif metric_type == "MSE":
                            from sklearn.metrics import mean_squared_error
                            MSE = mean_squared_error(ytest,pred)
                            st.write("Overall error is %.2f"%MSE)
                            
                        elif metric_type == "Accuracy Score":
                            from sklearn.metrics import r2_score
                            r2 =r2_score(ytest,pred)
                            st.write("Overall accuracy score is ",(r2*100))
                        
                        if len(selected_columns_name_x)==1:
                            st.write("Regression Plot for training and testing set")
                            sb.regplot(x=xtrain,y=ytrain)
                            st.pyplot()
                        
                    elif ml_type == "Decision Tree Regression":
                        
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        criterion_type = st.selectbox("Select Criterion Type",["squared_error","absolute_error","friedman_mse","poisson"])
                        maxdepth = st.selectbox("Select Max Depth",[2,3,4,5,6,7,8,9,10])
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        
                        from sklearn.tree import DecisionTreeRegressor
                        dtr = DecisionTreeRegressor(criterion = criterion_type,random_state=100,max_depth=maxdepth)
                        model = dtr.fit(xtrain,ytrain)
                        predicted = model.predict(xtest)
                        #acc = accuracy_score(xtrain,ytrain)
                        #st.write("Training accuracy is",acc)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["MAE","MSE","Accuracy Score","RMSE"])
                        if metric_type == "MAE":
                            from sklearn.metrics import mean_absolute_error
                            MAE = mean_absolute_error(ytest,predicted)
                            st.write("Overall Mean Absolute Error is %.2f"%MAE)
                            
                        elif metric_type == "MSE":
                            from sklearn.metrics import mean_squared_error
                            MSE = mean_squared_error(ytest,predicted)
                            st.write("Overall Mean Squared Error is %.2f"%MSE)
                            
                        elif metric_type == "Accuracy Score":
                            from sklearn.metrics import r2_score
                            r2 =r2_score(ytest,predicted)
                            st.write("Overall accuracy score is ",(r2*100))
                            
                        elif metric_type == "RMSE":
                            from sklearn.metrics import mean_squared_error
                            rmse=np.sqrt(mean_squared_error(ytest, predicted))
                            st.write("Root Mean Squared Error is %.2f"%rmse)
                        
                        st.write("Decision Tree Graph for Model:-")
                        fig = plt.figure(figsize=(15,10))
                        plot_tree(dtr,filled=True)
                        st.pyplot()
                            
                        st.write("Feature importance in model:-")
                        plt.barh(X.columns,dtr.feature_importances_)
                        st.pyplot()
                        
                        
                    elif ml_type == "Random Forest Regression":
                        
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        criterion_type = st.selectbox("Select Criterion Type",["squared_error","absolute_error","friedman_mse","poisson"])
                        maxfeatures = st.selectbox("Select Max Feature Type",["sqrt", "log2"])
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        
                        from sklearn.ensemble import RandomForestRegressor
                        rf = RandomForestRegressor(n_estimators=100,criterion = criterion_type,random_state=100,max_depth=5,max_features=maxfeatures)
                        model = rf.fit(xtrain,ytrain)
                        predicted = model.predict(xtest)
                        #acc = accuracy_score(xtrain,ytrain)
                        #st.write("Training accuracy is",acc)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["MAE","MSE","Accuracy Score","RMSE"])
                        if metric_type == "MAE":
                            from sklearn.metrics import mean_absolute_error
                            MAE = mean_absolute_error(ytest,predicted)
                            st.write("Overall Mean Absolute Error is %.2f"%MAE)
                            
                        elif metric_type == "MSE":
                            from sklearn.metrics import mean_squared_error
                            MSE = mean_squared_error(ytest,predicted)
                            st.write("Overall Mean Squared Error is %.2f"%MSE)
                            
                        elif metric_type == "Accuracy Score":
                            from sklearn.metrics import r2_score
                            r2 =r2_score(ytest,predicted)
                            st.write("Overall accuracy score is ",(r2*100))
                            
                        elif metric_type == "RMSE":
                            from sklearn.metrics import mean_squared_error
                            rmse=np.sqrt(mean_squared_error(ytest, predicted))
                            st.write("Root Mean Squared Error is %.2f"%rmse)
                            
                        st.write("Feature importance in model:-")
                        plt.barh(X.columns,rf.feature_importances_)
                        st.pyplot()
                        
                    elif ml_type == "KNN Regression":
                        
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        weight_type = st.selectbox("Select Weight Type",["uniform", "distance"])
                        algorithm_type = st.selectbox("Select Algorithm Type",["auto", "ball_tree", "kd_tree", "brute"])
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        
                        from sklearn.neighbors import KNeighborsRegressor
                        knn = KNeighborsRegressor(n_neighbors=5,weights=weight_type,algorithm=algorithm_type)
                        model = knn.fit(xtrain,ytrain)
                        predicted = model.predict(xtest)
                        #acc = accuracy_score(xtrain,ytrain)
                        #st.write("Training accuracy is",acc)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["MAE","MSE","Accuracy Score","RMSE"])
                        if metric_type == "MAE":
                            from sklearn.metrics import mean_absolute_error
                            MAE = mean_absolute_error(ytest,predicted)
                            st.write("Overall Mean Absolute Error is %.2f"%MAE)
                            
                        elif metric_type == "MSE":
                            from sklearn.metrics import mean_squared_error
                            MSE = mean_squared_error(ytest,predicted)
                            st.write("Overall Mean Squared Error is %.2f"%MSE)
                            
                        elif metric_type == "Accuracy Score":
                            from sklearn.metrics import r2_score
                            r2 =r2_score(ytest,predicted)
                            st.write("Overall accuracy score is ",(r2*100))
                            
                        elif metric_type == "RMSE":
                            from sklearn.metrics import mean_squared_error
                            rmse=np.sqrt(mean_squared_error(ytest, predicted))
                            st.write("Root Mean Squared Error is %.2f"%rmse)
                            
                    elif ml_type == "SVM Regression":
                        
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        kernel_type = st.selectbox("Select Kernel Type",["linear", "poly", "rbf", "sigmoid", "precomputed"])
                        gamma_type = st.selectbox("Select Gamma Type",["scale", "auto"])
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        
                        from sklearn.svm import SVR
                        svm1 = SVR(kernel=kernel_type,gamma=gamma_type)
                        model = svm1.fit(xtrain,ytrain)
                        predicted = model.predict(xtest)
                        #acc = accuracy_score(xtrain,ytrain)
                        #st.write("Training accuracy is",acc)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["MAE","MSE","Accuracy Score","RMSE"])
                        if metric_type == "MAE":
                            from sklearn.metrics import mean_absolute_error
                            MAE = mean_absolute_error(ytest,predicted)
                            st.write("Overall Mean Absolute Error is %.2f"%MAE)
                            
                        elif metric_type == "MSE":
                            from sklearn.metrics import mean_squared_error
                            MSE = mean_squared_error(ytest,predicted)
                            st.write("Overall Mean Squared Error is %.2f"%MSE)
                            
                        elif metric_type == "Accuracy Score":
                            from sklearn.metrics import r2_score
                            r2 =r2_score(ytest,predicted)
                            st.write("Overall accuracy score is ",(r2*100))
                            
                        elif metric_type == "RMSE":
                            from sklearn.metrics import mean_squared_error
                            rmse=np.sqrt(mean_squared_error(ytest, predicted))
                            st.write("Root Mean Squared Error is %.2f"%rmse)
                            #--------------------------------------------------------------
                        
                if problem_type == "Classification":
                    st.checkbox("Please select the type of machine learning algorithm")
                    ml_type = st.selectbox("Select Type",["Logistic Regression","Decision Tree Classification","Random Forest Classification","KNN Classification","SVM Classification"])
                            
                    if ml_type == "Logistic Regression":
                        #X = df.iloc[:,:-1] # Using all column except for the last column as X
                        #Y = df.iloc[:,-1] # Selecting the last column as Y
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        df[selected_columns_name_y] = le.fit_transform(df[selected_columns_name_y])
                        
                        Y = df[selected_columns_name_y]
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        from sklearn.linear_model import LogisticRegression   #import the func "linear regression" from sllearn
                        lr = LogisticRegression()     #making the object of logistic regression model
                        model = lr.fit(xtrain,ytrain)
                        
                        
                        pred= model.predict(xtest)
                        
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["Confusion Matrix","Classification Report","Precision Score","Recall Score","F1 Score","Accuracy Score"])
                        if metric_type == "Confusion Matrix":
                            from sklearn.metrics import confusion_matrix
                            cm1 = confusion_matrix(ytest,pred)
                            st.write("Confusion Matrix is \t",cm1)
                            
                        elif metric_type == "Classification Report":
                            from sklearn.metrics import classification_report
                            cr = classification_report(ytest,pred)
                            st.write(cr)
                            
                        elif metric_type == "Precision Score":
                            from sklearn.metrics import precision_score
                            ps =precision_score(ytest,pred,average="micro")
                            st.write("Overall Precision Score is ",(ps*100))
                            
                        elif metric_type == "Recall Score":
                            from sklearn.metrics import recall_score
                            rs = recall_score(ytest,pred,average="micro")
                            st.write("Overall Recall Score is ",(rs*100))
                            
                        elif metric_type == "F1 Score":
                            from sklearn.metrics import f1_score
                            f1s =f1_score(ytest,pred,average="micro")
                            st.write("Overall F1 Score is ",(f1s*100))
                            
                        elif metric_type == "Accuracy Score":
                            #from sklearn.metrics import accuracy_score
                            acc1 = accuracy_score(ytest,pred)
                            st.write("Overall Accuracy Score is ",(acc1*100))
                            
                        
                        
                        
                        
                    elif ml_type == "Decision Tree Classification":
                        
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        criterion_type = st.selectbox("Select Criterion Type",["entropy", "gini", "log_loss"])
                        maxdepth = st.selectbox("Select Max Depth",[2,3,4,5,6,7,8,9,10])
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        
                        from sklearn.tree import DecisionTreeClassifier
                        dtc = DecisionTreeClassifier(criterion = criterion_type,random_state=100,max_depth=maxdepth)
                        model = dtc.fit(xtrain,ytrain)
                        pred = model.predict(xtest)
                        #acc = accuracy_score(xtrain,ytrain)
                        #st.write("Training accuracy is",acc)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["Confusion Matrix","Classification Report","Precision Score","Recall Score","F1 Score","Accuracy Score"])
                        if metric_type == "Confusion Matrix":
                            from sklearn.metrics import confusion_matrix
                            cm1 = confusion_matrix(ytest,pred)
                            st.write("Confusion Matrix is \t",cm1)
                            
                        elif metric_type == "Classification Report":
                            from sklearn.metrics import classification_report
                            cr = classification_report(ytest,pred)
                            st.write(cr)
                            
                        elif metric_type == "Precision Score":
                            from sklearn.metrics import precision_score
                            ps =precision_score(ytest,pred,average="micro")
                            st.write("Overall Precision Score is ",(ps*100))
                            
                        elif metric_type == "Recall Score":
                            from sklearn.metrics import recall_score
                            rs = recall_score(ytest,pred,average="micro")
                            st.write("Overall Recall Score is ",(rs*100))
                            
                        elif metric_type == "F1 Score":
                            from sklearn.metrics import f1_score
                            f1s =f1_score(ytest,pred,average="micro")
                            st.write("Overall F1 Score is ",(f1s*100))
                            
                        elif metric_type == "Accuracy Score":
                            #from sklearn.metrics import accuracy_score
                            acc1 = accuracy_score(ytest,pred)
                            st.write("Overall Accuracy Score is ",(acc1*100))
                        
                        st.write("Decision Tree Graph for Model:-")
                        fig = plt.figure(figsize=(15,10))
                        plot_tree(dtc,filled=True)
                        st.pyplot()
                        
                        st.write("Feature importance in model:-")
                        plt.barh(X.columns,dtc.feature_importances_)
                        st.pyplot()
                        
                    elif ml_type == "Random Forest Classification":
                        
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        criterion_type = st.selectbox("Select Criterion Type",["entropy", "gini", "log_loss"])
                        maxfeatures = st.selectbox("Select Max Feature Type",["sqrt", "log2"])
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        
                        from sklearn.ensemble import RandomForestClassifier
                        rfc = RandomForestClassifier(n_estimators=100,criterion = criterion_type,random_state=100,max_depth=5,max_features=maxfeatures)
                        model = rfc.fit(xtrain,ytrain)
                        pred = model.predict(xtest)
                        #acc = accuracy_score(xtrain,ytrain)
                        #st.write("Training accuracy is",acc)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["Confusion Matrix","Classification Report","Precision Score","Recall Score","F1 Score","Accuracy Score"])
                        if metric_type == "Confusion Matrix":
                            from sklearn.metrics import confusion_matrix
                            cm1 = confusion_matrix(ytest,pred)
                            st.write("Confusion Matrix is \t",cm1)
                            
                        elif metric_type == "Classification Report":
                            from sklearn.metrics import classification_report
                            cr = classification_report(ytest,pred)
                            st.write(cr)
                            
                        elif metric_type == "Precision Score":
                            from sklearn.metrics import precision_score
                            ps =precision_score(ytest,pred,average="micro")
                            st.write("Overall Precision Score is ",(ps*100))
                            
                        elif metric_type == "Recall Score":
                            from sklearn.metrics import recall_score
                            rs = recall_score(ytest,pred,average="micro")
                            st.write("Overall Recall Score is ",(rs*100))
                            
                        elif metric_type == "F1 Score":
                            from sklearn.metrics import f1_score
                            f1s =f1_score(ytest,pred,average="micro")
                            st.write("Overall F1 Score is ",(f1s*100))
                            
                        elif metric_type == "Accuracy Score":
                            #from sklearn.metrics import accuracy_score
                            acc1 = accuracy_score(ytest,pred)
                            st.write("Overall Accuracy Score is ",(acc1*100))
                        
                        st.write("Feature importance in model:-")
                        plt.barh(X.columns,rfc.feature_importances_)
                        st.pyplot()
                        
                    elif ml_type == "KNN Classification":
                        
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        weight_type = st.selectbox("Select Weight Type",["uniform", "distance"])
                        algorithm_type = st.selectbox("Select Algorithm Type",["auto", "ball_tree", "kd_tree", "brute"])
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        
                        from sklearn.neighbors import KNeighborsClassifier
                        knnc = KNeighborsClassifier(n_neighbors=5,weights=weight_type,algorithm=algorithm_type)
                        model = knnc.fit(xtrain,ytrain)
                        pred = model.predict(xtest)
                        #acc = accuracy_score(xtrain,ytrain)
                        #st.write("Training accuracy is",acc)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["Confusion Matrix","Classification Report","Precision Score","Recall Score","F1 Score","Accuracy Score"])
                        if metric_type == "Confusion Matrix":
                            from sklearn.metrics import confusion_matrix
                            cm1 = confusion_matrix(ytest,pred)
                            st.write("Confusion Matrix is \t",cm1)
                            
                        elif metric_type == "Classification Report":
                            from sklearn.metrics import classification_report
                            cr = classification_report(ytest,pred)
                            st.write(cr)
                            
                        elif metric_type == "Precision Score":
                            from sklearn.metrics import precision_score
                            ps =precision_score(ytest,pred,average="micro")
                            st.write("Overall Precision Score is ",(ps*100))
                            
                        elif metric_type == "Recall Score":
                            from sklearn.metrics import recall_score
                            rs = recall_score(ytest,pred,average="micro")
                            st.write("Overall Recall Score is ",(rs*100))
                            
                        elif metric_type == "F1 Score":
                            from sklearn.metrics import f1_score
                            f1s =f1_score(ytest,pred,average="micro")
                            st.write("Overall F1 Score is ",(f1s*100))
                            
                        elif metric_type == "Accuracy Score":
                            #from sklearn.metrics import accuracy_score
                            acc1 = accuracy_score(ytest,pred)
                            st.write("Overall Accuracy Score is ",(acc1*100))
                            
                    elif ml_type == "SVM Classification":
                        
                        all_columns_names = df.columns.tolist()
                        selected_columns_name_x = st.multiselect("Select dependent column",all_columns_names,key=1)
                        
                        X = df[selected_columns_name_x]
                        selected_columns_name_y = st.selectbox("Select Independent column",all_columns_names,key=2)
                        Y = df[selected_columns_name_y]
                        kernel_type = st.selectbox("Select Kernel Type",["linear", "poly", "rbf", "sigmoid", "precomputed"])
                        gamma_type = st.selectbox("Select Gamma Type",["scale", "auto"])
                        from sklearn.model_selection import train_test_split
                        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
                        
                        from sklearn.svm import SVC
                        svc1 = SVC(kernel=kernel_type,gamma=gamma_type)
                        model = svc1.fit(xtrain,ytrain)
                        pred = model.predict(xtest)
                        #acc = accuracy_score(xtrain,ytrain)
                        #st.write("Training accuracy is",acc)
                        
                        st.write("Please select a metric to test model:")
                        metric_type = st.selectbox("Select Type",["Confusion Matrix","Classification Report","Precision Score","Recall Score","F1 Score","Accuracy Score"])
                        if metric_type == "Confusion Matrix":
                            from sklearn.metrics import confusion_matrix
                            cm1 = confusion_matrix(ytest,pred)
                            st.write("Confusion Matrix is \t",cm1)
                            
                        elif metric_type == "Classification Report":
                            from sklearn.metrics import classification_report
                            cr = classification_report(ytest,pred)
                            st.write(cr)
                            
                        elif metric_type == "Precision Score":
                            from sklearn.metrics import precision_score
                            ps =precision_score(ytest,pred,average="micro")
                            st.write("Overall Precision Score is ",(ps*100))
                            
                        elif metric_type == "Recall Score":
                            from sklearn.metrics import recall_score
                            rs = recall_score(ytest,pred,average="micro")
                            st.write("Overall Recall Score is ",(rs*100))
                            
                        elif metric_type == "F1 Score":
                            from sklearn.metrics import f1_score
                            f1s =f1_score(ytest,pred,average="micro")
                            st.write("Overall F1 Score is ",(f1s*100))
                            
                        elif metric_type == "Accuracy Score":
                            #from sklearn.metrics import accuracy_score
                            acc1 = accuracy_score(ytest,pred)
                            st.write("Overall Accuracy Score is ",(acc1*100))       
                
            
            
                    

st.sidebar.title("Created By:")
st.sidebar.subheader("Manan Sharma")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/manan-sharma-785269151/)")
st.sidebar.subheader("[GitHub Repository](https://github.com/manansharma27/automated_exploratory_data_analysis)") 
                        
                    

if __name__ == '__main__':
	main()
                
            
        
