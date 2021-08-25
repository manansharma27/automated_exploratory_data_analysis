
#Importing important libraries
import streamlit as st 
import pandas as pd  
import base64
import io
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)



# Data Viz Pkg
import matplotlib.image as img 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sb 

#Reading image and storing it
image = img.imread("eda.png")
#Displaying Image
st.image(image)

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
    task = ['Data Exploration','EDA','Fill Null Values']
    choice = st.sidebar.selectbox("Select Options",task)   #Making a sidebar for selection of activity
    
    if(choice == 'Data Exploration'):
        st.subheader("Basic Data Exploration!")
        
        data = st.file_uploader("Upload a dataset to analysis",type=["csv"])
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
                st.write(sb.heatmap(df.corr(),annot=True))      #Checking correlation using heatmap
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
        st.subheader("Data Visualization")
        data= st.file_uploader("Upload a dataset",type=["csv"])
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
                type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde","pie","pairplot"])
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
                        
                        
                     #Custom Plot   
                    elif type_of_plot:
                        plot = df[selected_columns_names].plot(kind=type_of_plot)
                        st.write(plot)
                        st.pyplot()
    
    elif choice == 'Fill Null Values':
        st.subheader("Fill Null Values")
        data= st.file_uploader("Upload a dataset",type=["csv"])
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
                    

st.sidebar.title("Created By:")
st.sidebar.subheader("Manan Sharma")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/manan-sharma-785269151/)")
st.sidebar.subheader("[GitHub Repository](https://github.com/manansharma27/automated_exploratory_data_analysis)") 
                        
                    

if __name__ == '__main__':
	main()
                
            
        