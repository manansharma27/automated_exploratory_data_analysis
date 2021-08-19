
#Importing important libraries
import streamlit as st 
import pandas as pd  
import base64
import time
st.set_option('deprecation.showfileUploaderEncoding', False)

timestr = time.strftime("%Y%m%d-%H%M%S")   #Making a variable taking current date,month and year

# Data Viz Pkg
import matplotlib.image as img 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sb 

#Reading image and storing it
image = img.imread("C:/Users/Manan/eda.png",0)
#Displaying Image
st.image(image)

# Making a function to download csv file.
def csv_download(data):
    csvfile = data.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "new_file_{}.csv".format(timestr)
    st.markdown("Download CSV File from below.")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}"> Download Here!!</a>'
    st.markdown(href,unsafe_allow_html=True)

#Making the main function
def main():
    """ Automated Exploratory Data Analysis App """
    st.header("Automated Data Analysis App")
    task = ['Data Exploration','EDA','Fill Null Values']
    choice = st.sidebar.selectbox("Select Options",task)   #Making a sidebar for selection of activity
    
    if(choice == 'Data Exploration'):
        st.subheader("Basic Data Exploration!")
        
        data = st.file_uploader("Upload a dataset to analysis",type=["csv","txt","xlsx"])
        if data is not None:
            df = pd.read_csv(data,encoding='utf-8',engine='python')  #Reading the csv file
            st.dataframe(df.head())
            
            st.write("Stastistics of dataset")
            st.write(df.describe())
            st.write("Shape")
            st.write(df.shape)
            st.write("Datatype of columns")
            st.write(df.dtypes)
                
            if st.checkbox("Show Columns"):   
                st.write(df.columns)    #To show columns of the dataset
            
            if st.checkbox("Show Selected Columns"):
                all_columns = df.columns.to_list()    #Taking all columns in a list
                selected_columns = st.multiselect("Select Columns",all_columns,key=0)  #Making a dropdown to select columns
                new_df = df[selected_columns]
                st.dataframe(new_df)
                
                
            if st.checkbox("Null Values"):
                st.write(df.isnull().sum())     #Checking Null values
                
            if st.checkbox("Visualize Null Values"):
                st.write(sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis'))      #Visulaizing null values
                st.pyplot()
                
            if st.checkbox("Show Value counts of column/ columns"):
                column = df.columns.to_list()
                selected_column = st.selectbox("Select Column",column,key=1)
                st.write(df[selected_column].value_counts())         #Checking value counts in selected column
                
            if st.checkbox("Show Correlation Map of Columns"):
                st.write(sb.heatmap(df.corr(),annot=True))      #Checking correlation using heatmap
                st.pyplot()
                
            if st.checkbox("Check Outliers in columns"):
                all_columns3 = df.columns.to_list()
                selected_columns3 = st.multiselect("Select Columns",all_columns3,key=2)
                new_df3 = df[selected_columns3]
                st.write(sb.boxplot(new_df3))      #Checking Outliers using boxplot
                st.pyplot()
                
    elif choice == 'EDA':
        st.subheader("Data Visualization")
        data= st.file_uploader("Upload a dataset",type=["csv","txt","xlsx"])
        if data is not None:
            df = pd.read_csv(data,encoding = 'utf-8',engine='python')
            st.dataframe(df.head())
            
            if st.checkbox("Show Datatypes of columns"):
                st.write(df.dtypes)
                
                all_columns_names = df.columns.tolist()
                #Making a dropdown option to choose the type of plot to generate
                type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde","pie"])
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
                        
                     #Custom Plot   
                    elif type_of_plot:
                        plot = df[selected_columns_names].plot(kind=type_of_plot)
                        st.write(plot)
                        st.pyplot()
    
    elif choice == 'Fill Null Values':
        st.subheader("Fill Null Values")
        data= st.file_uploader("Upload a dataset",type=["csv","txt","xlsx"])
        if data is not None:
            df = pd.read_csv(data,encoding = 'utf-8',engine='python')
            st.dataframe(df.head())
            
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
                
            
        