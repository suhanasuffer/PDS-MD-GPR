import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

noise=1
len_scale=2.5
import math, time

def kernel_function(x1,x2,len_scale):
    dist_sq=np.linalg.norm(x1-x2)**2
    term=-1/(2*len_scale**2)
    return noise*np.exp(dist_sq*term)

def cov_matrix(x1,x2):
    n=x1.shape[0]
    m=x2.shape[0]
    cov_mat=np.empty((n,m))

    for i in range(n):
        for j in range(m):
            cov_mat[i][j]=kernel_function(x1[i], x2[j], len_scale)
    return cov_mat

def GPR_train(trainX): #def GPR_train(trainX, trainY):
    K=cov_matrix(trainX, trainX)
    K_inv=np.linalg.inv(K+noise*np.identity(len(trainX)))
    return K, K_inv

def GPR_predict(trainX, trainY, testX):
    K1=cov_matrix(trainX, trainX)
    K2=cov_matrix(testX, testX)
    K3= K2 - np.matmul(K1.T, np.matmul(K_inv, K1))+noise*np.identity(len(testX))

    mean_prediction = np.matmul(K1.T, np.matmul(K_inv, trainY))
    std_prediction = np.sqrt(np.diag(K3))

    return mean_prediction, std_prediction

trainX=np.linspace(0,10,num=1000)
trainY=trainX*np.sin(trainX)

testX=np.linspace(0,10,num=1000)
testY=testX*np.sin(testX)

print('Training started')
K,K_inv=GPR_train(trainX)
print('Training completed')

print('Testing started')
stt=time.time()
mean_prediction, std_prediction= GPR_predict(trainX, trainY, testX)
et=time.time() #prediction now

# Streamlit
st.title("Gaussian Process Regression Demo")

st.sidebar.header("Model Configuration")
input_len_scale = st.sidebar.number_input("Length Scale", min_value=0.1, max_value=10.0, value=2.5)
input_noise = st.sidebar.number_input("Noise Level", min_value=0.01, max_value=10.0, value=1.0)

if st.sidebar.button("Update Model"):
    len_scale = input_len_scale
    noise = input_noise
    K, K_inv = GPR_train(trainX)

st.header("Data and Predictions")
st.line_chart(trainX * np.sin(trainX), use_container_width=True)

if st.checkbox("Show Predictions"):
    st.write("Calculating predictions...")
    testX = np.linspace(0, 10, num=1000)
    mean_prediction, std_prediction = GPR_predict(trainX, trainY, testX)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(testX, testX * np.sin(testX), color='black', label='True Data')
    ax.plot(testX, mean_prediction, ls=':', lw=2, color='red', label='Mean Prediction')
    ax.fill_between(testX, mean_prediction - 1.96 * std_prediction, mean_prediction + 1.96 * std_prediction, alpha=0.5, label='95% Confidence Interval')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    st.pyplot(fig)
