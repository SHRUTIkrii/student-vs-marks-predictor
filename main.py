import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "student" : ['A','B','C','D','E','F','G'],
    "hours" : [2, 2.5, 3, 4, 4.5, 6, 1],
    "marks" : [20, 29, 40, 58, 67, 80, 5]
}
df=pd.DataFrame(data) 

df.to_csv('data.csv',index=False)
df = pd.read_csv("data.csv")

from sklearn.model_selection import train_test_split

x= df[["hours"]]
y= df[["marks"]]

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train , y_train)

y_pred = model.predict(x_test)

hours = float(input("Enter study hours: "))
predicted_marks = model.predict([[hours]])
print("Predicted Marks:", predicted_marks[0])

plt.plot(df["hours"],df["marks"],linewidth=1,marker='*',color='red')
plt.title("STUDY HOUR vs MARKS PRIDICTION",fontsize=14,color='green')
plt.xlabel("Hours",fontsize=14)
plt.ylabel("Marks",fontsize=14)
plt.grid(True)
plt.savefig('study_VS_marks.png')
plt.show()