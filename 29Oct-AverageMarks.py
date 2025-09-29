import pandas as pd

def average_marksbySubject(data):
    df = pd.DataFrame(data)
    subjects = df.drop(columns=['Name'])
    average_marks = subjects.mean().round(2).to_dict()
    return average_marks

#Sample data
data = {
    "Name":["Alice","Bob","Charlie"],
    "Math":[80,70,90],
    "Science":[85,75,95],
    "English":[78,82,88]
}

result = average_marksbySubject(data)
print(result)