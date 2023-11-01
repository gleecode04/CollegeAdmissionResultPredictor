import pandas as pd
import csv
import random
cols = ["name","gpa","major","sat","act","ec","ld","admission_status"]
majors = ['Arts and Humanities', 'Natural Science', 'Public and Social Services', 'Social Science', 'Engineering', 'Medicine','Business']
stu_count = 0
#rows = len(data)
def get_three_numbers(s):
    valid_input = False
    sum = 0
    while not valid_input:
        sum = 0
        list = input(f"input number of low, medium , high {s}").split()
        print(list)
        if (len(list) != 3):
            print("input 3 numbers")
            continue
        try:
            intValues = [int(value) for value in list]
            print(intValues)
            for i in intValues:
                sum += i
        
            if all(value >= 0 for value in intValues) and sum == 10:
                valid_input = True
                return intValues
            else:
                print("input 3 whole numbers that add up to 10")
        except:
            print("input three positive integers")
    
def inputData():
    name = input("name")
    while True:
        gpa = float(input ("gpa"))
        gpa = round(gpa,2)
        if (gpa > 2.00 and gpa <= 4.50):
            break
        else:
            print("type gpa in the range of (2.00,4.50]")
    while True:
        major = input ("major(all lower case)")
        if major in majors:
            break
        else:
            print(f"please type a valid major: {majors}")
    test_status = ""
    while True:
        test_status = input("did the student take sat, act, or both? input one of (s,a,b)")
        if (test_status == 's' or test_status == 'a' or test_status== 'b'):
            break
    if (test_status == 's'):
        while True:
            sat_score = int(input ("sat"))
            if (sat_score > 0 and sat_score <= 1600 and sat_score % 10 == 0):
                act_score = None
                break
            else:
                print("please input a valid sat score")
    elif (test_status == 'a'):
        while True:
            act_score = int(input ("sat"))
            if (act_score > 0 and sat_score <= 36):
                sat_score = None
                break
            else:
                print("please input a valid sat score")
    else:
        while True:
            sat_score = int(input ("sat"))
            if (sat_score > 0 and sat_score <= 1600 and sat_score % 10 == 0):
                break
            else:
                print("please input a valid sat score")
        while True:
            act_score = int(input ("sat"))
            if (act_score > 0 and sat_score <= 36):
                break
            else:
                print("please input a valid sat score")
        
    ec = get_three_numbers("extracurriculars")
    ecScore = ec[2] * 8 + ec[1] * 6 + ec[0] * 1
    ld = get_three_numbers("leadership positions")
    ldScore = ld[2] * 5 + ld[1] * 3 + ld[0] * 1
    while True:
        admission_status = input("input a, w or r")
        if (admission_status == 'a' or admission_status == 'w' or admission_status == 'r'):
            break
    data =[name, gpa, major, sat_score, act_score, ecScore, ldScore,admission_status]
    addElement(data)

def addElement (data) :
    temp = pd.DataFrame(columns = ["name","gpa","major","sat","act","ec","ld","admission_status"])
    temp.to_csv('data2.csv',index = False)
    with open('data2.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

if __name__ == "__main__":
    inputData()
    data = pd.read_csv('data2.csv')
    print(data)
    #print(data['gpa'])
    
