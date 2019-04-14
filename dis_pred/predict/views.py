from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
# from .models import Patient, Doctor, Chat, Data
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.db import IntegrityError
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#import pickle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from predict.models import *
import numpy as np
import itertools
import sys
import time
import csv
import pprint
import xlrd

@csrf_exempt
def login(request):
	if request.method == "POST":
		username = request.POST.get("username")
		password = request.POST.get("password")
		# request.session["login"] = True
		user = User.objects.filter(username=username)
		# print("HIIII")
		if not user.exists():
			user = User(username = username, password = password)
			user.save()
			return JsonResponse({"status":"user_created"})
			# print(username)
		else:
			# print(user)
			if user[0].password != password:
				return JsonResponse({"status":"wrong_password"})
			return JsonResponse({"status":"success"})

	return JsonResponse({"status":"error"})

def tokenize(file_name):
    return [sorted(list(set(e.split(",")))) for e in
            open(file_name).read().strip(",").split('\n')]

def frequent_itemsets(sentences):
    # Counts sets with Apriori algorithm.
    SUPP_THRESHOLD = 100
    supps = []
 
    supp = {}
    for sentence in sentences:
        for key in sentence:
            if key in supp:
                supp[key] += 1
            else:
                supp[key] = 1
    #print ("|C1| = " + str(len(supp)))
    supps.append({k:v for k,v in supp.items() if v >= SUPP_THRESHOLD})
    #print ("|L1| = " + str(len(supps[0])))
 
    supp = {}
    for sentence in sentences:
        for combination in itertools.combinations(sentence, 2):
            if combination[0] in supps[0] and combination[1] in supps[0]:
                key = ','.join(combination)
                if key in supp:
                    supp[key] += 1
                else:
                    supp[key] = 1
    #print ("|C2| = " + str(len(supp)))
    supps.append({k:v for k,v in supp.items() if v >= SUPP_THRESHOLD})
    #print ("|L2| = " + str(len(supps[1])))
 
    supp = {}
    for sentence in sentences:
        for combination in itertools.combinations(sentence, 3):
            if (combination[0]+','+combination[1] in supps[1] and
                    combination[0]+','+combination[2] in supps[1] and
                    combination[1]+','+combination[2] in supps[1]):
                key = ','.join(combination)
                if key in supp:
                    supp[key] += 1
                else:
                    supp[key] = 1
    #print ("|C3| = " + str(len(supp)))
    supps.append({k:v for k,v in supp.items() if v >= SUPP_THRESHOLD})
    #print ("|L3| = " + str(len(supps[2])))
 
    return supps

def generate_rules(measure, supps, transaction_count):
    rules = []
    CONF_THRESHOLD = 0.8
    LIFT_THRESHOLD = 20.0
    CONV_THRESHOLD = 5.0
    if measure == 'conf':
        for i in range(2, len(supps)+1):
            for k,v in supps[i-1].items():
                k = k.split(',')
                for j in range(1, len(k)):
                    for a in itertools.combinations(k, j):
                        b = tuple([w for w in k if w not in a])
                        [conf, lift, conv] = measures(v,
                                supps[len(a)-1][','.join(a)],
                                supps[len(b)-1][','.join(b)],
                                transaction_count)
                        if conf >= CONF_THRESHOLD:
                            if (len(a)<2):
                                #a = str(a).replace(',','')
                                rules.append((a, b, conf, lift, conv))
            rules = sorted(rules, key=lambda x: (x[0], x[1]))
            rules = sorted(rules, key=lambda x: (x[2]), reverse=True)
    return rules

def find_unique(rule_dictionary):
    new_rule_dict = dict()
    for key, value in rule_dictionary.items():
        _unique = set()
        for x in value:
            if type(x) == tuple:
                for y in x:
                    _unique.add(y)
            else:
                _unique.add(x)
        new_rule_dict[key] = _unique

    return new_rule_dict

def measures(supp_ab, supp_a, supp_b, transaction_count):
    # Assumes A -> B, where A and B are sets.
    conf = float(supp_ab) / float(supp_a)
    s = float(supp_b) / float(transaction_count)
    lift = conf / s
    if conf == 1.0:
        conv = float('inf')
    else:
        conv = (1-s) / (1-conf)
    return [conf, lift, conv]


def market_basket(file,measure):
    dataset = pd.read_csv("predict/Manual-Data/UpdatedTraining.csv")
    dictionary = {}
    count = 0
    for row in dataset.iterrows():
        dictionary[count] = dataset.columns[row[1]==1]
        count+=1
    #print(dictionary)
    w = csv.writer(open("predict/Manual-Data/list.csv", "w",newline=''))
    for key, val in dictionary.items():
        my_string = val.values
        #my_string[-1] = my_string[-1].strip(",")
        w.writerow(my_string.tolist())
        
    sentences=tokenize(file)
    supps = frequent_itemsets(sentences)
 
    rules = generate_rules(measure, supps, len(sentences))
    no_of_rules = 0
    rules_dictionary = {}
    for rule in rules:
        '''
        print (("{{{}}} -> {{{}}}, "
           "conf = {:.2f}, lift = {:.2f}, conv = {:.2f}").format(
          ', '.join(rule[0]), ', '.join(rule[1]), rule[2], rule[3], rule[4]))
        '''
        #print(rule[1])
        temp = str(rule[0]).replace(',', '')
        
        try:
            if rules_dictionary[temp]:
                obj1=list(rules_dictionary[temp])
                obj1.append(rule[1])
                rules_dictionary[temp] = tuple(obj1)
        except KeyError:
            rules_dictionary[temp]=rule[1]
            
        no_of_rules+=1
    
    #print(no_of_rules)
    
    new_rule_dict = find_unique(rules_dictionary)
    #pprint.pprint(new_rule_dict)
    #print(len(rules_dictionary)) 
    
    with open('predict/Manual-Data/rules.csv', 'w') as f:
        for key in new_rule_dict.keys():
            f.write("%s,%s\n"%(key,new_rule_dict[key]))

@csrf_exempt
def train_model(request):
	if request.method == 'POST':
		# print(request.GET("key"))
		data = pd.read_csv("predict/Manual-Data/Training.csv")
		array=['Migraine','Tuberculosis','Peptic ulcer diseae','Bronchial Asthma','Chicken pox','Heart attack','Impetigo','Urinary tract infection','Dengue','Hypertension ','Hyperthyroidism']
		df = pd.DataFrame(data)
		df=df.loc[df['prognosis'].isin(array)]
		df=df.loc[:, (df != 0).any(axis=0)]

		df.to_csv('predict/Manual-Data/UpdatedTraining.csv')
		cols = df.columns
		cols = cols[:-1]
		x = df[cols]
		y = df['prognosis']
		
		dt = DecisionTreeClassifier()
		clf_dt=dt.fit(x, y)
		
		#pred_pro= dt.predict_proba(x)
		#print(pred_pro)
		
		filename = 'predict/finalized_model.pkl'
		joblib.dump(clf_dt, filename)

		measure = 'all'	
		file = "predict/Manual-Data/list.csv"
		market_basket(file, "conf")

		response_data = {}
		# response_data['scores'] = scores
		# response_data['mean_score'] = scores.mean()  
		return JsonResponse(response_data)


@csrf_exempt
def predict_symptoms(request):
    if request.method == "POST":
        rules_dict = {}
        # existing_symptoms = request.POST.get("symptoms") #existing symptoms from android
        existing_symptoms = ["abnormal_menstruation", "back_pain", "blood_in_sputum"]
        with open('predict/Manual-Data/rules.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                # if line_count < 3:
                #     print(row)
                # line_count += 1
                key = row[0].strip("'()")
                values = []
                for i in range(1, len(row)):
                    values.append(row[i].strip().strip("{}'"))
                rules_dict[key] = values
        #print(rules_dict)
        predicted_symptoms = list()

        for symptom in existing_symptoms:
            predicted_symptoms.append(rules_dict[symptom])

        intersection_symptoms = set(predicted_symptoms[0]).intersection(*predicted_symptoms)
        print(intersection_symptoms)


    return JsonResponse({})


@csrf_exempt
def generate_prescription(request):
    if request.method == "POST":
        #user = User.objects.filter(username =  request.POST.get("username"))[0]
        #disease_predicted = request.POST.get("symptoms") #edit using stringify
        #intensity = request.POST.get("intensity")
        #phase = request.POST.get("phase")

        #weight = User.objects.get(user)
        #age = User.objects.get(age)
        #disease=disease_predicted

        loc = ("predict/Manual-Data/Prescriptions.xlsx") 
        disease="Dengue"
        age=""
        intensity="NA"
        phase="Continuation"
        weight=60
        m=[]
        wb = xlrd.open_workbook(loc) 
        sheet = wb.sheet_by_index(0) 
          
        # For row 0 and column 0 
        sheet.cell_value(0, 0) 

        for i in range(1,sheet.nrows):
            if((sheet.cell_value(i, 0)==disease) and (sheet.cell_value(i, 1)==age or sheet.cell_value(i, 1)=="NA") and (sheet.cell_value(i, 2)==intensity or sheet.cell_value(i, 2)=="NA") and (sheet.cell_value(i, 3)==phase or sheet.cell_value(i, 3)=="NA")):
                if(sheet.cell_value(i, 4)==0):
                    for j in range(5,sheet.ncols):
                        if not(sheet.cell_value(i,j)=="-"):
                            m.append(sheet.cell_value(i, j))
                        else:
                            break
                else:
                    for j in range(5,sheet.ncols):
                        if not(sheet.cell_value(i,j)=="-"):
                            if not(j%3==0):
                                m.append(sheet.cell_value(i, j))
                            else:
                                m.append(weight*sheet.cell_value(i, j))
                        else:
                            break
        print(m)
        return JsonResponse({"result": m})
    



@csrf_exempt
def predict_diseases(request):
	if request.method == "POST":
		# user = User.objects.filter(username =  request.POST.get("username"))[0]
		# symptoms = request.POST.get("symptoms") #edit using stringify
		symptoms = [[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
		filename = 'predict/finalized_model.pkl'
		# df = pd.DataFrame(symptoms)
		model = joblib.load(filename)
		disease_predicted = model.predict(symptoms)
		print(disease_predicted)
		return JsonResponse({"result": disease_predicted})


@csrf_exempt
def add_user_details(request):
	if request.method == "POST":
		username = request.POST.get("username")
		user = User.objects.filter(username=username)
		print(user)
		if not user.exists():
			return JsonResponse({"status":"error"})

		else:
			user = user[0]
			user.height = request.POST.get("height")
			user.weight = request.POST.get("weight")
			user.blood_group = request.POST.get("blood_group")
			user.age = request.POST.get("age")
			user.gender = request.POST.get("gender")
			user.new_user = 0
			user.save()
			return JsonResponse({"status":"success"})

		return JsonResponse({"status":"error2"})

	if request.method == "GET":
		return HttpResponse("hi")