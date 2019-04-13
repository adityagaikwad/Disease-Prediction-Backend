from django.db import models

# class Doctor(models.Model):
#     patients = models.CharField(max_length = 100, default = '') #comma separated id's
#     pending_requests = models.CharField(max_length = 100, default = '') #comma separated id's
#     first_name = models.CharField(max_length = 50, default = '')
#     last_name = models.CharField(max_length = 50, default = '')
#     email = models.EmailField(unique = True, default = '')
#     password = models.CharField(max_length = 50, default = '')
#     age = models.IntegerField(default = 0)
#     experience = models.IntegerField(default = 0)
#     qualification = models.TextField(default = '')
#     address = models.TextField(default = '')
#     number = models.IntegerField(default = 0)
#     fees = models.IntegerField(default = 0)
#     gender = models.CharField(max_length = 50, default = '')

class User(models.Model):
	username = models.CharField(max_length = 100, default = '', null = True)
	password = models.CharField(max_length = 100, default = '', null = True)
	new_user = models.IntegerField(default = 1)
	height = models.IntegerField(default=1)
	weight = models.IntegerField(default=1)
	blood_group = models.CharField(max_length = 100, default = '', null = True)
	age = models.IntegerField(default=1)
	gender = models.CharField(max_length = 100, default = '', null = True)
	
#     def __str__(self):
#         return self.email

#     class Meta:
#         db_table = "doctor"

	
