from django.db import models

class Product(models.Model):
    product_name = models.CharField(max_length=255)
    positive_url = models.CharField(max_length=255)
    category = models.CharField(max_length=100)
    stock = models.IntegerField()
    price = models.FloatField()

    class Meta:
     db_table = "product"
    
