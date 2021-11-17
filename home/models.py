from django.db import models
# Create your models here.
class ProductModel(models.Model):
    class Meta:
        db_table = "product"
    product_url = models.CharField(max_length=256, default='')
    product_name = models.CharField(max_length=256, default='')
    product_num = models.IntegerField(null=True)
    categories = models.TextField(null=True)
    img_src = models.CharField(max_length=256, default='')
    price = models.CharField(max_length=32, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    total_value = models.FloatField(null=True, default=0)
    pos_neg_rate = models.IntegerField(null=True, default=0)
    search_value = models.IntegerField(null=True)


class ReviewModel(models.Model):
    class Meta:
        db_table = "review"
    product_id = models.ForeignKey(ProductModel, on_delete=models.CASCADE,default='')
    review = models.CharField(max_length=256)
    score = models.FloatField(null=True, default=0)
    keywords = models.CharField(max_length=256)
    morph = models.CharField(max_length=256, default='')
    xai_vale = models.CharField(max_length=256, default='')
    date = models.CharField(max_length=256, default='')

class ProductKeyword(models.Model):
    class Meta:
        db_table = "PDkeyword"
    product_id = models.ForeignKey(ProductModel, on_delete=models.CASCADE, default='')
    keyword = models.CharField(max_length=20, default='')
    summarization = models.CharField(max_length=256, default='')
    keyword_positive = models.FloatField(default=60)