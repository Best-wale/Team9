
from django.db import models

class ProductImage(models.Model):
    image = models.ImageField(upload_to='product_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"Image {self.id} uploaded at {self.uploaded_at}"
    
